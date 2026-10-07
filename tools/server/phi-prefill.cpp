// llama.phi: the prefill engine (phi-prefill.h).
#include "phi-prefill.h"

#include "log.h"

#include <cstdlib>
#include <cstring>
#include <list>
#include <map>

static const char * env_str(const char * name) {
    const char * v = getenv(name);
    return (v && *v) ? v : nullptr;
}

static int env_int(const char * name, int def) {
    const char * v = env_str(name);
    return v ? atoi(v) : def;
}

// "pattern=buft,...", as common's -ot (its parser is private to arg.cpp).
static bool parse_overrides(const std::string & value, std::vector<llama_model_tensor_buft_override> & out) {
    std::map<std::string, ggml_backend_buffer_type_t> bufts;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * dev  = ggml_backend_dev_get(i);
        auto * buft = ggml_backend_dev_buffer_type(dev);
        if (buft) {
            bufts[ggml_backend_buft_name(buft)] = buft;
        }
    }
    static std::list<std::string> kept;   // the patterns outlive the model
    for (const auto & one : string_split<std::string>(value, ',')) {
        auto pos = one.find('=');
        if (pos == std::string::npos) {
            LOG_ERR("phi-prefill: PHI_PREFILL_OT entry without '=': %s\n", one.c_str());
            return false;
        }
        std::string pattern = one.substr(0, pos);
        std::string buft    = one.substr(pos + 1);
        auto it = bufts.find(buft);
        if (it == bufts.end()) {
            LOG_ERR("phi-prefill: unknown buffer type %s in PHI_PREFILL_OT\n", buft.c_str());
            return false;
        }
        kept.push_back(pattern);
        out.push_back({kept.back().c_str(), it->second});
    }
    return true;
}

bool phi_prefill::init(const common_params & base) {
    enabled = env_int("PHI_PREFILL", 0) != 0;
    if (!enabled) {
        return true;
    }
    common_params p = base;          // the decode side's settings, then the prefill's own
    p.warmup = false;
    p.n_batch  = env_int("PHI_PREFILL_BATCH",  4096);
    p.n_ubatch = env_int("PHI_PREFILL_UBATCH", 2048);
    if (const char * v = env_str("PHI_PREFILL_NGL")) {
        p.n_gpu_layers = atoi(v);
    }
    if (const char * v = env_str("PHI_PREFILL_THREADS")) {
        p.cpuparams.n_threads       = atoi(v);
        p.cpuparams_batch.n_threads = atoi(v);
    }
    if (const char * v = env_str("PHI_PREFILL_KV_HOST")) {
        p.no_kv_offload = atoi(v) != 0;
    }
    if (const char * v = env_str("PHI_PREFILL_TS")) {
        std::fill(std::begin(p.tensor_split), std::end(p.tensor_split), 0.0f);
        size_t i = 0;
        for (const auto & s : string_split<std::string>(v, '/')) {
            if (i < std::size(p.tensor_split)) {
                p.tensor_split[i++] = std::stof(s);
            }
        }
    }
    if (const char * v = env_str("PHI_PREFILL_OT")) {
        p.tensor_buft_overrides.clear();
        if (!parse_overrides(v, p.tensor_buft_overrides)) {
            return false;
        }
        // common's parser pads the list with null entries up to the
        // library's maximum (llama_params_fit reads it so); the conversion
        // asserts that shape
        const size_t ntbo = llama_max_tensor_buft_overrides();
        while (p.tensor_buft_overrides.size() < ntbo) {
            p.tensor_buft_overrides.push_back({nullptr, nullptr});
        }
    }
    min_tokens = (size_t) env_int("PHI_PREFILL_MIN", 64);
    n_ubatch   = p.n_ubatch;
    // The prefill model's devices: those whose name starts with
    // PHI_PREFILL_DEVICES (default "CUDA"), so a backend meant for the
    // decode side alone (the cards' libggml_phi.so, whose glue keeps
    // per-graph tables that two contexts must not share) never sees this
    // model; the CPU is implicit.
    {
        const char * want = env_str("PHI_PREFILL_DEVICES");
        const std::string prefix = want ? want : "CUDA";
        p.devices.clear();
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto * dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                continue;
            }
            const std::string name = ggml_backend_dev_name(dev);
            if (name.compare(0, prefix.size(), prefix) == 0) {
                p.devices.push_back(dev);
            }
        }
        if (p.devices.empty()) {
            LOG_ERR("phi-prefill: no device named %s*\n", prefix.c_str());
            return false;
        }
    }

    size_t n_overrides = 0;
    for (const auto & o : p.tensor_buft_overrides) {
        n_overrides += o.pattern != nullptr;
    }
    LOG_INF("phi-prefill: loading the prefill model (batch %d, ubatch %d, ngl %d, KV %s, %zu overrides)\n",
            p.n_batch, p.n_ubatch, p.n_gpu_layers, p.no_kv_offload ? "host" : "device", n_overrides);
    llama_init = common_init_from_params(p);
    model = llama_init->model();
    ctx   = llama_init->context();
    if (!model || !ctx) {
        LOG_ERR("phi-prefill: could not load the prefill model\n");
        enabled = false;
        return false;
    }
    n_ctx = (int32_t) llama_n_ctx(ctx);
    worker = std::thread([this] { run(); });
    LOG_INF("phi-prefill: ready, n_ctx %d\n", n_ctx);
    return true;
}

void phi_prefill::shutdown() {
    if (!enabled) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(mtx);
        stop = true;
    }
    cv.notify_all();
    if (worker.joinable()) {
        worker.join();
    }
    llama_init.reset();
    model = nullptr;
    ctx   = nullptr;
}

bool phi_prefill::wants(const server_tokens & tokens, const server_prompt_cache * cache, int lcp_slots) const {
    if (!enabled || !cache) {
        return false;
    }
    if (tokens.has_mtmd || tokens.size() < min_tokens || (int32_t) tokens.size() >= n_ctx) {
        return false;
    }
    // Mostly in a slot already (a chat's next turn): the server's own reuse is cheaper.
    if (lcp_slots > 0 && (size_t) lcp_slots * 2 >= tokens.size()) {
        return false;
    }
    // Mostly in the cache already.
    for (const auto & st : cache->states) {
        const size_t lcp = st.prompt.tokens.get_common_prefix(tokens);
        if (lcp * 2 >= tokens.size()) {
            return false;
        }
    }
    return true;
}

void phi_prefill::submit(int id_task, const llama_tokens & tokens) {
    {
        std::lock_guard<std::mutex> lk(mtx);
        waiting.insert(id_task);
        queue.push_back({id_task, tokens});
    }
    cv.notify_one();
}

bool phi_prefill::is_waiting(int id_task) {
    std::lock_guard<std::mutex> lk(mtx);
    return waiting.count(id_task) > 0;
}

bool phi_prefill::is_ready(int id_task) {
    std::lock_guard<std::mutex> lk(mtx);
    return finished.count(id_task) > 0;
}

size_t phi_prefill::drain(server_prompt_cache & cache) {
    std::deque<done> got;
    {
        std::lock_guard<std::mutex> lk(mtx);
        got.swap(ready);
        for (const auto & d : got) {
            waiting.erase(d.id_task);
            finished.erase(d.id_task);
        }
    }
    for (auto & d : got) {
        if (d.state.empty()) {
            continue;   // a failed prefill: the server computes it itself
        }
        server_prompt prompt;
        d.tokens.pop_back();   // the entry covers what the state holds: all but the last token
        prompt.tokens = server_tokens(d.tokens, false);
        auto * st = cache.alloc(prompt, d.state.size(), 0);
        if (!st) {
            LOG_WRN("phi-prefill: the prompt cache refused a state of %zu bytes (raise --cache-ram)\n", d.state.size());
            continue;
        }
        memcpy(st->data.main.data(), d.state.data(), d.state.size());
    }
    return got.size();
}

bool phi_prefill::compute(job & j, std::vector<uint8_t> & state) {
    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    llama_batch batch = llama_batch_init(n_ubatch, 0, 1);
    bool ok = true;
    // All but the last token: the slot computes that one itself for its
    // logits. (Upstream rewinds one position for it when the whole prompt
    // is in place, and a recurrent layer cannot rewind, which made the slot
    // redo the prompt.) The state and the cache entry hold n - 1 tokens.
    const size_t n = j.tokens.size() - 1;
    for (size_t i = 0; i < n && ok; i += n_ubatch) {
        const int32_t k = (int32_t) std::min((size_t) n_ubatch, n - i);
        batch.n_tokens = k;
        for (int32_t t = 0; t < k; t++) {
            batch.token[t]     = j.tokens[i + t];
            batch.pos[t]       = (llama_pos) (i + t);
            batch.n_seq_id[t]  = 1;
            batch.seq_id[t][0] = 0;
            batch.logits[t]    = 0;   // the server recomputes the last token for its logits
        }
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("phi-prefill: decode failed at token %zu of %zu (task %d)\n", i, n, j.id_task);
            ok = false;
        }
    }
    llama_batch_free(batch);
    if (!ok) {
        return false;
    }
    const size_t size = llama_state_seq_get_size_ext(ctx, 0, 0);
    state.resize(size);
    const size_t got = llama_state_seq_get_data_ext(ctx, state.data(), size, 0, 0);
    if (got != size) {
        LOG_ERR("phi-prefill: state export gave %zu of %zu bytes (task %d)\n", got, size, j.id_task);
        return false;
    }
    return true;
}

void phi_prefill::run() {
    for (;;) {
        job j;
        {
            std::unique_lock<std::mutex> lk(mtx);
            cv.wait(lk, [this] { return stop || !queue.empty(); });
            if (stop) {
                return;
            }
            j = std::move(queue.front());
            queue.pop_front();
        }
        const int64_t t0 = ggml_time_us();
        std::vector<uint8_t> state;
        const bool ok = compute(j, state);
        const int64_t dt = ggml_time_us() - t0;
        LOG_INF("phi-prefill: task %d: %zu tokens in %.2f s (%.1f tok/s), state %.1f MiB%s\n",
                j.id_task, j.tokens.size(), dt / 1e6, j.tokens.size() * 1e6 / std::max<int64_t>(dt, 1),
                state.size() / 1048576.0, ok ? "" : " FAILED");
        {
            std::lock_guard<std::mutex> lk(mtx);
            n_prompts++;
            n_tokens += (int64_t) j.tokens.size();
            t_prefill_us += dt;
            finished.insert(j.id_task);
            ready.push_back({j.id_task, std::move(j.tokens), ok ? std::move(state) : std::vector<uint8_t>()});
        }
        if (on_done) {
            on_done();
        }
    }
}
