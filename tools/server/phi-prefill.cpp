// llama.phi: the prefill engine (phi-prefill.h).
#include "phi-prefill.h"

#include "log.h"

#include <cpp-httplib/httplib.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
    min_tokens = (size_t) env_int("PHI_PREFILL_MIN", 64);
    n_ctx      = (int32_t) base.n_ctx;
    if (const char * v = env_str("PHI_PREFILL_URL")) {
        // the client form: the prefill server does the prompts
        url = v;
        const char * d = env_str("PHI_PREFILL_DIR");
        if (!d) {
            LOG_ERR("phi-prefill: PHI_PREFILL_URL needs PHI_PREFILL_DIR, the prefill server's --slot-save-path\n");
            return false;
        }
        dir = d;
        if (!dir.empty() && dir.back() != '/') {
            dir += '/';
        }
        timeout_s = env_int("PHI_PREFILL_TIMEOUT", 3600);
        httplib::Client cli(url);
        cli.set_connection_timeout(5, 0);
        auto res = cli.Get("/health");
        if (!res || res->status != 200) {
            LOG_WRN("phi-prefill: the prefill server at %s does not answer /health yet (%s); prompts go to it when it does\n",
                    url.c_str(), res ? std::to_string(res->status).c_str() : httplib::to_string(res.error()).c_str());
        }
        worker = std::thread([this] { run(); });
        LOG_INF("phi-prefill: ready, prompts of %zu tokens or more go to %s, states read from %s\n", min_tokens, url.c_str(), dir.c_str());
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
        submitted.insert(id_task);
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

bool phi_prefill::was_submitted(int id_task) {
    std::lock_guard<std::mutex> lk(mtx);
    return submitted.count(id_task) > 0;
}

void phi_prefill::forget(int id_task) {
    std::lock_guard<std::mutex> lk(mtx);
    submitted.erase(id_task);
}

// The state a job will hold is its tokens but the last; it covers the task
// when that is a prefix of the task's tokens longer than what any slot holds.
static bool job_covers(const llama_tokens & job, const server_tokens & tokens, size_t lcp_known) {
    if (job.size() < 2) {
        return false;
    }
    const size_t n = job.size() - 1;
    if (n <= lcp_known || n > tokens.size()) {
        return false;
    }
    for (size_t i = 0; i < n; i++) {
        if (job[i] != tokens[i]) {
            return false;
        }
    }
    return true;
}

bool phi_prefill::covers(const server_tokens & tokens, size_t lcp_known) {
    std::lock_guard<std::mutex> lk(mtx);
    if (busy && job_covers(current.tokens, tokens, lcp_known)) {
        return true;
    }
    for (const auto & j : queue) {
        if (job_covers(j.tokens, tokens, lcp_known)) {
            return true;
        }
    }
    return false;
}

int phi_prefill::submit_prefetch(const llama_tokens & tokens, std::string & err) {
    if (!enabled) {
        err = "the prefill engine is off (PHI_PREFILL=1 starts it)";
        return -1;
    }
    if (tokens.size() < 2) {
        err = "a prefetch needs at least 2 tokens (the state holds all but the last)";
        return -1;
    }
    if (n_ctx > 0 && (int32_t) tokens.size() >= n_ctx) {
        err = "the prompt does not fit the context (" + std::to_string(tokens.size()) + " of " + std::to_string(n_ctx) + " tokens)";
        return -1;
    }
    int id;
    {
        std::lock_guard<std::mutex> lk(mtx);
        id = next_prefetch_id++;
        // the records are bounded: the oldest settled one goes first, then the oldest
        while (prefetches.size() >= 64) {
            auto victim = prefetches.end();
            for (auto it = prefetches.begin(); it != prefetches.end(); ++it) {
                if (it->second.status == PREFETCH_DONE || it->second.status == PREFETCH_FAILED) {
                    victim = it;
                    break;
                }
            }
            prefetches.erase(victim == prefetches.end() ? prefetches.begin() : victim);
        }
        prefetches[id] = {tokens, PREFETCH_WAITING, ""};
        waiting.insert(id);
        queue.push_back({id, tokens});
    }
    cv.notify_one();
    LOG_INF("phi-prefill: prefetch %d: %zu tokens to the prefill engine\n", id, tokens.size());
    return id;
}

std::string phi_prefill::prefetch_state(int id, std::string * reason) {
    std::lock_guard<std::mutex> lk(mtx);
    auto it = prefetches.find(id);
    if (it == prefetches.end()) {
        return "";
    }
    if (reason) {
        *reason = it->second.reason;
    }
    switch (it->second.status) {
        case PREFETCH_WAITING: return finished.count(id) > 0 ? "ready" : "waiting";
        case PREFETCH_READY:   return "ready";
        case PREFETCH_DONE:    return "done";
        case PREFETCH_FAILED:  return "failed";
    }
    return "";
}

void phi_prefill::audit(const server_prompt_cache & cache) {
    std::lock_guard<std::mutex> lk(mtx);
    for (auto & [id, rec] : prefetches) {
        if (rec.status != PREFETCH_READY) {
            continue;
        }
        const size_t n = rec.tokens.size() - 1;
        bool present = false;
        for (const auto & st : cache.states) {
            if (st.prompt.tokens.size() != n) {
                continue;
            }
            size_t i = 0;
            while (i < n && st.prompt.tokens[i] == rec.tokens[i]) {
                i++;
            }
            if (i == n) {
                present = true;
                break;
            }
        }
        if (!present) {
            rec.status = PREFETCH_DONE;
            LOG_INF("phi-prefill: prefetch %d: its state left the prompt cache (loaded or replaced)\n", id);
        }
    }
}

size_t phi_prefill::drain(server_prompt_cache & cache) {
    std::deque<done> got;
    {
        std::lock_guard<std::mutex> lk(mtx);
        got.swap(ready);
        for (const auto & d : got) {
            waiting.erase(d.id_task);
            finished.erase(d.id_task);
            auto it = prefetches.find(d.id_task);
            if (it != prefetches.end()) {
                it->second.status = d.state.empty() ? PREFETCH_FAILED : PREFETCH_READY;
                if (d.state.empty()) {
                    it->second.reason = last_error.empty() ? "the prefill engine could not compute the prompt" : last_error;
                }
            }
        }
    }
    for (auto & d : got) {
        if (d.state.empty()) {
            continue;   // a failed prefill: the server computes it itself
        }
        server_prompt prompt;
        d.tokens.pop_back();   // the entry covers what the state holds: all but the last token
        prompt.tokens = server_tokens(d.tokens, false);
        // An entry that holds these tokens and more (an older branch of the
        // same sequence) makes the cache decline this state, and a slot
        // taking that entry for a task that continues past these tokens
        // differently has to drop the rest: a hybrid memory cannot, and
        // reads the task's prompt again from a checkpoint or from zero.
        // The state computed for exactly these tokens replaces such entries.
        for (auto it = cache.states.begin(); it != cache.states.end();) {
            if (it->prompt.tokens.size() >= prompt.tokens.size() && it->prompt.tokens.get_common_prefix(prompt.tokens) == prompt.tokens.size()) {
                LOG_INF("phi-prefill: %s %d: a cache entry of %zu tokens (%.1f MiB) held its %zu tokens and more; replaced by the state of exactly these\n",
                        d.id_task >= PREFETCH_ID_BASE ? "prefetch" : "task", d.id_task, it->prompt.tokens.size(), it->size() / 1048576.0, prompt.tokens.size());
                it = cache.states.erase(it);
            } else {
                ++it;
            }
        }
        auto * st = cache.alloc(prompt, d.state.size(), 0);
        if (!st) {
            LOG_WRN("phi-prefill: the prompt cache refused a state of %.1f MiB (its limit is %.1f MiB; raise --cache-ram)\n",
                    d.state.size() / 1048576.0, cache.limit_size / 1048576.0);
            continue;
        }
        memcpy(st->data.main.data(), d.state.data(), d.state.size());
        LOG_INF("phi-prefill: %s %d: state of %zu tokens (%.1f MiB) in the prompt cache: %zu entries, %.1f MiB\n",
                d.id_task >= PREFETCH_ID_BASE ? "prefetch" : "task", d.id_task, prompt.tokens.size(),
                d.state.size() / 1048576.0, cache.states.size(), cache.size() / 1048576.0);
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

// The client form: the prefill server computes all but the last token,
// saves slot 0, and the saved file is read here. The state bytes after the
// file's token list are exactly llama_state_seq_get_data's.
bool phi_prefill::fetch(job & j, std::vector<uint8_t> & state) {
    const llama_tokens sent(j.tokens.begin(), j.tokens.end() - 1);
    httplib::Client cli(url);
    cli.set_connection_timeout(10, 0);
    cli.set_read_timeout(timeout_s, 0);
    cli.set_write_timeout(timeout_s, 0);

    const json body = {
        {"prompt",       sent},
        {"n_predict",    0},
        {"id_slot",      0},
        {"cache_prompt", true},
    };
    auto res = cli.Post("/completion", body.dump(), "application/json");
    if (!res) {
        LOG_ERR("phi-prefill: %s/completion: %s (task %d)\n", url.c_str(), httplib::to_string(res.error()).c_str(), j.id_task);
        return false;
    }
    if (res->status != 200) {
        LOG_ERR("phi-prefill: %s/completion answered %d: %s (task %d)\n", url.c_str(), res->status, res->body.substr(0, 200).c_str(), j.id_task);
        return false;
    }
    double pp_tok_s = 0;
    try {
        const json r = json::parse(res->body);
        if (r.contains("timings")) {
            pp_tok_s = r["timings"].value("prompt_per_second", 0.0);
        }
    } catch (const std::exception & e) {
        LOG_WRN("phi-prefill: the completion's reply is not JSON: %s\n", e.what());
    }

    const std::string name = "phi-prefill-" + std::to_string(j.id_task) + ".bin";
    const json save = {{"filename", name}};
    auto res2 = cli.Post("/slots/0?action=save", save.dump(), "application/json");
    if (!res2 || res2->status != 200) {
        LOG_ERR("phi-prefill: %s/slots/0?action=save: %s (task %d)\n", url.c_str(),
                res2 ? res2->body.substr(0, 200).c_str() : httplib::to_string(res2.error()).c_str(), j.id_task);
        return false;
    }

    const std::string path = dir + name;
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        LOG_ERR("phi-prefill: cannot read %s: is PHI_PREFILL_DIR the prefill server's --slot-save-path? (task %d)\n", path.c_str(), j.id_task);
        return false;
    }
    const size_t size = (size_t) f.tellg();
    f.seekg(0);
    uint32_t hdr[3];
    bool ok = size >= sizeof(hdr) && f.read((char *) hdr, sizeof(hdr)).good();
    if (ok && (hdr[0] != LLAMA_STATE_SEQ_MAGIC || hdr[1] != LLAMA_STATE_SEQ_VERSION)) {
        LOG_ERR("phi-prefill: %s: magic %08x version %u, expected %08x %u (task %d)\n", path.c_str(),
                hdr[0], hdr[1], LLAMA_STATE_SEQ_MAGIC, LLAMA_STATE_SEQ_VERSION, j.id_task);
        ok = false;
    }
    llama_tokens packed;
    if (ok) {
        const size_t n_packed = hdr[2];
        ok = sizeof(hdr) + n_packed * sizeof(llama_token) <= size;
        if (ok) {
            packed.resize(n_packed);
            ok = f.read((char *) packed.data(), n_packed * sizeof(llama_token)).good();
        }
    }
    if (ok) {
        // The file holds the bytes of state_seq_write_data; the buffer that
        // llama_state_seq_set_data takes puts its own magic and the sequence
        // id in front of them (llama-context.cpp, io_magic; the import checks
        // the magic, so a drift there is said, not silent).
        const uint32_t io_magic = 0xaf143cd8;
        const int32_t  seq_id   = 0;
        const size_t n_state = size - sizeof(hdr) - packed.size() * sizeof(llama_token);
        state.resize(sizeof(io_magic) + sizeof(seq_id) + n_state);
        memcpy(state.data(), &io_magic, sizeof(io_magic));
        memcpy(state.data() + sizeof(io_magic), &seq_id, sizeof(seq_id));
        ok = f.read((char *) state.data() + sizeof(io_magic) + sizeof(seq_id), n_state).good();
    }
    f.close();
    std::remove(path.c_str());
    if (!ok) {
        LOG_ERR("phi-prefill: %s is short or unreadable (%zu bytes, task %d)\n", path.c_str(), size, j.id_task);
        state.clear();
        return false;
    }
    llama_tokens got;
    try {
        got = server_tokens::deserialize(packed, false).get_text_tokens();
    } catch (const std::exception & e) {
        LOG_ERR("phi-prefill: the saved slot's tokens: %s (task %d)\n", e.what(), j.id_task);
        state.clear();
        return false;
    }
    if (got != sent) {
        LOG_ERR("phi-prefill: the prefill server saved %zu tokens, %zu were sent; its slot must be free for this engine alone (task %d)\n",
                got.size(), sent.size(), j.id_task);
        state.clear();
        return false;
    }
    LOG_INF("phi-prefill: %s %d: the prefill server read %zu tokens at %.1f tok/s\n",
            j.id_task >= PREFETCH_ID_BASE ? "prefetch" : "task", j.id_task, sent.size(), pp_tok_s);
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
            current = j;   // covers() sees the job in flight
            busy = true;
        }
        const int64_t t0 = ggml_time_us();
        std::vector<uint8_t> state;
        const bool ok = url.empty() ? compute(j, state) : fetch(j, state);
        const int64_t dt = ggml_time_us() - t0;
        LOG_INF("phi-prefill: %s %d: %zu tokens in %.2f s (%.1f tok/s), state %.1f MiB%s\n",
                j.id_task >= PREFETCH_ID_BASE ? "prefetch" : "task",
                j.id_task, j.tokens.size(), dt / 1e6, j.tokens.size() * 1e6 / std::max<int64_t>(dt, 1),
                state.size() / 1048576.0, ok ? "" : " FAILED");
        {
            std::lock_guard<std::mutex> lk(mtx);
            busy = false;
            current.tokens.clear();
            if (!ok) {
                last_error = "the prefill engine could not compute " + std::to_string(j.tokens.size()) + " tokens (the decode server's log says why)";
            }
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
