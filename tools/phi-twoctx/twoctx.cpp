// twoctx: can llama.cpp run a prompt in one context while another context
// generates, on the same model, at once? The feasibility test for llama.phi
// (prefill and decode decoupled). One model, the rack's placement (experts
// of blocks 0..27 on the GPUs by tensor_split 7/7/7/28, the rest and the
// per-layer embeddings in host memory, KV in host memory); context A
// prefills a long prompt in chunks on one thread while context B generates
// tokens on another; each rate is also measured alone. Then A's sequence
// is exported (llama_state_seq_get_data) and imported into B as a second
// sequence, which must continue it.
//
//   twoctx MODEL PROMPT_FILE [n_decode=64] [chunk=512]
// Build (against the stock build's libraries, the fork's headers):
//   g++ -O2 -std=c++17 twoctx.cpp -I$L/include -I$L/ggml/include -L$L/build/bin -lllama -lggml -lggml-base -lpthread -Wl,-rpath,$L/build/bin -o twoctx
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "ggml-backend.h"
#include "llama.h"

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool bos) {
    std::vector<llama_token> out(text.size() + 16);
    int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), out.data(), (int) out.size(), bos, true);
    if (n < 0) { out.resize(-n); n = llama_tokenize(vocab, text.c_str(), (int) text.size(), out.data(), (int) out.size(), bos, true); }
    out.resize(n);
    return out;
}

// Prefill `toks` into ctx as seq, chunked; returns tokens per second.
static double prefill(llama_context * ctx, const std::vector<llama_token> & toks, int seq, int chunk, const char * tag) {
    llama_batch b = llama_batch_init(chunk, 0, 1);
    double t0 = now();
    for (size_t i = 0; i < toks.size(); i += chunk) {
        int n = (int) std::min((size_t) chunk, toks.size() - i);
        b.n_tokens = n;
        for (int j = 0; j < n; j++) {
            b.token[j] = toks[i + j]; b.pos[j] = (llama_pos) (i + j);
            b.n_seq_id[j] = 1; b.seq_id[j][0] = seq; b.logits[j] = (i + j + 1 == toks.size());
        }
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "%s: decode failed at %zu\n", tag, i); break; }
    }
    double r = toks.size() / (now() - t0);
    llama_batch_free(b);
    fprintf(stderr, "%s: prefill %zu tokens: %.1f tok/s\n", tag, toks.size(), r);
    return r;
}

// Generate n tokens greedily in ctx on seq from position pos; returns tok/s and the text.
static double generate(llama_context * ctx, const llama_vocab * vocab, int seq, llama_pos pos, int n, std::string & text, const char * tag) {
    llama_sampler * s = llama_sampler_init_greedy();
    llama_batch b = llama_batch_init(1, 0, 1);
    double t0 = now();
    int made = 0;
    for (int i = 0; i < n; i++) {
        llama_token t = llama_sampler_sample(s, ctx, -1);
        if (llama_vocab_is_eog(vocab, t)) break;
        char buf[256];
        int k = llama_token_to_piece(vocab, t, buf, sizeof buf, 0, true);
        if (k > 0) text.append(buf, k);
        b.n_tokens = 1; b.token[0] = t; b.pos[0] = pos++; b.n_seq_id[0] = 1; b.seq_id[0][0] = seq; b.logits[0] = 1;
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "%s: decode failed\n", tag); break; }
        made++;
    }
    double r = made / (now() - t0);
    llama_batch_free(b);
    llama_sampler_free(s);
    fprintf(stderr, "%s: generated %d tokens: %.2f tok/s\n", tag, made, r);
    return r;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: twoctx MODEL PROMPT_FILE [n_decode] [chunk]\n"); return 2; }
    int n_decode = argc > 3 ? atoi(argv[3]) : 64;
    int chunk = argc > 4 ? atoi(argv[4]) : 512;
    std::ifstream f(argv[2]); std::stringstream ss; ss << f.rdbuf();
    std::string prompt = ss.str();

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    float ts[16] = {7, 7, 7, 28};
    mp.tensor_split = ts;
    llama_model_tensor_buft_override ov[3] = {
        {"blk\\.(2[8-9]|3[0-9]|4[0-7])\\.ffn_(up|gate|down)_exps\\.weight", ggml_backend_cpu_buffer_type()},
        {"per_layer_token_embd\\.weight", ggml_backend_cpu_buffer_type()},
        {nullptr, nullptr},
    };
    mp.tensor_buft_overrides = ov;
    mp.use_extra_bufts = false;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "no model\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    auto cparams = [&](uint32_t n_ctx, uint32_t n_batch, int n_seq) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = n_ctx; cp.n_batch = n_batch; cp.n_ubatch = n_batch; cp.n_seq_max = n_seq;
        cp.n_threads = 12; cp.n_threads_batch = 12;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.offload_kqv = false;             // KV in host memory
        cp.kv_unified = true;               // one stream: a state moves between contexts (n_stream must match)
        cp.type_k = GGML_TYPE_Q8_0; cp.type_v = GGML_TYPE_Q8_0;
        return cp;
    };
    llama_context * A = llama_init_from_model(model, cparams(16384, chunk, 1));   // prefill
    llama_context * B = llama_init_from_model(model, cparams(16384, 64, 2));      // decode
    if (!A || !B) { fprintf(stderr, "no context\n"); return 1; }

    std::vector<llama_token> big = tokenize(vocab, prompt, true);
    std::vector<llama_token> small = tokenize(vocab, "The four Xeon Phi cards in this machine", true);
    fprintf(stderr, "prompt: %zu tokens; small: %zu tokens\n", big.size(), small.size());

    // Alone: B's generation after its small prompt.
    prefill(B, small, 0, 64, "B-alone");
    std::string t1;
    double tg_alone = generate(B, vocab, 0, (llama_pos) small.size(), n_decode, t1, "B-alone");
    // Alone: A's prefill.
    double pp_alone = prefill(A, big, 0, chunk, "A-alone");
    llama_memory_seq_rm(llama_get_memory(A), 0, -1, -1);
    llama_memory_seq_rm(llama_get_memory(B), 0, -1, -1);

    // Together: A prefills while B generates.
    prefill(B, small, 0, 64, "B-pre");
    std::string t2;
    double pp_both = 0, tg_both = 0;
    std::thread ta([&] { pp_both = prefill(A, big, 0, chunk, "A-with-B"); });
    std::thread tb([&] { tg_both = generate(B, vocab, 0, (llama_pos) small.size(), n_decode, t2, "B-with-A"); });
    ta.join(); tb.join();

    // Handoff: A's finished prompt into B as seq 1, continued there.
    double t0 = now();
    size_t sz = llama_state_seq_get_size(A, 0);
    std::vector<uint8_t> buf(sz);
    size_t got = llama_state_seq_get_data(A, buf.data(), sz, 0);
    size_t put = llama_state_seq_set_data(B, buf.data(), got, 1);
    fprintf(stderr, "handoff: %zu bytes exported, %zu imported, %.3f s\n", got, put, now() - t0);
    std::string t3;
    double tg_hand = put ? generate(B, vocab, 1, (llama_pos) big.size(), 32, t3, "B-handoff") : 0;

    printf("tg_alone=%.2f tg_with_prefill=%.2f pp_alone=%.1f pp_with_decode=%.1f handoff_bytes=%zu tg_after_handoff=%.2f\n",
           tg_alone, tg_both, pp_alone, pp_both, got, tg_hand);
    printf("same text alone vs together: %s\n", t1 == t2 ? "yes" : "NO");
    printf("continuation after handoff: %.*s\n", (int) std::min<size_t>(200, t3.size()), t3.c_str());
    llama_free(A); llama_free(B); llama_model_free(model); llama_backend_free();
    return 0;
}
