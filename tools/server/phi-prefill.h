// llama.phi: the prefill engine. Prompt processing on its own model placement
// and its own thread, decoupled from the server's decode loop: a completion
// task's prompt is computed here, the finished sequence's state is handed
// to the server's prompt cache, and the slot that takes the task restores it
// and generates at once (docs/phi/README.md).
//
// Enabled by PHI_PREFILL=1. The prefill model is a second load of the same
// file with its own placement, read from the environment (each unset one
// keeps the server's value):
//   PHI_PREFILL_TS      tensor split, "a/b/c/d" (one weight per GPU)
//   PHI_PREFILL_OT      tensor buffer overrides, "pattern=buft,..." (as -ot)
//   PHI_PREFILL_NGL     layers on the GPUs (as -ngl)
//   PHI_PREFILL_BATCH   logical batch (as -b)
//   PHI_PREFILL_UBATCH  physical batch (as -ub)
//   PHI_PREFILL_THREADS threads for the prefill model's CPU work (as -t)
//   PHI_PREFILL_KV_HOST 1: the prefill context's KV cache in host memory (as -nkvo)
//   PHI_PREFILL_MIN     prompts shorter than this many tokens stay with the server (default 64)
// The decode context's KV layout is mirrored (kv_unified, n_seq_max, K and V
// types), since the exported state must import there.
#pragma once

#include "common.h"
#include "llama.h"
#include "server-task.h"

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

struct phi_prefill {
    bool enabled = false;

    // Called on the engine's thread when a prompt is done: the server wakes
    // its loop with it, so the deferred task is offered again (the loop
    // otherwise sleeps until a new task arrives).
    std::function<void()> on_done;

    bool init(const common_params & base);
    void shutdown();

    // A task's prompt is worth prefilling here: long enough, text only, and
    // not already held (mostly) by the cache or a slot.
    bool wants(const server_tokens & tokens, const server_prompt_cache * cache, int lcp_slots) const;

    // Submit a task's prompt; it is computed in order of arrival.
    void submit(int id_task, const llama_tokens & tokens);

    bool is_waiting(int id_task);   // submitted, state not yet in the server's cache
    bool is_ready(int id_task);     // computed, state waiting to be drained

    // Move every finished state into the server's prompt cache; the ids
    // moved leave the waiting set. Called from the server's thread.
    size_t drain(server_prompt_cache & cache);

    // Counters for /metrics and the log.
    int64_t n_prompts = 0;
    int64_t n_tokens  = 0;
    int64_t t_prefill_us = 0;

private:
    struct job {
        int id_task;
        llama_tokens tokens;
    };
    struct done {
        int id_task;
        llama_tokens tokens;
        std::vector<uint8_t> state;
    };

    common_init_result_ptr llama_init;
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;
    int32_t n_ubatch = 2048;
    int32_t n_ctx    = 0;
    size_t  min_tokens = 64;

    std::mutex mtx;
    std::condition_variable cv;
    std::deque<job>  queue;
    std::deque<done> ready;
    std::unordered_set<int> waiting;
    std::unordered_set<int> finished;
    bool stop = false;
    std::thread worker;

    void run();
    bool compute(job & j, std::vector<uint8_t> & state);
};
