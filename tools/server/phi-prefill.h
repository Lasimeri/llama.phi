// llama.phi: the prefill engine. Prompt processing on its own model placement
// and its own thread, decoupled from the server's decode loop: a completion
// task's prompt is computed here, the finished sequence's state is handed
// to the server's prompt cache, and the slot that takes the task restores it
// and generates at once (docs/phi/README.md).
//
// Enabled by PHI_PREFILL=1. Two forms:
//
// The client form (PHI_PREFILL_URL set): the prompts go to another
// llama-server, the prefill server, started with the GPU placement and
// --slot-save-path (scripts/phi-serve.sh starts both). The engine posts the
// prompt to it (/completion, n_predict 0, slot 0), asks it to save the slot
// (/slots/0?action=save), reads the saved file (LLAMA_STATE_SEQ_MAGIC,
// version, the slot's packed tokens, then exactly the bytes of
// llama_state_seq_get_data) and hands the state to this server's prompt
// cache. No memory is shared between the two processes, so a backend in
// this one (the cards') never runs beside the prefill's.
//   PHI_PREFILL_URL     the prefill server, e.g. http://127.0.0.1:8002
//   PHI_PREFILL_DIR     its --slot-save-path (a directory this server can read)
//   PHI_PREFILL_TIMEOUT seconds to wait for one prompt (default 3600)
//
// The in-process form (no URL): the prefill model is a second load of the
// same file in this process with its own placement, read from the
// environment (each unset one keeps the server's value):
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
//
// Prefetch (both forms): a client hands a prompt over before it needs it
// (POST /phi/prefetch {"tokens": [...]} -> {"id": N}); the engine reads it
// while the slots keep generating, no slot and no task involved, and the
// state lands in the prompt cache. The client polls GET /phi/prefetch?id=N
// ({"state": "waiting" | "ready" | "done" | "failed"}) and then sends its
// completion with those tokens (and more after them): the slot restores
// the state and reads only what follows it.
#pragma once

#include "common.h"
#include "llama.h"
#include "server-task.h"

#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <string>
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

    // A task whose prompt was submitted, until forget(): the server does not
    // submit it twice (a failed prefill would otherwise go round for ever).
    bool was_submitted(int id_task);
    void forget(int id_task);

    // A job queued or in flight will hold a longer prefix of these tokens
    // than lcp_known (the most any slot holds): a task with them should wait
    // for it rather than read the prompt itself.
    bool covers(const server_tokens & tokens, size_t lcp_known);

    // Move every finished state into the server's prompt cache; the ids
    // moved leave the waiting set. Called from the server's thread.
    size_t drain(server_prompt_cache & cache);

    // The prefetch API (server-context.cpp routes it). The id is in its own
    // range above any task id; -1 with the reason in err when refused.
    int submit_prefetch(const llama_tokens & tokens, std::string & err);

    // "waiting": queued or being read; "ready": computed, its state is in
    // the prompt cache or lands there with the next completion; "done": the
    // state left the cache (a slot loaded it, a longer prompt replaced it,
    // or the cache limit evicted it); "failed": the prefill failed (a
    // completion reads the prompt itself; the reason, when asked for, is
    // the engine's last error for it). "" for an unknown id.
    std::string prefetch_state(int id, std::string * reason = nullptr);

    // After the prompt cache changed on the server's thread: prefetches
    // whose state is no longer there turn "done".
    void audit(const server_prompt_cache & cache);

    // Counters for /metrics and the log.
    int64_t n_prompts = 0;
    int64_t n_tokens  = 0;
    int64_t t_prefill_us = 0;

    static constexpr int PREFETCH_ID_BASE = 1 << 30;

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
    enum prefetch_status {
        PREFETCH_WAITING,
        PREFETCH_READY,
        PREFETCH_DONE,
        PREFETCH_FAILED,
    };
    struct prefetch_rec {
        llama_tokens tokens;   // as submitted; the state holds all but the last
        prefetch_status status;
        std::string reason;    // why it failed
    };
    std::string last_error;    // the engine's last failure, for the record (under mtx)

    // the client form
    std::string url;
    std::string dir;
    int timeout_s = 3600;

    // the in-process form
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
    std::unordered_set<int> submitted;
    std::map<int, prefetch_rec> prefetches;   // by id, so in order of submission
    int next_prefetch_id = PREFETCH_ID_BASE;
    job  current;                 // the job in flight (tokens empty when none)
    bool busy = false;
    bool stop = false;
    std::thread worker;

    void run();
    bool compute(job & j, std::vector<uint8_t> & state);   // in-process
    bool fetch(job & j, std::vector<uint8_t> & state);     // client
};
