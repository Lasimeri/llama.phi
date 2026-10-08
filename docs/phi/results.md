# llama.phi: measurements

## 2026-10-06: two contexts at once, and the handoff (`tools/phi-twoctx`)

Rack: EPYC 7742, four RTX 3080 20 GB, four Xeon Phi 3120 (the cards'
backend from Intel-Phi-AVX512 with `PHI_GGML_PP_ONLY=1`, so the cards
took the prompt's host-block multiplies and never a generation step).
Model: Qwen3.8 Flash Next UD-Q6_K_XL, experts of blocks 0 to 27 on the
GPUs (`tensor_split` 7/7/7/28), blocks 28 to 47 and the per-layer
embeddings in host memory, both contexts' KV in host memory, Q8 K and V,
one stream (`kv_unified`), 12 threads each. Prompt: 6034 tokens of
code, in 512-token chunks; generation greedy.

```
twoctx MODEL calib-code.txt 64 512      (twoctx2.log, twoctx2.out)
```

| | alone | together |
| --- | --- | --- |
| context A, prefill, tok/s | 172.8 | 168.6 |
| context B, generation, tok/s | 11.00 | 8.28 |

Together: A's whole prefill ran on one thread while B generated 64
tokens on another, on the same model and the same GPUs. The prefill
lost 2 percent; the generation kept 75 percent of its rate while a full
prompt ran beside it (the first run of the pair, before the one-stream
fix, kept 87: 9.26 against 10.67). B's text was identical alone and
together.

Handoff: A's finished sequence exported with `llama_state_seq_get_data`
(216,804,420 bytes for 6034 tokens: 35.9 KB a token, Q8 K and V plus
the recurrent state) and imported into B as a second sequence with
`llama_state_seq_set_data` in 0.081 s; B then continued the prompt at
9.52 tok/s with a correct continuation. The import fails with
`n_stream mismatch` unless both contexts have the same stream count:
`kv_unified = true` on both gives one stream whatever `n_seq_max` is.

What this settles: one process, one model, a prefill context and a
decode context on their own threads, is sound on this build (llama.cpp
c479922ac) with CUDA and the cards' backend entered from two threads.
The server design in `README.md` stands. What the next measurements
decide is the placement of each context's model: the prefill one for
the GPUs at large batches, the decode one for the CPU and the cards.

## 2026-10-06, later: the placement of each side (llama-bench, stock build)

Prompts on the GPUs alone, batch 4096, ubatch 2048, KV in host memory,
op offload on, the per-layer embeddings memory-mapped:

| prefill model | blocks with experts in VRAM | pp4096 tok/s |
| --- | --- | --- |
| UD-Q6_K_XL, `-ts 6/6/6/30` | 0 to 23 | 440.1, 453.0 |
| UD-Q4_K_XL, `-ts 8/8/8/24` | 0 to 29 | 560.5 |

(With 28 Q6 blocks or 34 Q4 blocks the 2048-token compute buffers did
not fit: `cudaMalloc failed`.) Generation on the CPU, no GPU:

| decode side | tg64 tok/s |
| --- | --- |
| CPU, 32 threads | 7.62 |
| CPU, 64 threads | 7.25 |
| CPU, 12 threads, plus the four cards (row share, `PHI_GGML_PP_ONLY=0`) | 7.12 |
| (the shared placement of the morning, GPUs and host) | 7.6 to 8.0 |

So the decode side loses 5 percent by leaving the GPUs, and the GPUs
then take the prompts five times faster. The cards do not help the
decode on this host (the row share costs more than it saves at one
token).

`-sm row` (llama.cpp's own tensor parallelism) does not exist in this
version for CUDA: `device CUDA0 does not support split buffers`. Its
micro-batch pipelining across the GPUs is off whenever the KV cache is
in host memory or any tensor override is set (`llama-context.cpp`,
`pipeline_parallel`), which this placement needs; the fork adds
`LLAMA_PHI_PIPELINE=1` to ask for it anyway, measured below.

## 2026-10-06, later: the server (`scripts/phi-smoke.sh`, the fork)

Decode on the CPU (32 threads, two slots, 131072 cells, Q8 K and V, one
KV stream), the prefill engine on the GPUs with the Q6 placement above.
A 6034-token prompt with 32 tokens to generate, and two seconds into
it a short request generating 48 tokens.

| | short request | long request | prefill engine |
| --- | --- | --- | --- |
| server reading its own prompts (first run, the state import failed) | 48 tokens at 6.9 tok/s, in flight during the prompt | 104.5 s wall: prompt 6034 tokens at 74.5 tok/s on the CPU, then 32 at 7.2 | 6034 tokens in 19.8 s (305 tok/s), unused |
| engine's state restored (last token left to the slot) | 48 tokens at 6.5 tok/s, in flight during the prompt | **23.8 s wall: prompt 1 token (181 ms), then 32 at 7.1** | 6034 tokens in 19.1 s (315 tok/s) |

Three defects met on the way, each fixed in the fork:
`tensor_buft_overrides` must end with a null entry (the conversion
asserts); the server's loop sleeps until a new task arrives, so a
finished prefill has to wake it (`pop_deferred_task`); and the slot
rewinds one position to recompute the last token's logits, which a
recurrent layer cannot do (the slot then redid the prompt), so the
engine prefills all but the last token.

Also found: llama-server never calls `ggml_backend_load_all`, so
`GGML_BACKEND_PATH` (the cards' backend) was ignored by it while
llama-bench honoured it; the fork adds the call.

## 2026-10-06, late: the cards beside the engine (open)

With the cards' backend in the decode process and the prefill engine
on, the server crashed within seconds every time (four runs, four
sites: the CPU q6_K dot product, the tiled mixture unpack with a
non-canonical pointer, a jump to a data address inside the backend's
`host_rows_id` compute, a threadpool worker writing to the main
thread's stack); never without the engine (0.37 tok/s at 32 decode
threads, the known contention; 12 threads is the backend's rule), never
without the backend (23.8 s, twice). Giving the prefill model the CUDA
devices only (`PHI_PREFILL_DEVICES`) and building ggml without OpenMP
changed the site, not the outcome. ECC reports no memory errors; the
cards' memory is the resident share by design. The next build is the
two-process form: a prefill llama-server on the GPUs, the engine as its
client (`/completion` with the tokens, `/slots/0?action=save`), the
saved-slot file (magic GGSQ, version 4, token count, tokens, then the
state bytes) read into the decode server's prompt cache; no memory
shared, so no race. Until then the working configuration is the engine
with decode on the CPU and no cards' backend (`scripts/phi-serve.sh`
run directly, not through `phi-ggml.sh`).

## 2026-10-06, night: the two-process form (`scripts/phi-serve.sh`, the fork)

Built as planned: the prefill server (llama-server on the GPU placement,
port 8002, `--slot-save-path /mnt/raid0/phi-prefill/`, one slot) and the
decode server (port 8001, `-ngl 0`, 12 threads, the cards' backend with
`PHI_GGML_TG_ONLY=1`), the engine in the decode server as the prefill
server's client (`PHI_PREFILL_URL`, `PHI_PREFILL_DIR`): `/completion`
with the prompt's tokens but the last, `n_predict` 0, then
`/slots/0?action=save`, the file read and its tokens checked against
what was sent. Both servers load the model from the raid0 copy.

First run (`smoke-2proc-1.log`): no crash with the cards' backend beside
the engine, which is what the form is for. The prefill server read 6033
tokens at 354 tok/s (17.2 s). The import failed: `wrong sequence state
magic`. The saved file holds the bytes of `state_seq_write_data` after
its token list, while the buffer `llama_state_seq_set_data` takes starts
with llama-context.cpp's own `io_magic` (0xaf143cd8) and the sequence
id; `llama_state_seq_get_data` writes those two, the file writer does
not. The engine now puts them in front of the file's state bytes. With
the import failed the decode server read the prompt itself with 12 host
threads under `PHI_GGML_TG_ONLY` (26.7 tok/s), and the short request's
48 tokens rode in the prompt's batches (0.19 tok/s): upstream's
coupling, which is what the engine removes once the state imports.

Second run (`smoke-2proc-2.log`), the prefix in place: the state imports.

| | short request (2 s into the long prompt) | long request (6034 tokens, 32 generated) |
| --- | --- | --- |
| wall | 29.2 s | 31.2 s |
| prompt | 11 tokens, 0.68 s | prefill server 6033 tokens in 17.2 s (353 tok/s), then 1 token in the slot (0.33 s) |
| generation | 48 tokens at 1.65 tok/s | 32 tokens at 4.21 tok/s |

During the short request's generation the cards ran at 50 percent (114
of 228 threads, the request's size) on all four, the GPUs at 72 to 96
percent on the prefill: the two sides at once, the form the fork is for.
Two things were wrong in the numbers. The long request's generation never
reached the cards (0.1 percent while it ran): both slots were generating,
and a step of two slots is a two-token multiply, which
`PHI_GGML_TG_ONLY=1` keeps on the host. And 1.65 tok/s beside the prefill
against 6.57 alone (`curl`, 64 tokens, the same server idle otherwise) is
the prefill's host side, 24 threads streaming the experts of blocks 24 to
47 through the memory bus the generation needs; the CPU showed 6 percent
busy, the cards 50: neither was the limit, the bus was.

`PHI_GGML_TG_ONLY` is now a count (the AVX-512 repo): a multiply of up
to K tokens goes to the cards, K the slots generating at once; the
launcher sets it to the slot count. Third run (`smoke-2proc-3.log`, the
prefill server's slot still held the prompt from the run before, so the
prefill was a cache hit and the GPUs idle):

| | short request | long request | two slots at once, 48 tokens each, nothing else running |
| --- | --- | --- | --- |
| generation | 48 at 5.04 tok/s | 32 at 3.92 tok/s (the short one beside it) | 5.56 and 5.56 tok/s, the cards at 50 percent on all four |

Standing: the two-process form works end to end with the cards in the
decode path (three runs, no crash). Generation with the cards is 6.57
tok/s alone against 7.1 on the CPU with 32 threads: the cards' fixed
cost per multiply (the backend's known pole) eats what their bandwidth
gives. Beside a real prefill the bus is shared with the prefill server's
host side; the lever there is more of the prefill model in VRAM (the
Q4_K_XL file fits 30 blocks, 560 tok/s), fewer prefill threads, or both.

## 2026-10-07, night: the restored state, and the tensor split for the prefill

**The handed-over state was not loaded (fixed, 9750f6ebf).** With the
decode server restarted under the cards (expert placement, 12 threads) and
the harness pinned to slot 1, the prefill server read the harness's
9250-token prompt in 31.1 s (297 tok/s, state 256.9 MiB), and then the
decode server's slot 1 read the same prompt again itself: 131.7 s at 70
tok/s. The cause is upstream's `get_available_slot`: a slot picked by id
skips the similarity pass, and when it is empty (new, or cleared after its
idle save, `cache_idle_slots` being on by default) `f_keep` is 0/0, NaN,
so the prompt cache is never consulted and the engine's state stays there
unused. An empty slot now always restores from the cache. Measured after
the fix on the live harness:

| | before | after |
| --- | --- | --- |
| harness prompt, handed over | 9250 tokens: 31.1 s at the prefill server, then 131.7 s read again by the decode slot | 11561 tokens: 9.67 s (the prefill server's own cache held the start), then a 1-token prompt eval (0.22 s) |
| the other slot meanwhile (`bench/async-2026-10-07/xslot.sh`, 22 prompt tokens, 64 generated, slot 0) | 0.44 tok/s: every step shared a 2047-token chunk of slot 1's local read | not measured beside a handoff yet; 4.1 tok/s with slot 1 generating beside it |

A turn's increment (229 to 430 tokens) is still read by the decode server
itself (`phi_prefill::wants`: mostly in a slot already), 35 to 58 tok/s,
about 7 s a turn, and those steps hold up any other slot's generation the
same way.

**The tensor split for the prefill server** (`-sm tensor`, the meta
device; `bench/tp-2026-10-07/tp.sh`, `tp2.sh`): one 9477-token prompt of
real source (`prompt.json`), one server per arm on port 8012, arms
interleaved, the production prefill server stopped meanwhile and started
again unchanged; the decode server stayed up (1.6 GB on GPU 0). Same model
(Q6_K_XL), `-c 131072 --kv-unified -fa on -ctk q8_0 -ctv q8_0 -b 4096 -t
24 -tb 24`, experts of blocks 24 to 47 in host memory.

| arm | tok/s, two rounds | mean SM busy, GPU 0 / 1 / 2 / 3 |
| --- | --- | --- |
| layer split (today's: `-ts 6/6/6/30 -nkvo -ub 2048`) | 304.6, 300.8; 304.6, 301.1 | 60-66 / 3-5 / 1-2 / 14-18 |
| `-sm tensor -nkvo` | aborts: `ggml-backend-meta.cpp:830`, the gated delta net's state (src 5) in host memory has no split axis | |
| `-sm tensor`, KV on the GPUs, `-ub 2048` | aborts: 2373 MiB more on GPU 0 does not fit (the split is even, GPU 0 also holds the decode server's 1.6 GB) | |
| `-sm tensor`, experts of blocks 22 to 47 in host memory | 135.7, 136.3 | 12-14 / 12-15 / 12-13 / 15 |
| `-sm tensor -ub 1024` | 142.0, 141.2 | 14-29 / 12-14 / 12-14 / 13-14 |

The tensor split runs on this model, and its continuations (32 greedy
tokens) read as the layer split's (one arm words one line differently),
but it takes 2.2 times as long. The meta device has no `offload_op`
(`ggml-backend-meta.cpp`, device interface): a multiply whose weights sit
in host memory is not offloaded to any GPU, so the experts of the host
blocks (24, or 26 in the first tensor arm) run on the 24 CPU threads while the four GPUs wait (12 to 15
percent busy). Under the layer split the same experts are offloaded, all
of them to GPU 0 (the first device that takes them), whose one x8 link
carries the 45 GB a batch: GPU 0 busy 60 to 66 percent, the other three
nearly idle. Without NCCL in this build (`GGML_CUDA_NCCL` off, no NCCL
installed), four devices fall back to the meta device's butterfly
all-reduce through host memory.

Standing: the production prefill server stays on the layer split. The
tensor split pays only if the host experts reach the GPUs: an `offload_op`
in the meta device that sends each GPU its quarter of a host weight's rows
over its own link (four x8 links instead of one) and lets the split
machinery combine the results.

## 2026-10-08: the prompt read ahead of its completion (prefetch), and the pinned slot that loads it

The harness keeps one live sequence in slot 1 and gets its input while
that slot is generating. Until today its only way to have the input read
was to send the whole prompt as the next completion: the decode server
read the increment itself (57 to 68 tok/s, `phi-decode.log`, a 1580-token
turn 27 s) while the slot stood still, and a prompt that did not extend
the slot's own tokens cost a re-read from a checkpoint or from zero (a
hybrid memory cannot rewind: `llama_memory_hybrid::seq_pos_min` is the
recurrent tail, so a task that shares fewer tokens than the slot holds
trips `pos_min >= pos_min_thold` in `update_slots`).

Three things changed in `tools/server` (the prefill engine, the server's
slot selection, two routes):

1. `POST /phi/prefetch {"tokens": [...]}` answers `{"id": N, "n_tokens":
   n}` at once; the engine reads the prompt (all but its last token, as
   it does for a task) on the prefill server while the slots keep
   generating; no slot and no task are involved. `GET /phi/prefetch?id=N`
   answers `{"id": N, "state": s}` with `waiting` (queued or being read),
   `ready` (computed; in the decode server's prompt cache, or there with
   the next completion), `done` (the state left the cache: a slot loaded
   it, a longer prompt replaced it, or the cap evicted it) or `failed`
   (with `"error"`: the reason; a completion then reads the prompt
   itself); an unknown id is 404, an empty or one-token list 400. The ids
   start at 2^30, above any task id; the engine files them under its own
   records (64 kept). Every completion task drains the engine's finished
   states into the cache before a slot is chosen, and `update_slots`
   drains on every pass, so a state is in the cache within one decode
   step of the prefill server finishing it.
2. `get_available_slot`: a slot picked by id that holds a shorter prefix
   of the task than a cache entry does saves itself and loads the entry
   (the cache's own criteria, `server_prompt_cache::load`, applied first,
   so the save is never for nothing). Upstream looks at the cache only
   when the slot would lose half its context (`f_keep < 0.5`), and the
   pinned slot after a prefetch holds its earlier turn plus what it
   generated meanwhile, `f_keep` near 1. A slot still processing is left
   alone (the empty-slot rule of 9750f6ebf had no such guard). A
   completion whose prompt a job in flight covers with a longer prefix
   than any slot holds waits for that job (`phi_prefill::covers`), and a
   task whose prefill failed is not submitted again (`was_submitted`).
3. The drain replaces a cache entry that holds the prefetched tokens
   and more (an older branch of the same sequence): `alloc` would
   decline the state as already held, and a slot taking the longer entry
   for a task that continues past the prefix differently cannot drop the
   rest. The first forced injection below ran into exactly that.

Memory: the decode server ran `--cache-ram -1` beside a 188 GB model in a
251 GB host. A state is 22 KiB a token (412.1 MiB for 19189 tokens, 237.4
MiB for 7999, 714.6 MiB for 38570; 2.1 GiB at 99k), and a slot that loads
a better entry first saves its own, so each injection leaves one state
of the live sequence's length behind. `scripts/phi-serve.sh` now gives
`--cache-ram ${PHI_CACHE_RAM:-24576}`: the oldest entry goes when the cap
is reached (`server_prompt_cache::alloc`, `update`), an entry that is a
prefix of a newer one goes at once, and the cap also lifts the cache's
token limit (with `-1` the token limit is `n_ctx`, 131072 tokens in all:
one 99k state would have evicted the other; with a cap it is the cap
over the measured bytes a token, about 1.1M tokens).

**The forced injection** (`bench/async-2026-10-08/inject3.sh`, log
`inject3.log`; the decode server on this build with the production
config, cards 0 to 2, the harness live in slot 1 throughout; slot 0 for
every request; H = 3500 tokens of `calib-code.txt`, T = 4500 tokens of
`server-common.cpp`, everything greedy, `temperature 0, top_k 1`):

| step | request on slot 0 | prompt eval | what happened |
| --- | --- | --- | --- |
| a | H, 64 tokens | 1 token, 0.26 s | the engine read H on the GPUs (26.5 s in all), the slot restored it; the slot holds H+G |
| ref | H+T+G, 8 tokens | 1 token, 0.30 s | the engine read H+T+G[:-1] on the GPUs (52.7 s in all, the state 238 MiB); the slot, holding H+G, saved itself and loaded it (the new rule); reference tokens `20668 11 18738 20668 553 4310 1070 1518` |
| a2 | H, 64 tokens | 3500 tokens, 51.5 s at 68 tok/s | the slot held H+T+G+ref; upstream saved it (f_keep below 0.5) and the hybrid memory read H again from zero: the cost the prefetch removes; the slot holds H+G' |
| prefetch | `POST /phi/prefetch` H+T | | `{"id":1073741824,"n_tokens":8000}` at once; `waiting` for 9.8 s (the prefill server's slot held H+T from ref: 7999 tokens at 3600 tok/s of its own cache), then `ready`; the drain replaced the entry of step a2 (H+T+G+ref, which held H+T and more) with the state of exactly H+T[:-1] |
| c | H+T+G, 8 tokens | **65 tokens, 4.73 s** | the slot, holding H+G', saved itself and loaded the prefetched entry, read T[-1]+G (65 = |G|+1) and generated `20668 11 18738 20668 553 4310 1070 1518`: **equal to the reference**, which came from a different split of the same computation (the GPUs through H+T+G[:-1] there, the GPUs through H+T[:-1] and the CPU with the cards through 65 tokens here); `GET` then says `done` |

The decode log of step c: `selected slot by id (0)`, then `llama.phi: the
prompt cache holds 7999 of the task's 8064 tokens, the slot 3500: the
slot saves itself and loads the entry`, then `prompt eval time = 4731 ms
/ 65 tokens`. An unknown id answers 404, an empty token list 400. The
first run of the day (`inject.sh`, same steps in another order) showed
the entry-replacement need: with step ref's long entry still in the
cache the prefetched state was declined as "already in the cache" and
step c loaded the long entry and read 1 token, correct because that
entry happened to continue with the same G; a harness whose G differs
would have re-read.

**Measurement 1, generation beside a prefill** (`decode-rate.sh`, log
`decode-rate.log`): slot 0, a 19-token prompt, 128 greedy tokens with
`ignore_eos`, alone and while the prefill server reads a 16000-token
prompt (`POST` to port 8002, `n_predict 0`, a different prompt each
round so its slot cannot hit its own cache), two rounds interleaved.
The harness's slot 1 generated throughout (so "alone" is two slots, as
in production).

| round | alone | beside the prefill | the prefill itself |
| --- | --- | --- | --- |
| 1 | 4.13 tok/s | 2.10 tok/s | 16000 tokens in 79.5 s, 201 tok/s |
| 2 | 3.99 tok/s | 2.24 tok/s | 16000 tokens in 78.1 s, 205 tok/s |

Generation keeps 51 to 56 percent of its rate beside a prefill (the Q6
run of 2026-10-06 kept 25 percent, 1.65 of 6.57): the 16000-token read
is 8 ubatches, each streaming the 29 host blocks' experts over GPU 0's
link, and the generation shares the memory bus with that stream for 78
s. The prefill's host threads are the knob below.
