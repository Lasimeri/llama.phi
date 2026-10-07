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
