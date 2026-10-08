# llama.phi

A fork of llama.cpp (branch `phi`, from upstream c479922ac, 2026-10-06)
whose server keeps prompt processing and token generation apart: two
engines in flight at once on one model, so a new request's prompt never
stalls the tokens another request is receiving. Everything else is
upstream, unchanged; the fork's own files are `tools/server/phi-prefill.*`,
one hook in `tools/server/server-context.cpp`, `tools/phi-twoctx`,
`scripts/phi-serve.sh`, `scripts/phi-smoke.sh` and `docs/phi/`. The fork
tracks upstream by rebasing them.

## Why

On the GPU rack (four RTX 3080 20 GB, four Xeon Phi 3120 cards, 251 GB)
with Qwen3.8 Flash Next at UD-Q6_K_XL, prompt processing is compute
bound (every token through every weight) and generation is bandwidth
bound (one token, the active weights streamed). Upstream llama-server
runs both through one context on one placement: every prompt chunk
sits in the same decode call as the live tokens, and the devices take
turns on each call. Decoupling them is a server design, not a kernel.

## How (`tools/server/phi-prefill.h`)

llama-server stays whole and is the decode side. Beside it, when
`PHI_PREFILL=1`, a **prefill engine** takes every completion task whose
prompt is long enough and not already held by a slot or the prompt
cache: the task is deferred, the prompt is computed apart (all but its
last token), the finished sequence's state is put into the server's own
prompt cache as that prompt's entry, and the deferred task then takes a
slot, which restores the entry (`server_prompt_cache::load`, upstream
code), computes the last token for its logits and generates. Nothing in
the response path changed: every endpoint, streaming, tool parsing, the
Anthropic and OpenAI formats are upstream's.

The engine has two forms:

- **Client** (`PHI_PREFILL_URL` set): the prompt goes to a second
  llama-server, the *prefill server*, started on the GPU placement with
  `--slot-save-path`. The engine posts the tokens (`/completion`,
  `n_predict` 0, slot 0), asks for the slot to be saved
  (`/slots/0?action=save`), reads the file (`LLAMA_STATE_SEQ_MAGIC`,
  version, the slot's packed tokens, then exactly the bytes of
  `llama_state_seq_get_data`) from `PHI_PREFILL_DIR`, checks the tokens
  against what it sent, and hands the state over. The two processes
  share no memory, so a backend in the decode server (the cards') never
  runs beside the prefill's: the rack's form.
- **In-process** (no URL): the engine loads the same model file a second
  time in the server's process with its own placement (`PHI_PREFILL_TS`,
  `PHI_PREFILL_OT`, `PHI_PREFILL_NGL`, `PHI_PREFILL_BATCH`,
  `PHI_PREFILL_UBATCH`, `PHI_PREFILL_THREADS`, `PHI_PREFILL_KV_HOST`,
  `PHI_PREFILL_DEVICES`) and computes on its own thread. One process,
  one model file read, but no second backend beside it (the cards'
  backend crashed in this form; `docs/phi/results.md`).

Both forms: `PHI_PREFILL_MIN` (prompts shorter than this stay with the
server, 64), `--cache-ram` on (`-1` for no limit), `--kv-unified` and
the same K and V types on both contexts (a state imports only into the
same layout). The prefill server's slot belongs to the engine alone.

### Prefetch: a prompt read ahead of its completion

A client that keeps one live sequence in a pinned slot (Phi Stream's
harness, slot 1) gets its input while the slot is generating. It hands
the input over at once, without a slot or a task, and the GPUs read it
while the slot keeps generating:

```
POST /phi/prefetch  {"tokens": [...]}        -> {"id": N, "n_tokens": n}   at once
GET  /phi/prefetch?id=N                      -> {"id": N, "state": "waiting"}
                                                "waiting"  queued or being read
                                                "ready"    computed; in the prompt cache, or there with the next completion
                                                "done"     the state left the cache: a slot loaded it, a longer prompt replaced it, or the cap evicted it
                                                "failed"   plus "error": the reason; a completion reads the prompt itself
                                                404        an unknown id
```

The tokens are the live history up to the input, then the input; the
state holds all but the last of them (as the engine's own prefills do).
The client then sends its completion with those tokens followed by
whatever the slot generated meanwhile, and the slot reads only that
last token plus the generated ones. Two rules in `server-context.cpp`
make the slot take the state: every completion task drains the engine's
finished states into the prompt cache before a slot is chosen, and a
slot picked by id that holds a shorter prefix of the task than the cache
does (its own earlier turn plus what it generated, against the task's
prompt itself) saves itself and loads the entry (`get_available_slot`;
upstream only looks at the cache when the slot would lose half its
context, and a hybrid memory that must drop tokens past the common
prefix cannot rewind: it reads the prompt again from a checkpoint or
from zero). A completion that arrives while a job in flight covers its
prompt waits for that job instead of reading the prompt itself. Ids are
in their own range (from 2^30), never a task's.

## The rack (`scripts/phi-serve.sh`)

One command starts both servers and stops both on Ctrl-C:

```
scripts/phi-serve.sh              # decode on port 8001 (two slots), prefill on 8002
scripts/phi-smoke.sh              # a 6000-token prompt and a short request at once
PHI_CARDS=0 scripts/phi-serve.sh  # decode on the CPU alone, no cards
```

Prefill server: experts of blocks 0 to 23 on the four GPUs (`6/6/6/30`,
room for 2048-token compute buffers), blocks 24 to 47 and the per-layer
embeddings in host memory, batch 4096, ubatch 2048, 24 threads, KV in
host memory, states saved under `/mnt/raid0/phi-prefill`. Decode server:
the model on the CPU (`-ngl 0`) under the AVX-512 backend
(`phi-ggml.sh`, `PHI_GGML_TG_ONLY=1`: the four cards take the generation
steps' multiplies, never a prompt's), 12 host threads, the prompt cache
capped at `PHI_CACHE_RAM` MiB (24576: a state is 22 KiB a token, 2.1 GiB
at 99k tokens, and a slot saves its own state before it loads a better
entry; the oldest entry goes at the cap, an entry that is a prefix of a
newer one goes at once). The numbers behind each choice and every
command are in `docs/phi/results.md`.

## Build

```
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server
```
