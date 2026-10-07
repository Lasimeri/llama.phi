#!/bin/bash
# phi-serve.sh: llama.phi's server on the rack. The decode side (llama-server
# itself) runs the model on the CPU, the prefill engine on the four GPUs
# (docs/phi/results.md for the numbers behind each choice). Options after
# the script name go to llama-server (default: port 8001, two slots).
#   phi-serve.sh            start in the foreground
#   phi-serve.sh -np 4 ...  more slots, other server options
set -u
root=$(cd "$(dirname "$0")/.." && pwd)
M=${PHI_MODEL:-/mnt/raid0/Qwen3.8-Flash-Next-GGUF/UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf}
# The prefill model: experts of blocks 0..23 on the GPUs (room left for
# 2048-token compute buffers), the rest and the per-layer embeddings in host
# memory, big batches, its KV in host memory (the state moves to the decode
# context, which keeps its own there).
export PHI_PREFILL=${PHI_PREFILL:-1}
export PHI_PREFILL_TS=${PHI_PREFILL_TS:-6/6/6/30}
export PHI_PREFILL_OT=${PHI_PREFILL_OT:-'blk\.(2[4-9]|3[0-9]|4[0-7])\.ffn_(up|gate|down)_exps\.weight=CPU,per_layer_token_embd\.weight=CPU'}
export PHI_PREFILL_NGL=${PHI_PREFILL_NGL:-999}
export PHI_PREFILL_BATCH=${PHI_PREFILL_BATCH:-4096}
export PHI_PREFILL_UBATCH=${PHI_PREFILL_UBATCH:-2048}
export PHI_PREFILL_THREADS=${PHI_PREFILL_THREADS:-24}
export PHI_PREFILL_KV_HOST=1
export PHI_PREFILL_MIN=${PHI_PREFILL_MIN:-64}
# The decode side: everything on the CPU (-ngl 0; generation measured 7.6
# tok/s there against 8.0 with the GPUs, which the prefill model now fills),
# one KV stream (the prefill state imports only into the same layout), Q8 K
# and V, the prompt cache unlimited (it carries the prefilled states).
exec "$root/build/bin/llama-server" -m "$M" -ngl 0 -t ${PHI_DECODE_THREADS:-32} -tb ${PHI_DECODE_THREADS:-32} \
    -c ${PHI_CTX:-131072} -np ${PHI_SLOTS:-2} --kv-unified -fa on -ctk q8_0 -ctv q8_0 \
    --cache-ram -1 --jinja --host 0.0.0.0 --port ${PHI_PORT:-8001} "$@"
