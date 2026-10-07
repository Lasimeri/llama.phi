#!/bin/bash
# phi-smoke.sh [PORT]: llama.phi's server exercised. A long prompt (the bench's
# calib-code.txt, about 6000 tokens) and, two seconds into it, a short request
# that generates 48 tokens: with the prefill engine the short one's tokens
# arrive while the long prompt is still being read. Prints each request's
# timings from the server's own numbers.
port=${1:-8001}
P=/mnt/raid5/phi/bench/calib-code.txt
[ -f "$P" ] || { echo "no $P" >&2; exit 1; }
until curl -sf "http://127.0.0.1:$port/health" > /dev/null; do sleep 2; done
j=$(mktemp)
jq -Rs --arg n 32 '{prompt: ., n_predict: ($n|tonumber), temperature: 0, cache_prompt: true}' "$P" > "$j"
t0=$(date +%s.%N)
curl -s "http://127.0.0.1:$port/completion" -d @"$j" > /tmp/phi-long.json &
long=$!
sleep 2
t1=$(date +%s.%N)
curl -s "http://127.0.0.1:$port/completion" -d '{"prompt":"List five prime numbers and explain why they are prime.","n_predict":48,"temperature":0,"cache_prompt":false}' > /tmp/phi-short.json
t2=$(date +%s.%N)
wait $long
t3=$(date +%s.%N)
rm -f "$j"
echo "short request (started 2 s into the long prompt): $(echo "$t2 - $t1" | bc) s wall; server: $(jq -c '.timings | {prompt_n, prompt_ms, predicted_n, predicted_ms, predicted_per_second}' /tmp/phi-short.json)"
echo "long prompt: $(echo "$t3 - $t0" | bc) s wall; server: $(jq -c '.timings | {prompt_n, prompt_ms, prompt_per_second, predicted_n, predicted_per_second}' /tmp/phi-long.json)"
echo "long answer starts: $(jq -r '.content' /tmp/phi-long.json | head -c 160)"
