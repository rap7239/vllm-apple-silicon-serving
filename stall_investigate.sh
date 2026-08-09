#!/usr/bin/env bash
set -u

API_PID=64114
ENGINE_PID=64206
OUTDIR="stall_dumps"
mkdir -p "$OUTDIR"
rm -f "$OUTDIR"/*.txt

curl -s -w "\nHTTP:%{http_code}\n" \
  -H "Authorization: Bearer local-vllm-key" \
  -H "Content-Type: application/json" \
  http://127.0.0.1:8000/v1/chat/completions \
  -d '{"model":"mlx-community/Mistral-7B-Instruct-v0.3-4bit","messages":[{"role":"user","content":"Write three paragraphs about the history of GPUs."}],"max_tokens":300}' \
  -o /tmp/stall_test_response.json &
CURL_PID=$!

i=0
echo "Dumping py-spy stacks for both processes every 5s while request runs..."
while kill -0 "$CURL_PID" 2>/dev/null; do
  i=$((i+1))
  ts=$(date +%H%M%S)
  echo "--- dump $i at $(date +%H:%M:%S) ---"
  sudo py-spy dump --pid "$API_PID" > "$OUTDIR/api_${i}_${ts}.txt" 2>&1
  sudo py-spy dump --pid "$ENGINE_PID" > "$OUTDIR/engine_${i}_${ts}.txt" 2>&1
  reading=$(curl -s http://127.0.0.1:9400/metrics | grep -E "^powermetrics_gpu_power_milliwatts ")
  echo "  power: $reading"
  sleep 5
done

wait "$CURL_PID"
echo "--- request finished, dumps saved in $OUTDIR/ ---"
