#!/usr/bin/env bash
set -u

ENGINE_PID=48069
OUT="stall_dumps/native_sample_$(date +%H%M%S).txt"
mkdir -p stall_dumps

echo "Priming sudo credentials first, so the real sample later isn't blocked on a password prompt..."
sudo -v

RESP="/tmp/stall_test_response_$(date +%H%M%S).json"
curl -s -w "\nHTTP:%{http_code}\n" \
  -H "Authorization: Bearer local-vllm-key" \
  -H "Content-Type: application/json" \
  http://127.0.0.1:8000/v1/chat/completions \
  -d '{"model":"mlx-community/Mistral-7B-Instruct-v0.3-4bit","messages":[{"role":"user","content":"Write three paragraphs about the history of GPUs."}],"max_tokens":300}' \
  -o "$RESP" &
CURL_PID=$!
START_EPOCH=$(date +%s)

echo "Waiting 10s to make sure we're past initial admission..."
sleep 10

# ps -p (not kill -0) avoids the zombie-PID false positive: a finished-but-
# unreaped background job can still make `kill -0` succeed.
if ps -p "$CURL_PID" > /dev/null 2>&1; then
  echo "Request confirmed still in flight ($(( $(date +%s) - START_EPOCH ))s elapsed) -- sampling now..."
  sudo sample "$ENGINE_PID" 45 -f "$OUT"
  echo "--- native sample saved to $OUT ---"
else
  echo "Request already finished before we could sample (took under 10s) -- no stall this time, rerun to try again."
fi

wait "$CURL_PID"
echo "--- request finished, total elapsed: $(( $(date +%s) - START_EPOCH ))s ---"
ls -la "$RESP"
