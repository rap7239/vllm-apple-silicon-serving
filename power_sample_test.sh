#!/usr/bin/env bash
set -u

curl -s -w "\nHTTP:%{http_code}\n" \
  -H "Authorization: Bearer local-vllm-key" \
  -H "Content-Type: application/json" \
  http://127.0.0.1:8000/v1/chat/completions \
  -d '{"model":"mlx-community/Mistral-7B-Instruct-v0.3-4bit","messages":[{"role":"user","content":"Write three paragraphs about the history of GPUs."}],"max_tokens":300}' \
  -o /tmp/power_test_response.json &
CURL_PID=$!

echo "Polling GPU power while request runs..."
while kill -0 "$CURL_PID" 2>/dev/null; do
  reading=$(curl -s http://127.0.0.1:9400/metrics | grep gpu_power_milliwatts)
  echo "$(date +%H:%M:%S)  $reading"
  sleep 1
done

wait "$CURL_PID"
echo "--- request finished ---"
echo "Response saved to /tmp/power_test_response.json"
tail -c 300 /tmp/power_test_response.json
