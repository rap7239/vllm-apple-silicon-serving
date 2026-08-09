#!/usr/bin/env bash
set -euo pipefail

BASE_URL="http://127.0.0.1:8000/v1"
VLLM_API_KEY="local-vllm-key"
MODEL="mlx-community/Mistral-7B-Instruct-v0.3-4bit"

fire() {
  local tag="$1" max_tokens="$2"
  curl --silent --show-error \
    -H "Authorization: Bearer $VLLM_API_KEY" \
    -H "Content-Type: application/json" \
    "$BASE_URL/chat/completions" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a short paragraph about the weather.\"}],\"temperature\":0.2,\"max_tokens\":$max_tokens}" \
    -o "/tmp/batching_${tag}.json" -w "[$tag done] http:%{http_code} time:%{time_total}s\n"
}

echo "Firing 3 requests: A(150) B(900) C(450) -- all distinct lengths"
fire A 150 &
fire B 900 &
fire C 450 &
wait
echo "All requests complete."
