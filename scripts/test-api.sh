#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL="${MODEL:-mlx-community/Mistral-7B-Instruct-v0.3-4bit}"
VLLM_API_KEY="${VLLM_API_KEY:-local-vllm-key}"

echo "Available models:"
curl --fail --silent --show-error \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  "$BASE_URL/models"
echo

echo "Chat completion:"
curl --fail --silent --show-error \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  "$BASE_URL/chat/completions" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"vLLM is an LLM serving engine whose continuous-batching scheduler can add and remove requests between decoding steps. Explain the throughput benefit in one sentence.\"}
    ],
    \"temperature\": 0.2,
    \"max_tokens\": 80
  }"
echo
