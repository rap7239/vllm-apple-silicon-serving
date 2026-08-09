#!/usr/bin/env bash
curl -s -w "\nHTTP:%{http_code} took %{time_total}s\n" \
  -H "Authorization: Bearer local-vllm-key" \
  -H "Content-Type: application/json" \
  http://127.0.0.1:8000/v1/chat/completions \
  -d '{"model":"mlx-community/Mistral-7B-Instruct-v0.3-4bit","messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":10}'
