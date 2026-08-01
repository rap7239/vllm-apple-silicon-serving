#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-mlx-community/Mistral-7B-Instruct-v0.3-4bit}"
VLLM_API_KEY="${VLLM_API_KEY:-local-vllm-key}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
VLLM_BIN="${VLLM_BIN:-.venv-metal/bin/vllm}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"

# Keep single-node distributed initialization on the loopback interface.
# Without these defaults, macOS may select a transient/private interface that
# the local worker cannot connect back to.
export VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo0}"

if [[ ! -x "$VLLM_BIN" ]]; then
  echo "vLLM is not installed at $VLLM_BIN" >&2
  echo "Run this script from the project root after completing installation." >&2
  exit 1
fi

ARGS=(serve "$MODEL"
  --dtype auto
  --host 127.0.0.1
  --port 8000
  --api-key "$VLLM_API_KEY"
  --max-model-len "$MAX_MODEL_LEN")

if [[ -n "$GPU_MEMORY_UTILIZATION" ]]; then
  ARGS+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
fi

exec "$VLLM_BIN" "${ARGS[@]}"
