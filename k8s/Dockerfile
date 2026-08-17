# Phase 2 environment adaptation (2026-08-13 decision, see README): the
# native server this project actually runs on the M4 Mac mini uses
# vllm-metal, a macOS/Metal-only wheel that cannot run inside a Linux
# container. This image runs real, upstream vLLM instead, built for
# CPU-only inference (no Metal acceleration) — genuine inference, just
# without the GPU-equivalent speed. Not a stub/mock server.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models

# vLLM's CPU backend's native extension (vllm._C) links against libnuma at
# runtime for NUMA-aware memory placement. python:3.12-slim doesn't ship it,
# which makes the import silently fail (torch then can't find
# torch.ops._C.init_cpu_memory_env, crashing the worker process later with
# no useful traceback of its own).
#
# g++ is needed because vLLM's CPU backend runs torch.compile/inductor for
# certain fused ops (gelu, rms_norm, etc.) regardless of --enforce-eager —
# that flag only skips CUDA graph capture, not this compilation path.
RUN apt-get update && apt-get install -y --no-install-recommends libnuma1 g++ \
    && rm -rf /var/lib/apt/lists/*

# vLLM's official CPU wheel for Linux/aarch64 (GitHub release asset, not on
# PyPI — PyPI only publishes the default CUDA-targeted manylinux wheel).
# It depends on torch==2.11.0+cpu, which itself only lives on PyTorch's own
# wheel index (not PyPI), hence --extra-index-url below.
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0%2Bcpu-cp38-abi3-manylinux_2_34_aarch64.whl

# Bake the model into the image at build time so pod startup doesn't need
# outbound network access and readiness probes see a bounded, repeatable
# startup time.
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')"

# Only take the pod offline *after* the model is baked in above.
ENV HF_HUB_OFFLINE=1

EXPOSE 8000

# --enforce-eager skips CUDA-graph-equivalent capture (kept for a smaller,
# more predictable startup); g++ above handles the separate torch.compile
# path this flag does NOT skip on the CPU backend.
# On vLLM's CPU backend, --gpu-memory-utilization is repurposed to mean
# "fraction of host memory to reserve" (default ~0.9) rather than GPU VRAM.
# The kind worker node only has ~7.75GiB total with a few GiB already
# claimed by other pods/system overhead, so the default wildly overshoots
# what's actually free — capped here well under the container's own 7Gi
# memory limit.
ENTRYPOINT ["vllm", "serve", "Qwen/Qwen2.5-0.5B-Instruct", "--host", "0.0.0.0", "--port", "8000", "--gpu-memory-utilization", "0.4", "--enforce-eager"]
