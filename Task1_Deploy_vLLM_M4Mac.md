# Task 1 — Deploy vLLM Locally, Serve via OpenAI-Compatible API

**Plan reference:** Phase 1, Week 1-4, Task 1 — "Deploy vLLM locally with pip install vllm; serve Llama-3-8B or Mistral-7B via OpenAI-compatible API" (4h, T1)

**Platform note:** Mainline `vllm` on PyPI is CUDA-only — it won't run on an M4 Mac. The plan's literal `pip install vllm` doesn't apply here. Instead, use **vllm-metal**, the community plugin that runs vLLM on Apple Silicon using MLX as the compute backend (Metal kernels, unified memory, full vLLM engine + OpenAI-compatible server). Same deliverable, different install path.

---

## 0. Requirements

- macOS on Apple Silicon (your M4 Mac mini qualifies)
- **Native arm64 Python 3.12** — Rosetta/x86_64 Python will not work. Check with:
  ```bash
  python3.12 -c "import platform; print(platform.machine())"
  ```
  This must print `arm64`, not `x86_64`. If you don't have Python 3.12, install it via `brew install python@3.12`.
- Xcode Command Line Tools (vLLM core compiles from source via `clang++`; the Metal kernels themselves ship prebuilt):
  ```bash
  xcode-select --install
  ```

## 1. Install vllm-metal

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

This creates a self-contained venv at `~/.venv-vllm-metal` with the vllm-metal plugin, vLLM core, and dependencies.

Activate it (do this in every new terminal session you use for vLLM):

```bash
source ~/.venv-vllm-metal/bin/activate
```

You should now have the `vllm` CLI available: `vllm --version`.

## 2. Pick a model

Use a pre-quantized **MLX 4-bit** checkpoint — full-precision 7-8B weights alone are ~14-16GB, which eats your whole unified memory budget before you even load a KV cache. 4-bit brings weights down to ~4-5GB, leaving headroom.

Both models named in the plan are officially supported by vllm-metal:

| Model | Checkpoint | Notes |
|---|---|---|
| Llama 3 | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` | Gated on Hugging Face — you'll need an HF account, to accept Meta's license on the base `meta-llama` repo, and to run `huggingface-cli login` with a token first. |
| Mistral-7B | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | Not gated — simplest path for a first run. |

**Recommendation:** start with Mistral-7B to avoid the HF gating step on your first pass, then repeat with Llama-3 once the flow works (the plan wants you comfortable with either).

If you go the Llama-3 route:
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
# then accept the license at https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
```

## 3. Serve it

```bash
vllm serve mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --host 0.0.0.0 \
  --port 8000
```

First run downloads the checkpoint from Hugging Face (a few GB) — expect that to take a few minutes depending on your connection. Subsequent starts are fast (cached under `~/.cache/huggingface`).

Watch the logs for the line confirming the OpenAI-compatible server is up, e.g. `Uvicorn running on http://0.0.0.0:8000`.

## 4. Verify the OpenAI-compatible API

In a second terminal:

```bash
curl http://localhost:8000/v1/models
```

Should return the served model's ID. Then a real generation call:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "messages": [{"role": "user", "content": "In one sentence, what is PagedAttention?"}]
  }'
```

You should get back a normal OpenAI chat-completion JSON response. If that works, you can also point the official `openai` Python SDK at `base_url="http://localhost:8000/v1"` with any placeholder API key — vLLM doesn't check it locally.

## 5. Sanity checks before calling this done

- [ ] `vllm serve` starts without error and stays running
- [ ] `/v1/models` returns your model
- [ ] `/v1/chat/completions` returns a coherent completion
- [ ] You've watched Activity Monitor / `sudo powermetrics` briefly to see memory pressure and GPU (ANE/GPU) usage while a request is in flight — this is your first real signal for Task 4 (the saturation curve later in the same week)
- [ ] (Optional, recommended) Repeat steps 2-4 with the Llama-3.1-8B checkpoint so you've served both models named in the plan

## Troubleshooting

- **`vllm: command not found`** — you forgot to `source ~/.venv-vllm-metal/bin/activate` in this terminal.
- **Install/build errors mentioning `clang++`** — Xcode CLI tools aren't installed or aren't accepted yet; run `sudo xcodebuild -license` after `xcode-select --install`.
- **OOM / system becomes sluggish** — you're likely on a lower-RAM configuration or a non-4bit checkpoint slipped in; confirm the checkpoint name ends in `-4bit` and close other memory-heavy apps.
- **Gated repo 403 on Llama** — you haven't accepted the license on the `meta-llama` org page or `huggingface-cli login` wasn't run with a valid token.
- **Reinstall from scratch** — `rm -rf ~/.venv-vllm-metal && curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash`

## Sources

- [vLLM Metal plugin — GitHub](https://github.com/vllm-project/vllm-metal)
- [vllm-metal installation docs](https://docs.vllm.ai/projects/vllm-metal/en/latest/installation/)
- [vllm-metal supported models](https://docs.vllm.ai/projects/vllm-metal/en/latest/supported_models/)
