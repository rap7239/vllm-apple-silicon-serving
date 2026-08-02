# Phase 1 Log — Task 2: Core vLLM Parameters

Running narrative log of decisions, mistakes, and lessons learned while working
through Task 2 (`--tensor-parallel-size`, `--max-model-len`,
`--gpu-memory-utilization`, `--max-num-seqs`). Meant to be read later as source
material for a "lessons learned" writeup, not just a changelog.

> **Standing instruction for Claude (any session, new or continued):** at the
> end of each hands-on work session in this repo, append a new dated entry to
> this file summarizing that session's narrative — what was attempted, what
> broke, root causes, fixes, and the lesson worth remembering. Do this before
> the session is considered closed, without waiting to be reminded. If this
> file is visible to you (attached, read, or in this repo), treat this
> instruction as active regardless of what chat session you're in. Keep
> entries technical/professional — personal reflections belong in the
> local-only `NOTES_PERSONAL.md`, not here.

## Entry: `--max-num-seqs` — the flag that silently did nothing

**Date:** 2026-08-01

### What I set out to do

Test `--max-num-seqs` by capping it at 1 and firing three concurrent chat
completion requests, expecting to see vLLM's scheduler queue two of them while
one ran — visible evidence of the scheduler enforcing the cap.

### What actually happened first

`scripts/serve.sh` at the time only supported `MODEL`, `VLLM_API_KEY`,
`MAX_MODEL_LEN`, `VLLM_BIN`, and `GPU_MEMORY_UTILIZATION` as env-var
overrides. It had no `MAX_NUM_SEQS` support at all — no variable declaration,
no conditional `--max-num-seqs` flag passed to `vllm serve`.

I ran `MAX_NUM_SEQS=1 MODEL=... ./scripts/serve.sh` anyway. Bash silently
ignored the unused env var (it was never read by the script), so vLLM started
with its own internal default `max_num_seqs` (in the low hundreds). The three
concurrent requests all ran simultaneously:

```
Running: 3 reqs, Waiting: 0 reqs
```

This looked like `--max-num-seqs` "not working," but the real bug was
upstream of vLLM entirely — the script just never passed the flag.

### The pattern recognition moment

This was the *second* time in the same session that a described-but-not-yet-
applied file edit was silently relied on as if it had already happened — the
first was the README "Known issues filed upstream" section, which was missing
from the first GitHub push because the append command had been given in chat
but never actually run before the commit. Recognizing the repeat of that exact
failure mode (edit described in chat ≠ edit applied to disk) was the key
diagnostic step, not any vLLM-specific knowledge. Lesson: verify state with a
`cat`/`git diff` before trusting that a described change took effect.

### The fix

Rewrote `scripts/serve.sh` via `cat > ... << 'EOF'` heredoc to add:

```bash
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
...
if [[ -n "$MAX_NUM_SEQS" ]]; then
  ARGS+=(--max-num-seqs "$MAX_NUM_SEQS")
fi
```

Verified with `cat scripts/serve.sh` *before* restarting the server, rather
than assuming the rewrite worked.

### Re-running the experiment — confirmed working

Restarted with `MAX_NUM_SEQS=1 MODEL=mlx-community/Mistral-7B-Instruct-v0.3-4bit MAX_MODEL_LEN=2048 ./scripts/serve.sh`,
then fired the same three concurrent curl requests. Server log:

```
Running: 1 reqs, Waiting: 2 reqs, GPU KV cache usage: 0.1%, Prefix cache hit rate: 0.0%
```

Contrast with the earlier broken run (`Running: 3, Waiting: 0`) is clean,
direct evidence of `--max-num-seqs` capping the scheduler's active batch
size — same model, same three requests, only the cap differs.

### Side observation: cold-start CPU spike on restart

Noticed a sharp CPU spike (System 4.99%, User 13.65%, with a visible peak in
the CPU Load graph) right when the server restarted. This is expected, not a
regression from the `MAX_NUM_SEQS` change:

- `serve.sh` uses `exec`, so stopping the prior process fully kills it —
  nothing is warm-cached in RAM for the next run.
- Model weights are re-read from disk and re-dequantized (4-bit Mistral-7B).
- MLX JIT-compiles its Metal shader kernels fresh per-process, which is
  CPU-bound work (explains why it showed as CPU load, not a GPU indicator).
- vLLM re-parses CLI args, rebuilds the scheduler/KV-cache allocator (same
  `metal_limit`/budget arithmetic from the `--gpu-memory-utilization`
  experiment), and re-establishes the APIServer↔EngineCore IPC.

Same cold-start cost would occur on any restart, independent of which flags
changed.

### Open item: elevated swap after the experiment

Memory Pressure panel post-experiment showed 15.10GB / 16GB used, 13.03GB
swap. Worth checking for a leftover/orphaned `vllm serve` process (e.g. via
`ps aux | grep -i vllm`) before assuming this is expected steady-state memory
behavior — a `MAX_NUM_SEQS=1` run should use *less* peak memory than the
default, not more, so high swap here is a flag to investigate rather than
dismiss.

**Resolved:** `ps aux | grep -i vllm` confirmed exactly one `vllm serve`
process (with `--max-num-seqs 1` visible in its command line) and one
`VLLM::EngineCore` worker — no duplicate or orphaned PIDs. Not a zombie
process. Attributed instead to normal memory pressure at this model size:
4-bit Mistral-7B weights, KV cache, MLX-compiled kernels, and vLLM/Python
overhead together sit close to the 16GB unified-memory ceiling, and macOS
proactively swaps out colder pages for other apps rather than this
indicating a leak. No OOM crash or degraded latency observed, so treating
this as expected steady-state rather than a bug to chase further.

## Meta-lesson for the writeup

The most reusable lesson from this task isn't about vLLM internals — it's a
verification discipline: when a described fix doesn't produce the expected
result, check whether the fix was actually applied before assuming the
underlying system behaves differently than documented. Two separate incidents
in one session (README section, `serve.sh` edit) followed the identical
failure shape.
