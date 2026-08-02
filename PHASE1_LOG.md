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

## Entry: Building `vllm-benchmark` — streaming harness, wrong tree, and a cold-start TTFT outlier

**Date:** 2026-08-02

### What I set out to do

Phase 1 Task 3: hook vLLM into a benchmark harness, extending the log schema
with TPOT, ITL, `gpu_util_pct`, `kv_cache_used_pct`, and `queue_depth`. An
existing harness (`llm-benchmark-module1`, a Coursera lab) already logged
`ttft_ms` / `total_ms` / `tokens_per_sec` / `cost_per_task`, but against
OpenAI's hosted API, non-streaming, sequential — it approximated
`ttft_ms` as total request latency, which cannot separate prefill from
decode. Decision: don't retrofit that harness. Build a new one, purpose-built
for a local streaming vLLM target, and keep the old harness only as a
reference for field naming.

### Design decisions made before writing code

- **Streaming-only**, not a `--stream/--no-stream` flag. True TTFT/TPOT/ITL
  requires per-token timestamps, which only streaming provides.
- **Metrics via vLLM's `/metrics` Prometheus endpoint**, not stdout log
  regex. Same mechanism the later Grafana task will use — not throwaway
  code.
- **New standalone repo**, not a folder inside `vllm-apple-silicon-serving`.
  The master plan itself names Phase 1–8 deliverables under an
  `llm-serving-lab/*` convention.

### Repo naming: plan says `org/repo`, GitHub said no

The master plan's literal naming (`llm-serving-lab/vllm-benchmark`) reads
like an org-scoped repo path. Tried to create it under a personal account —
GitHub auto-collapsed it to `llm-serving-lab-vllm-benchmark`, since `/` isn't
a legal character in a personal-account repo name. Considered just accepting
the hyphenated name (zero setup) vs. creating a real `llm-serving-lab`
GitHub organization (true `org/repo` URLs, a real landing page at
`github.com/llm-serving-lab`, closer portfolio signal to a recruiter looking
at 5+ repos over the next 22 weeks). Went with the real org — worth the
five minutes of setup for a project this size.

### Verification-first didn't stop applying just because the harness worked

The harness ran clean on the first real attempt against the live server —
5 requests, 5 ok, 0 errors, real non-zero TTFT/TPOT numbers per request.
Easy to declare victory there. But all three scheduler metrics
(`num_requests_running`, `num_requests_waiting`, `kv_cache_usage_perc`) read
`0.0` across every row, and "concurrency=1 plausibly explains all-zeros" is
exactly the kind of unverified assumption the `--max-num-seqs` entry in this
log already burned time on. Ran `curl http://localhost:8000/metrics | grep
vllm:` directly against the idle server instead of trusting the plausible
story. Confirmed: metric names and format matched the harness's regex
exactly, and `0.0` was a genuinely accurate idle-state reading, not a
parsing failure silently returning `None`-as-zero. Correct outcome, but only
because it was checked rather than assumed.

### The 21-second TTFT outlier — real signal, not a bug

First run's per-request TTFT values: prompt 3 at 21,423ms, prompts 1/4/2/5
in the 295ms–1.7s range. `asyncio.as_completed` yields results in completion
order, not submission order — prompt 3 was submitted first and happened to
finish last. Cross-referenced against the vLLM server log: "Avg prompt
throughput: 1.5 tokens/s" on the very first request, consistent with the
MLX Metal-shader JIT-compile cold-start cost already documented in the
`--max-num-seqs` entry above. Lesson for the eventual writeup: TTFT as
measured isn't purely "prompt processing time" — it also absorbs whatever
one-time or queueing cost happens to land on that specific request. A
saturation-curve run (the next Phase 1 task) needs either a warm-up request
excluded from the stats, or an explicit note that request 1 in any cold-start
run is not comparable to steady-state TTFT.

### Open item carried forward

`vllm:num_requests_waiting_by_reason` (splits waiting requests into
`capacity` vs `deferred`) exists in this vLLM version's `/metrics` output
but isn't captured by the harness yet. Not needed for the saturation-curve
task, but relevant once the plan reaches admission-control/backpressure
work later in Phase 1/3 — that label is exactly what distinguishes "queue is
full" from "something else is blocking scheduling."

## Meta-lesson for the writeup

Same verification discipline as the previous entry, applied in a new spot:
a clean first run with plausible-looking output (all-zero metrics at
concurrency=1) is not the same as a verified run. The five-second `curl`
check cost almost nothing and turned an assumption into a fact.
