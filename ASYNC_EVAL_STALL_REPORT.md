# `mx.async_eval()` occasionally blocks for tens of seconds to minutes after idle, with the GPU doing no measurable work

**Suggested title for filing:** `mx.async_eval() blocks in eval_impl's condition_variable::wait for 30s-6min+ after the engine sits idle, GPU shows no activity during the stall`

**Where to file:** primarily [`vllm-project/vllm-metal`](https://github.com/vllm-project/vllm-metal) (where this was first observed, in `vllm_metal/v1/model_runner.py`), and worth cross-posting to [`ml-explore/mlx`](https://github.com/ml-explore/mlx) since the actual blocking call lives inside MLX's own native code (`libmlx.dylib`), not vllm-metal's Python layer. Same pattern as this project's two prior upstream reports ([`vllm-metal#274`](https://github.com/vllm-project/vllm-metal/issues/274), [`tvm-ffi#697`](https://github.com/apache/tvm-ffi/issues/697)/[`xgrammar#799`](https://github.com/mlc-ai/xgrammar/issues/799)) — see this repo's `README.md`.

## Environment

- Hardware: Mac mini, `Mac16,10` (M4)
- OS: macOS 26.6 (build 25G72)
- Python: 3.12.12
- `vllm-metal`: 0.3.0.dev20260730125022
- `mlx`: 0.32.0
- Model: `mlx-community/Mistral-7B-Instruct-v0.3-4bit` (also observed with the 8-bit build of the same model), served via `vllm serve` with `--max-model-len 2048`

## Summary

After the vLLM engine sits idle for a while (observed anywhere from ~8 minutes to ~11 hours — no clean minimum threshold found), the next request's first token can take anywhere from ~30 seconds to several minutes to arrive, sometimes hitting the client's read timeout entirely. Root-caused this to `mx.async_eval()`, called from `vllm_metal/v1/model_runner.py`'s `_submit_paged_forward_outputs`, blocking inside MLX's native `eval_impl` on a genuine `std::condition_variable::wait` for the full duration of the stall. Real-time GPU power telemetry confirms the GPU itself does essentially no work for the vast majority of the stall — it snaps to full activity only in the final 1-3 seconds, right as the token is actually delivered.

## Reproduction

1. Serve any model via `vllm-metal` (`vllm serve <model> ...`).
2. Let the engine sit completely idle (no requests) for at least several minutes — the exact minimum wasn't pinned down precisely; observed instances at ~8-9 minutes, ~11 hours, and points in between, but not at very short (<1 minute) idle gaps.
3. Send a single request (any prompt, streaming or not).
4. Observe: TTFT is dramatically elevated compared to a "warm" baseline (typically 200ms-3s once warm), sometimes reaching 30-115+ seconds, occasionally exceeding a 120s client read timeout entirely.

A single throwaway "warmup" request immediately before the real one reliably works around it — the warmup itself eats the stall, and anything sent right after comes back at normal latency. This is the workaround currently in use in this project's own benchmarking scripts.

## Evidence

### 1. GPU power telemetry (real, continuous sampling — not per-request snapshots)

Sampling `powermetrics`' GPU power reading once per second across a full stalled request:

- ~95 of ~115 total seconds sat at 3-290 mW (indistinguishable from true idle baseline, independently measured at 52 mW).
- Power then jumped to 6,700-7,600 mW for ~19 seconds — real, active compute — right before the response completed.

In a separate, longer capture (a ~28.8s-worst-case instance from an earlier idle-recovery investigation on the same stack): GPU frequency sat flat at idle baseline (~750-800 MHz) for 25+ of the 28.8 seconds, then snapped to max observed frequency (~1,576-1,578 MHz) in about one second, timed to token delivery.

**This rules out the GPU itself being slow to wake up** — if it were, frequency/power would ramp up gradually over the stall, not stay flat and then jump instantaneously right at the end.

### 2. Python-level stack trace (`py-spy dump`, both vLLM processes, repeated every 5s across a live stall)

vLLM runs as two processes: an API server (asyncio/HTTP) and a separate `EngineCore` process (scheduling + MLX/Metal dispatch). Across 33 consecutive dumps of `EngineCore`, spanning 5.5 minutes of a reproduced stall, the trace never moved from one exact line:

```
Thread 0x... (active): "MainThread"
    _submit_paged_forward_outputs (vllm_metal/v1/model_runner.py:582)
    _start_paged_forward (vllm_metal/v1/model_runner.py:1185)
    execute_model (vllm_metal/v1/model_runner.py:2351)
    execute_model (vllm_metal/v1/worker.py:279)
    ...
    run_busy_loop (vllm/v1/engine/core.py:1364)
```

`model_runner.py:582` is:
```python
def _submit_paged_forward_outputs(self, *outputs: mx.array) -> None:
    eval_outputs = list(outputs)
    runtime = self._paged_attention_runtime
    if runtime is not None:
        runtime.extend_forward_eval_outputs(eval_outputs)
    mx.async_eval(*eval_outputs)   # <-- line 582, where the thread was stuck
```

### 3. Native (C/C++) stack trace (macOS's `sample` tool, one layer beneath what `py-spy` can see)

`py-spy` can't see past a call into native code. macOS's built-in `sample` tool can. A verified-valid capture (confirmed by cross-checking file timestamps against the actual request duration, since a first attempt was invalidated by a stale `sudo` prompt — see caveat below) shows, for **100% of 38,629 samples across a full 45-second window**, on `EngineCore`'s main thread:

```
mlx::core::async_eval(std::vector<mlx::core::array>)  (in libmlx.dylib)
  mlx::core::eval_impl(std::vector<mlx::core::array>, bool)  (in libmlx.dylib)
    std::condition_variable::wait(std::unique_lock<std::mutex>&)  (in libc++.1.dylib)
      _pthread_cond_wait
        __psynch_cvwait
```

This is a genuine, real condition-variable wait inside MLX's own native `eval_impl` — not a busy spin, not the GIL, not a Python-level lock, not `asyncio`. The thread is legitimately blocked waiting to be signaled.

At the same time, MLX's own internal `ThreadPool`/`StreamThread` worker threads (also visible in the same capture) are sitting idle on their *own* separate condition variables, waiting for work — consistent with `eval_impl` waiting on one of these workers to actually pick up and dispatch the queued computation, and that worker simply not getting scheduled promptly.

### 4. Ruled out: `caffeinate`

Restarted the server under `caffeinate -i` (prevents macOS idle-sleep assertions) to test whether OS-level idle throttling was the cause. The stall still occurred (63s on a fresh cold-start request under `caffeinate`), so simple system-sleep prevention is not sufficient to fix this — though the exact mechanism (if it is OS scheduling-related at all) may need process/thread-level QoS intervention rather than a system-wide sleep assertion, which wasn't confirmed further (see caveat below).

## Related reports

- [`ollama/ollama#16170`](https://github.com/ollama/ollama/issues/16170) — "Severe inter-prompt delay regression on Apple Silicon MLX," filed against a completely different serving stack (Ollama, not `vllm-metal`) on different hardware (M3 Ultra vs. this report's M4), but describing what looks like the same underlying issue: a delay specifically *between* requests (not during generation), CPU near-idle throughout the stall, and the reporter's own hypothesis — *"suggesting a synchronization / scheduler / MLX handoff issue rather than raw inference slowdown"* — closely matching this report's conclusion, reached independently. That issue is a symptom report with no maintainer response and no proposed mechanism; this report is offered as a candidate root-cause diagnosis for the same underlying class of problem, reproduced independently on a different stack.
- **Not the same as** [`ml-explore/mlx#1571`](https://github.com/ml-explore/mlx/discussions/1571) (timing difference between `mx.eval` and `mx.async_eval`) — that discussion is about a different, already-understood phenomenon (first-call JIT/compilation overhead on a cold function). The stall described here occurs on an *already-warm* engine that has served requests successfully before, purely after an idle gap, which is a structurally different trigger.

## What this is *not*

- Not the JIT/shader-compile cold start already documented elsewhere (that's a genuine one-time cost, confirmed via server logs showing steady low-throughput generation, not this pattern of complete GPU silence followed by a sudden burst).
- Not a harness-side measurement artifact — confirmed via dedicated harness instrumentation (`queue_wait_ms`/`connect_wait_ms` fields) showing near-zero client-side queueing in every instance; the delay is genuinely server-side.
- Not simply "the model needs to warm up batched execution" (a separate, real, one-time ~22s cost already documented) — this stall recurs repeatedly, is much larger in the worst cases, and happens even for single, unbatched requests.

## Open question / where this investigation stopped

The most likely remaining explanation is that `eval_impl`'s condition variable is waiting on an MLX-internal worker thread, and that thread isn't being scheduled promptly by macOS after the process has been idle — but this was not directly confirmed. Attempted to check actual thread QoS/scheduling-class values via `powermetrics --samplers tasks` (no QoS field in that sampler's output) and `ps` (STAT flags constant across fresh-start and long-idle states, not itself conclusive). The tool that could show this directly — Instruments' Thread State trace template — requires full Xcode, not installed on this machine; decided the marginal value didn't justify a 10+ GB install given how strong the existing evidence already is.

## Caveat, in the interest of full transparency

An earlier attempt at capturing a native stack trace produced a misleading result — the profiler had actually run nearly an hour after the request it was meant to be sampling had already completed (caused by a combination of an expired `sudo` credential blocking on a password prompt, and a process-liveness check that can false-positive on an unreaped zombie PID). Caught by cross-checking the response file's mtime against the profiler's own report timestamp before trusting it; discarded and re-captured correctly (that's the trace shown in section 3 above). Mentioning this not because it affects the finding, but because reproducing this investigation should account for the same trap.

## Raw evidence available on request

Full `py-spy` dumps (74 files across two live-stall reproductions) and native `sample` captures are preserved in this project's own repo history (`stall_dumps/`, commits `2e95a28` and `4a25ef3` in `vllm-apple-silicon-serving`), along with the continuous GPU power time-series (`burst_timeseries.csv`). Happy to attach specific excerpts or the raw files if useful for triage.
