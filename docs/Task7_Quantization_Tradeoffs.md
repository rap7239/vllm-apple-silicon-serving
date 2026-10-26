# Task 7 — 4-bit vs 8-bit MLX Quantization Tradeoffs

**Plan reference:** Phase 1, Task 7 — originally "bf16 vs fp8 comparison." Reframed
to **4-bit vs 8-bit MLX quantization**, since vllm-metal (this project's Apple
Silicon backend) supports neither bf16 nor fp8. Same underlying question the
plan is after — what do you give up and gain by quantizing more aggressively —
answered with the formats this hardware can actually run.

---

## 1. Problem

Quantization shrinks a model's weights from their original precision down to
fewer bits per parameter, trading some output quality for less memory and
(usually) faster inference. The question this task answers: on this specific
hardware, model, and serving stack, **what exactly do you get and give up**
moving from 4-bit to 8-bit — in memory, latency, and throughput terms an
engineer would actually need to make a real deployment decision?

## 2. Constraints

- **Hardware:** M4 Mac mini, 17.2GB total unified memory (shared CPU/GPU pool
  — no discrete VRAM).
- **Model:** `mlx-community/Mistral-7B-Instruct-v0.3`, compared as its `-4bit`
  and `-8bit` MLX-quantized checkpoints.
- **Serving stack:** vllm-metal (vLLM's engine, MLX/Metal compute backend),
  `--max-model-len 2048`, `--gpu-memory-utilization 0.92` — identical config
  for both variants, only the checkpoint differs.
- **Benchmark harness:** this project's own `vllm-benchmark` (Task 3),
  streaming TTFT/TPOT measurement via `/metrics`, 20-prompt merged dataset
  (Task 4), tested at concurrency 1 and 10.

## 3. Live proof — memory footprint

Captured directly from each server's own startup log
(`Paged attention memory breakdown:`), not estimated:

| | 4-bit | 8-bit |
|---|---|---|
| `metal_limit` (usable Metal budget) | 12.71GB | 12.71GB |
| `usable_metal` (after utilization fraction) | 11.70GB | 11.70GB |
| `model_memory` (weights) | **4.08GB** | **7.70GB** |
| `kv_budget` (what's left for KV cache) | **7.06GB** | **3.28GB** |
| `num_blocks` / `max_tokens_cached` | 3,366 blocks / 53,856 tokens | 1,564 blocks / 25,024 tokens |
| Max concurrency @ 2,048 tokens/request | **26.30x** | **12.22x** |

8-bit's weights take almost double 4-bit's memory (roughly 2 bytes/param vs
~1 byte/param, as expected). Because `model_memory` and `kv_budget` are drawn
from the same fixed 11.70GB pool, that extra weight memory comes directly out
of the KV cache — 8-bit ends up with **less than half** the cacheable-token
budget, which **directly halves the practical concurrent-request ceiling**
before the scheduler has to start queueing or evicting. This is the single
most consequential number in this comparison: it's not just "8-bit is a bit
slower," it's "8-bit can serve about half as many concurrent long-context
users on the same hardware."

## 4. Live proof — latency and throughput

`vllm-benchmark/scripts/analyze.py` run against both quant levels at
concurrency 1 and 10 (warmup-cluster requests excluded per the statistical
detector documented in `PHASE1_LOG.md`):

| Variant | TTFT p50 | TTFT p95 | TTFT p99 | TPOT p50 | Aggregate tok/s |
|---|---|---|---|---|---|
| 4-bit, concurrency=1 | 273ms | 364ms | 545ms | 65.9ms | 282.4 |
| 8-bit, concurrency=1 | 379ms | 1,089ms | **23,366ms** | 104.7ms | 135.9 |
| 4-bit, concurrency=10 | 728ms | 785ms | 864ms | 209.7ms | 100.3 |
| 8-bit, concurrency=10 | 767ms | 1,223ms | 1,224ms | 222.6ms | 117.1 |

Chart: `vllm-benchmark/charts/saturation_curve.png`.

> **Correction (2026-10-13): the aggregate tok/s column above is wrong**,
> overstated by `analyze.py` dividing by the longest single request instead
> of the run's wall-clock span (see the correction note in `PHASE1_LOG.md`'s
> Task 4 close-out). Recomputed from the same CSVs:
>
> | Variant | Aggregate tok/s |
> |---|---|
> | 4-bit, concurrency=1 | 14.8 |
> | 8-bit, concurrency=1 | 8.8 |
> | 4-bit, concurrency=10 | 36.7 |
> | 8-bit, concurrency=10 | 41.2 |
>
> So at concurrency 1, 8-bit delivers about 40% less throughput than 4-bit
> (not "roughly halves", and not 282.4 → 135.9), consistent with the TPOT
> gap. At concurrency 10, 8-bit is slightly *ahead* (41.2 vs 36.7); the
> "non-monotonic dip" the caveat in section 6 relies on was an artefact of
> the same bug, so that caveat no longer applies as written. TTFT and TPOT
> figures are unaffected.

## 5. Internals — why the numbers look this way

- **TPOT (time per output token) tracks weight bytes read, plus a fixed
  serving cost.** Decode is memory-bandwidth-bound: every output token reads
  all the weights once. 8-bit reads 1.89x the bytes of 4-bit (7.70 vs
  4.08GB), so bandwidth alone predicts ~89% higher TPOT. Through the server
  it was 59% higher (104.7 vs 65.9ms). The raw-MLX check below shows why:
  vLLM-metal adds ~23-28ms per token to both checkpoints, and a fixed cost
  shrinks the ratio. (An earlier version of this section said 8-bit needs
  more compute per matrix multiply. That was backwards: 4-bit does *more*
  arithmetic per byte, to unpack and rescale its weights; 8-bit is slower
  because it moves more bytes.)
- **Aggregate throughput at concurrency=1 drops by ~40%** (14.8 → 8.8
  tok/s, the corrected figures in section 4), consistent with the TPOT gap:
  with one request in flight, throughput is roughly 1000 / TPOT.
- **KV budget, not raw compute, is what caps concurrency.** The
  `max_tokens_cached` numbers above (53,856 vs 25,024) are what actually
  determine how many simultaneous long-context requests the server can hold
  before it has to start queueing — this is a memory-capacity ceiling, not a
  latency curve, and it's the more operationally important number for a
  production deployment decision than any single percentile above.

### Raw-MLX check: is the gap the server or the kernels? (added 2026-10-26)

If decode is bandwidth-bound, why is 8-bit only 59% slower? Same checkpoints
run through `mlx_lm` directly, no server (mlx_lm 0.31.3, MLX 0.32.0, one
prompt, 256 output tokens, temperature 0, 3 interleaved warm runs each,
median; Docker quit to avoid swap):

| | 4-bit | 8-bit | Ratio |
|---|---|---|---|
| TPOT via vLLM-metal (section 4) | 65.9ms | 104.7ms | 1.59x |
| TPOT, raw `mlx_lm` | 42.4ms | 76.3ms | 1.80x |
| Achieved bandwidth, raw (weight bytes / TPOT) | ~96GB/s | ~101GB/s | |
| Share of the M4's 120GB/s | ~80% | ~84% | |

- Without the server, both checkpoints run close to the bandwidth limit and
  the ratio (1.80x) is near the 1.89x that bytes alone predict.
- Fitting TPOT = fixed + bytes / bandwidth to the two raw points gives
  ~107GB/s and ~4ms fixed per step. So any 4-bit-specific dequantization or
  dispatch cost is at most ~4ms per step here; most of the compression to
  1.59x comes from vLLM-metal's per-token overhead.
- Cold first runs were ~15% slower (20.2 / 12.1 tok/s, ratio 1.67x) and
  would have pointed to a larger 4-bit penalty. Only warm runs are used.
- Limits: one prompt, two bit widths, three runs each. What the remaining
  ~4ms is (dequantization, dispatch, attention, sampling) needs a Metal
  trace; see section 7.

## 6. Engineering judgment — design decisions, tradeoffs, and honest caveats

**When 4-bit wins:** memory-constrained or high-concurrency deployments where
maximizing simultaneous users/context length per GPU matters more than
per-request output fidelity — edge deployments, cost-sensitive serving,
scenarios where the quality delta between 4-bit and 8-bit is acceptable for
the task.

**When 8-bit is worth it:** lower-concurrency or single-user scenarios where
memory headroom isn't the binding constraint and higher output fidelity is
worth ~1.5-2x the memory and per-token latency cost. Important honest gap:
**this task did not directly measure output quality** (no perplexity or eval
harness was run) — the "8-bit is higher fidelity" side of this tradeoff rests
on the general, well-established property of less-aggressive quantization,
not on a measurement taken in this project. Flagged as a real limitation of
this comparison, not glossed over.

**Caveat on the 8-bit p99 number above (23,366ms):** this is very likely
**not representative of 8-bit's true steady-state tail latency**. It's almost
certainly contaminated by the idle-recovery phenomenon investigated at length
this same session (see `PHASE1_LOG.md`, "Task 7 — reframed to 4-bit vs 8-bit
MLX quantization, and the idle-recovery investigation"): a sequential
20-prompt run at concurrency=1 produces real gaps between requests, which is
exactly the condition that triggered idle-recovery TTFT spikes of similar
magnitude (2x-37x baseline) in that dedicated investigation. Reporting this
p99 without that context would misrepresent 8-bit's real-world tail latency —
it's presented here transparently rather than smoothed over or silently
excluded.

**Caveat on the concurrency=10 throughput result (100.3 vs 117.1 tok/s):**
this sits inside a concurrency region already flagged in Task 4 as a
non-monotonic dip for the 4-bit model (aggregate throughput at
concurrency=10 measured lower than both concurrency=1 and concurrency=25 in
that earlier sweep). This single comparison point shouldn't be read as
"8-bit beats 4-bit in throughput" — both numbers likely sit within the same
noisy regime, not a clean signal either direction.

**Caveat on scope:** idle-recovery was only deeply investigated on the 8-bit
model. Whether 4-bit exhibits the same phenomenon was not tested here and is
flagged as open follow-up work, not assumed away.

## 7. Next experiment

- Root-cause the scheduling/dispatch-layer stall behind idle-recovery
  (identified but not resolved — see `PHASE1_LOG.md`).
- Test whether 4-bit shows the same idle-recovery pattern as 8-bit.
- Add a real output-quality comparison (perplexity or a small eval set) to
  complete the tradeoff picture beyond memory/latency/throughput.
- Investigate the session-level latency drift observed during the
  idle-recovery replications (thermal-throttling hypothesis, untested).
- Metal trace of one decode step per checkpoint, to attribute the ~4ms
  per-step cost left in raw MLX, and profile where vLLM-metal's ~23-28ms
  per token goes (scheduler, sampling, streaming).

## Sources

- Memory numbers: direct server startup logs, both variants, this session.
- Latency/throughput numbers: `vllm-benchmark/scripts/analyze.py` output
  against `runs/concurrency1.csv`, `runs/8bit-concurrency1.csv`,
  `runs/concurrency10.csv`, `runs/8bit-concurrency10.csv` (2026-08-04).
- Idle-recovery findings: `PHASE1_LOG.md`, Task 7 entry (2026-08-04).
