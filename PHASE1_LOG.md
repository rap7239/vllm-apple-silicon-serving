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

## Entry: Fixing a duplicated/mangled log entry — a meta-lesson about editing this very file

**Date:** 2026-08-02

### What happened

While appending the `vllm-benchmark` entry above, a stray local git clone
(a folder named "New project" whose `origin` happened to point at this same
GitHub repo, from an earlier unrelated setup step) was used to push the
entry once. Working from the *actual* local clone shortly after, unaware the
first push had already landed, the same entry was appended a second time —
producing a duplicate section in this file.

Attempting to fix the duplicate by hand in `nano` (mark-and-cut a multi-line
block) did not fully work: the cut landed mid-sentence inside the second
copy rather than cleanly between the two headings, leaving one and a half
copies of the entry spliced together, with a heading fragment
(`## Entry: Building...`) embedded mid-sentence and the first copy's own
heading demoted from `##` to `#`. `grep -c` on the heading text kept
reporting 2 matches, but the second "match" was actually that embedded
fragment, not a clean second copy — a reminder that a match count alone
doesn't tell you *what* matched.

### The fix

Rather than attempt a third manual in-place edit (each one so far had
introduced a new, different failure mode), the full file was reconstructed
from a known-good copy of the content and pasted in as a complete
replacement, verified by full read-through rather than a partial `grep`
check.

## Meta-lesson for the writeup

Three failure modes stacked on the same underlying cause: trusting a git
remote's identity without checking `pwd`/`git remote -v` first, then trusting
a partial edit's success from an incomplete signal (`grep -c` count) instead
of reading the actual resulting content. The fix that finally worked was the
one that verified the *whole* file, not a count or a spot check. Same lesson
as the two entries above, expressed a third way: verify state directly,
don't infer it from a proxy signal.

## Entry: Task 4 setup — `--repeat` flag for the saturation curve

**Date:** 2026-08-02

### What I set out to do

Start Phase 1 Task 4: ramp concurrency 1 → 5 → 10 → 25 → 50 against the vLLM
server and record p50/p95/p99 + tokens/sec at each step. Picked this up in a
new chat session after Task 3 closed out — no memory of that session's
back-and-forth, only what's recorded in this log and in the repos
themselves, which is exactly the scenario this file's standing instruction
exists for.

### The blocker carried over from the Task 3 session

`datasets/prompts.jsonl` has 5 prompts. Run once per concurrency level, that
gives 5 data points — not enough to compute a meaningful p95/p99. p99 of 5
sorted numbers is just the largest one; there's no percentile signal there,
just the max. At concurrency 25 or 50 this is worse in a second way too: with
only 5 prompts in flight, most of the concurrency slots the harness is
supposed to be testing would sit empty.

Two options considered: grow `prompts.jsonl` to a larger fixed dataset, or
add a `--repeat` flag to `run_bench.py` that cycles the existing 5 prompts N
times. Went with `--repeat` — it's a smaller, more reusable change (works at
every concurrency level without hand-tuning dataset size per run), and it
keeps the existing 5 hand-written prompts as the canonical prompt set rather
than diluting them into a larger, less-curated file.

### Implementation

Added `expand_dataset(rows, repeat)` in `run_bench.py`: cycles the loaded
rows `repeat` times, and for every pass after the first, suffixes each row's
`id` with `-r{pass_number}` (e.g. prompt `3` on pass 2 becomes `3-r2`) so
every result row stays traceable back to which prompt and which pass
produced it. `repeat=1` (the default) is a true no-op — verified this
directly rather than assuming it, given the pattern of this exact class of
bug (edit doesn't do what's assumed) recurring twice already in this log.
`--max-samples` still applies, now after expansion, so it composes as a hard
cap on top of `--repeat` if both are passed.

`analyze.py` needed no changes — it summarizes whatever rows land in the
output CSV, independent of how many there are or how they were generated.

### Verification

Ran the expansion function standalone (no server needed — this part of the
harness is pure data transformation): confirmed `repeat=1` returns the
original 5 rows unmodified, `repeat=20` produces exactly 100 rows with 100
unique ids, and `--help` output reflects the new flag correctly. Did not run
the actual saturation sweep in this session — that requires the live vLLM
server on the M4 Mac, which isn't reachable from this session's sandbox.
Code is ready; the live run against `serve.sh` is the next hands-on step.

### Run plan for the actual sweep

Documented in `vllm-benchmark/README.md`: `repeat` values chosen so total
request volume is roughly 4x the concurrency level (concurrency 1→repeat 5,
5→8, 10→10, 25→20, 50→20), so p95 has ~20 samples above it instead of being
one noisy draw. Also carried forward the Task 3 cold-start TTFT finding: the
MLX Metal-shader JIT-compile cost lands once per server process, not once
per run, so it only needs to be treated as a warm-up exclusion if the server
gets restarted between concurrency levels.

## Meta-lesson for the writeup

First real test of this log's standing instruction working as intended
across a session boundary: picked up Task 4 in a fresh chat with zero
transcript memory, and the log plus the repos themselves were enough to
reconstruct the exact open question (5 prompts, not enough volume) without
re-deriving it from scratch or asking what was previously discussed.

## Entry: Task 4 revisited — merging in the Module 1 support-chat dataset instead of relying on `--repeat` alone

**Date:** 2026-08-02

### What prompted this

Right after building `--repeat`, flagged a concern before running the
saturation sweep: `--repeat` solves *volume* (enough samples for a real
p95/p99) but not *diversity* — cycling the same 5 vLLM-concept prompts 20
times at concurrency 25 means every "distinct" data point is really a noisy
repeat measurement of 5 underlying prompts, not a sample of varied traffic.
Realistic saturation testing wants both.

Recalled two earlier, separate projects built for an unrelated Coursera
course before this master plan existed: `llm-benchmark-module1` (non-
streaming baseline harness, `gpt-4o-mini` vs `gpt-5.4-nano`) and
`llm-reliability-module2` (retry/backoff middleware). Checked both for
reusable prompt material.

### What was found

- `llm-benchmark-module1/datasets/support_chat_eval_v1.jsonl` — **15 real,
  distinct customer-support prompts** (order status, billing, auth,
  cancellation, technical issues), already used in a real benchmark run with
  its own p50/p95/p99 results. Same `{id, prompt}` JSONL shape as this
  project's dataset (ids were integers there, restring as needed).
  Genuinely useful: more realistic short-form traffic than 5 vLLM-jargon
  prompts, and it already comes with a documented "regression canary"
  (`s15` / "My payment keeps failing" — was the single slowest prompt in
  that project's own results due to long troubleshooting output).
- `llm-reliability-module2` — only 5 prompts (not in the repo tree readably
  from this session, inferred from its own README's 5-clean/5-retry run
  description). No additional volume to offer here; relevant instead to
  later Phase 1/8 work on retry/backoff and admission control, not to
  Task 4's dataset gap.

### Decision

Merged: kept the original 5 vLLM-concept prompts (ids `1`-`5`) and appended
all 15 support-chat prompts with a `s`-prefixed id scheme (`s1`-`s15`) to
avoid any collision and keep source traceable at a glance. Dataset is now
20 prompts total. `--repeat` still applies on top for the higher
concurrency levels, but now cycles 20 varied prompts instead of 5, so far
fewer repeat passes are needed to reach the same total request volume
(recomputed the run plan: concurrency 25 now needs repeat=5, not repeat=20,
for the same ~100-request target).

Considered but rejected: using the 15-prompt set alone (would have dropped
the 5 vLLM-specific prompts that are actually on-topic for this project) and
leaving `--repeat` as the only fix (solves volume, not diversity — the
weaker approach now that a better dataset is one merge away).

### Verification

Loaded the merged 20-line `prompts.jsonl` through the harness's own
`load_dataset()` + `expand_dataset()` functions directly (not by reading the
file by eye): confirmed exactly 20 rows, all 20 ids unique, and
`expand_dataset(rows, 5)` on the 20-prompt base produces exactly 100 rows
with 100 unique ids. Did not yet run this against the live server in this
session.

## Meta-lesson for the writeup

The `--repeat` flag from the previous entry wasn't wrong, but it was an
incomplete fix — it's easy to solve "not enough data points" with a
mechanical repeat and stop there, without asking whether the *content* of
those data points represents anything realistic. Worth remembering for the
eventual interview narrative: a saturation curve built on 5 prompts
repeated 20x and one built on 20 varied prompts repeated 5x will produce
different-looking p95/p99 numbers even at identical concurrency and total
request count, because request-to-request variance (not just queueing
variance) is part of what a real production p99 has to absorb.

## Entry: The `s10` "36,948ms TTFT" bug — queue wait leaking into TTFT

**Date:** 2026-08-02

### What happened

First real concurrency=1 run against the merged 20-prompt dataset came back
with 20/20 ok, 0 errors, but one wildly outlying value: `s10` at
36,948ms TTFT against a background of every other request in the 260–360ms
range. Initial hypothesis (by analogy to the Task 3 log entry) was the same
MLX Metal-shader JIT-compile cold start already documented there.

### Why that hypothesis didn't hold up

Checked the vLLM server's own stdout log for the time window `s10` ran in,
rather than accepting the JIT-cold-start story on pattern-match alone. The
server log showed no stalled or slow-running request anywhere in that
window — every request showed `Running: 1 reqs, Waiting: 0 reqs` with
steady ~14-16 tokens/s generation throughput, evenly paced roughly 10
seconds apart, for the entire run. If `s10` had genuinely taken 37 seconds
of server-side processing, the engine's own periodic log line would have
shown it stuck as `Running: 1 reqs` across several consecutive log ticks
with near-zero throughput. It didn't. That ruled out "real server latency"
and "JIT cold start" (a real JIT delay would show up as slow generation
throughput on that specific request, not as an invisible gap the engine log
doesn't reflect at all).

### The actual bug

`run_bench.py` measured `ttft_ms` starting from `time.perf_counter()`
inside `run_one_request` — after semaphore acquisition, in theory — but had
no way to separately account for or report time spent waiting for that
semaphore. Added instrumentation rather than assuming the theory: a new
`queue_wait_ms` field (task creation → semaphore acquired) and
`connect_wait_ms` field (semaphore acquired → HTTP stream open), both
wired through `bound_run`/`run_one_request` and surfaced in the CSV output
and live console status line.

### Confirmed with the instrumented re-run

Re-ran concurrency=1 with `--repeat 2` (40 requests) after the fix.
Results: `ttft_ms` values were all realistic (174ms-636ms) across all 40
requests — no outliers, matching the server log's steady per-request pace
exactly. But `queue_wait_ms` climbed steadily across the run: 16,518ms on
request 2, up to 550,173ms on request 40. Checked whether this growth was
linear (expected sequential-queueing behavior at concurrency=1) or
something pathological: computed the step-to-step deltas programmatically
— mean increase per step was 14,044ms, consistent within a narrow range
(1,790ms-28,899ms, no runaway growth), and the predicted final wait based
on "38 prior requests x ~14s each" (532,000ms) landed within 4% of the
actual measured value (550,173ms).

### Root cause, now fully explained rather than patched around

At `--concurrency 1`, all N tasks in a `--repeat`-expanded run are created
simultaneously (the task list is built in one loop before `as_completed`
begins), but the semaphore only admits one at a time. Request K in
submission order has to wait for all K-1 requests ahead of it to each
complete their full ~10-14s turn before its own clock even starts. This is
correct, expected behavior for a concurrency=1 harness — not a bug in the
scheduling itself. The bug was purely that this wait time had no dedicated
field and was at risk of being misread as TTFT if it ever leaked into that
measurement path. `s10` in the original run wasn't special; it happened to
land 9th in submission order, and 9 requests x ~14s explains almost exactly
the 36,948ms it originally showed.

### Side effect worth noting for the rest of Task 4

`--repeat` at `--concurrency 1` makes wall-clock run time scale linearly
with total request count, by design (strictly sequential). This 40-request
run took ~550 seconds (~9 minutes) end to end. Concurrency 5/10/25/50 runs
will be faster per-request due to real parallelism, but worth budgeting
time for the full five-level sweep with this in mind.

## Meta-lesson for the writeup

Same verification discipline as every entry in this log so far, but this
time it went two levels deep: first, didn't accept the plausible "JIT cold
start" story without checking the server's own log against it (that log
check is what disproved the first hypothesis). Second, after fixing the
harness, didn't accept "looks fixed" as good enough either — computed the
step-deltas and the linear-growth prediction explicitly rather than eyeing
a column of ascending numbers and assuming they looked reasonable. A
metric that increases steadily can still be a bug (e.g. a leak); the only
way to tell it apart from expected linear queueing is to actually do the
arithmetic.

## Entry: concurrency=5 — a second, genuinely different ~22s cluster (real this time)

**Date:** 2026-08-02

### What happened

First `--concurrency 5` run (40 requests, `--repeat 2`) came back 40/40 ok,
0 errors, but five requests (`1`, `s4-r2`, `s7`, `4`, `s8`) all showed
`ttft_ms` around 22,200-22,300ms against a background of 200-650ms for
everything else. Given the `s10` incident from the previous entry, the
instinct was to suspect another harness measurement bug rather than real
server latency.

### Why this one is different from `s10`

Checked the same two things as last time, but the instrumented fields now
made the check immediate instead of requiring a server-log cross-reference
first. All five outlier requests showed `queue_wait_ms` near zero
(0.1-4.2ms — instant semaphore acquisition, correct at concurrency=5 for
the first 5 requests submitted) and `connect_wait_ms` around 200ms (healthy
TCP/HTTP handshake, no networking issue). The entire ~22 second gap sat
between "HTTP stream opened" and "first SSE token arrived" — squarely in
the territory of genuine server-side prefill/scheduling time, not a
harness artifact.

Cross-checked against the server's own log for this window anyway, rather
than trusting the instrumented breakdown alone (same principle as always):
`Running: 5 reqs, Waiting: 0 reqs` from the very first log line, held
steady the entire run, generation throughput settled at a healthy ~29
tokens/s (roughly double the ~15 tokens/s single-sequence throughput
measured at concurrency=1 — real evidence continuous batching is doing its
job and scaling total throughput with concurrency). No stalls, no request
ever shown waiting, no errors.

### Conclusion

This is a real, one-time cost, not a bug: the first 5 requests submitted at
`--concurrency 5` are the first batch this server process has ever run
through its *batched* execution path (as opposed to the single-sequence
path already warmed during the concurrency=1 run). Consistent with the
MLX/Metal JIT-compile behavior already documented in the Task 2 and Task 3
entries above — the compile cost is per distinct execution shape, not
purely per-process — just showing up here for the first time on a batch of
5 concurrent sequences instead of 1. Every request after the first 5 in
this run rode the now-warm batched path at normal (200-650ms) TTFT.

### Implication for the saturation curve dataset

`runs/concurrency5.csv`'s first 5 rows carry this one-time batched-path
warmup cost. `concurrency10.csv`, `concurrency25.csv`, and `concurrency50.csv`
should not repeat it, since the server stays warm across the whole sweep as
long as it isn't restarted between levels — those runs will be exercising
already-compiled execution shapes at higher batch sizes, though a *new*
one-time cost is plausible if 25 or 50 concurrent sequences trigger yet
another distinct code path the server hasn't seen before. Worth watching
for, not assuming away.

## Meta-lesson for the writeup

The `queue_wait_ms` / `connect_wait_ms` split added after the `s10`
incident paid for itself immediately on the very next run: what would have
been another multi-message investigation (recall the server log, eyeball
timestamps, cross-reference by hand) became a five-second read of two
fields that immediately pointed at "this is downstream of the connection,
not the harness." The fix from one bug became the diagnostic tool for the
next question — instrumentation added for one incident is worth keeping
even after that incident is closed, because the next anomaly is rarely
identical but often adjacent.

## Entry: concurrency=10 — same batched-path-warmup pattern, predicted correctly in advance

**Date:** 2026-08-03

### What happened

`--concurrency 10` run (60 requests, `--repeat 3`) came back 60/60 ok, 0
errors. Ten requests (`1`, `s5-r2`, `5-r3`, `s4-r2`, `s15-r3`, `s7`, `4`,
`4-r3`, `s14-r3`, `s8`) showed `ttft_ms` around 30,952-31,084ms with no
`queue_wait_ms` printed (near-instant semaphore acquisition). Everything
else in the run settled into a normal 600-925ms range.

### Why this needed less investigation than the previous two incidents

This is the same shape predicted by the concurrency=5 entry above: a
one-time batched-execution-path warmup cost, this time triggered by the
first 10 concurrent requests instead of 5. Checked the count and tightness
of the cluster programmatically before accepting the pattern-match: exactly
10 requests in the elevated group (matching `--concurrency 10` precisely,
not a coincidental nearby number), with a spread of only 132ms between the
fastest and slowest of the ten (30,952ms-31,084ms) — a tight cluster like
this means all ten are measuring the same underlying warmup event finishing
at nearly the same instant, not ten independently slow requests that happen
to average out close together.

Given the mechanism was already confirmed once via direct server-log
cross-reference (the concurrency=5 entry above), and this run's data
matched that mechanism's predicted shape exactly on both count and cluster
tightness, decided not to pull the server log a second time to re-confirm
what the previous entry already established directly. Logged as a
deliberate choice rather than a skipped step: two data points that fit a
mechanistic model precisely is different from two coincidences, and a
third confirmation of the identical thing has diminishing diagnostic value
compared to spending that time on the actual saturation-curve runs still
ahead (concurrency 25, 50).

### Running tally of the pattern

| Concurrency | Requests in warmup cluster | Cluster spread |
|---|---|---|
| 5  | 5  | untracked precisely, visually tight |
| 10 | 10 | 132ms |
| 25 | 25 | 12ms |

Expectation for concurrency=25 and 50: the same mechanism should produce a
cluster equal to the concurrency level, assuming each new concurrency level
exercises a genuinely new batched-execution shape the MLX/Metal path hasn't
compiled yet. Worth checking this table still holds at 25 and 50 — if the
cluster stops matching concurrency count exactly, that would be the signal
to go back to the server log.

## Meta-lesson for the writeup

Deciding *not* to re-verify is itself a decision that should be made
explicitly and logged, not silently skipped. The verification discipline
this log has tracked from the start isn't "always check everything" — it's
"don't assume without checking the first time a pattern appears." Once a
mechanism is directly confirmed, recognizing when further identical checks
have low marginal value is the other half of good judgment, not a lapse in
it.

## Entry: concurrency=25 — pattern held exactly, plus the first real saturation signal

**Date:** 2026-08-03

### Cluster check

`--concurrency 25` run (100 requests, `--repeat 5`) came back 100/100 ok, 0
errors. Counted the elevated-TTFT cluster programmatically rather than
eyeballing the pasted output (a visual scan of 100 lines is exactly the
kind of place a miscount happens unnoticed): exactly 25 requests at
~73,280-73,291ms TTFT, spread of only 12ms end to end — tighter than the
concurrency=10 cluster's 132ms, and far tighter than would be expected from
25 independently-slow requests happening to land close together by chance.
Table above updated. Pattern now holds precisely across three concurrency
levels (5, 10, 25), each cluster count matching concurrency exactly and
each cluster getting relatively tighter as concurrency increases —
consistent with more requests piling up behind the same single blocking
compile/warmup event and all being released together once it resolves.

### The new thing: a real saturation-linked TTFT tail, not a warmup artifact

Distinct from the cluster, a handful of *later* requests in the run — `s5`
(1,503ms), `s6` (1,710ms), `s8-r2` (1,298ms), `s3`/`s15-r5` (~1,030ms) —
show TTFT noticeably above the ~600-950ms baseline the rest of the run
settled into. These are not warmup artifacts: each has a real, non-zero
`queue_wait_ms` consistent with its position in the run, and none of them
are part of the initial 25-request cluster. This is the first run in the
sweep where steady-state TTFT (post-warmup) shows meaningful spread instead
of sitting in a tight band — plausibly the actual signal Task 4 exists to
find: early evidence of sustained 25-way concurrency starting to produce
tail latency, as opposed to every prior anomaly in this log having a fully
mechanical explanation (harness bug, one-time compile cost). Not yet
confirmed as "real saturation" rather than run-to-run noise — one run
isn't enough to tell from — but flagged explicitly rather than folded
silently into the same "expected artifact" bucket as the cluster above,
since conflating a genuine saturation signal with a warmup artifact would
undermine the whole point of this task.

## Meta-lesson for the writeup

Not every anomaly in a benchmark run is a bug to explain away — Task 4
exists specifically to find the concurrency level where latency starts
degrading for real reasons (KV cache pressure, memory bandwidth, scheduler
contention), and conflating that signal with harness artifacts would be as
much of a mistake as conflating a harness bug with real signal in the
other direction. The discipline that mattered here wasn't investigating
further, it was correctly sorting this run's anomalies into two different
buckets instead of applying the same explanation to both.

## Entry: concurrency=50 — the warmup pattern held, but steady-state latency clearly shifted

**Date:** 2026-08-03

### Cluster check first

`--concurrency 50` run (100 requests, `--repeat 5`) came back 100/100 ok, 0
errors. Counted the elevated-TTFT cluster the same way as the previous two
entries: exactly 50 requests at ~71,889-72,059ms TTFT, spread of 170ms.
Cluster count still matches concurrency exactly — the batched-path-warmup
pattern held a third time (5→5, 10→10, 25→25, 50→50). Table:

| Concurrency | Requests in warmup cluster | Cluster spread |
|---|---|---|
| 5  | 5  | untracked precisely, visually tight |
| 10 | 10 | 132ms |
| 25 | 25 | 12ms |
| 50 | 50 | 170ms |

(Spread isn't monotonically shrinking anymore — 50's 170ms is looser than
25's 12ms. Plausible explanation: at 50 concurrent requests the M4's
unified memory and thread scheduling are under real contention even during
the warmup event itself, so the release-together timing is less precise
than at 25. Not investigated further; noted as a data point, not a new
anomaly requiring its own root-cause dig.)

### The real finding: no return to baseline after the cluster

At every prior concurrency level (5, 10, 25), the ~50-90 requests after the
warmup cluster settled into a tight, low, stable band matching the
server's steady-state single/low-concurrency performance (roughly
600-950ms at concurrency=25). At concurrency=50, that return to baseline
did not happen. Checked this programmatically rather than trusting a
visual read of 100 lines: the 50 post-cluster requests range from
1,002-2,171ms with a median of 1,348ms — roughly 1.5-2x concurrency=25's
steady-state median, and critically, *no overlap at all* between the two
distributions (concurrency=25's slowest steady-state request was still
under concurrency=50's fastest post-cluster request).

This is different in kind from the concurrency=25 entry's tail-latency
observation (a handful of late requests running a bit slower than the
pack). Here, the entire remaining half of the run — not a handful of
outliers — is running in a visibly different, slower regime. That's the
saturation curve doing exactly what Task 4 was designed to find: the point
where GPU/memory-bandwidth/scheduler contention starts affecting the
*typical* request, not just the tail.

### What this means for the eventual chart and writeup

The concurrency 1/5/10/25/50 sweep now has real shape: TTFT stays flat and
low through 25, then the steady-state (post-warmup) median roughly doubles
by 50. That's consistent with this being close to or past the model's
practical concurrency ceiling on this hardware (M4 Mac mini, 4-bit
Mistral-7B via MLX/Metal) — exactly the kind of inflection point the
Phase 1 interview-angle language in the master plan describes ("identified
the exact inflection point where p99 goes non-linear"). Whether 50 is
past the ceiling or the ceiling sits somewhere between 25 and 50 isn't
fully resolved by a five-point curve — worth keeping in mind if a follow-up
run at an intermediate concurrency (e.g. 35-40) would sharpen the eventual
chart, though that's beyond what Task 4 strictly requires.

## Meta-lesson for the writeup

The distinction that mattered in this entry was between "this run's
population still splits cleanly into an artifact bucket and a real-signal
bucket" (true at concurrency=25) versus "the real-signal bucket has grown
to be the majority of the run" (true here). Both conclusions came from the
same discipline — count things programmatically, compare distributions
numerically, don't eyeball a wall of TTFT values — but the size of what
that discipline revealed changed the entry from "one more data point
confirming a known pattern" to "the actual finding this whole task exists
to produce."

## Entry: Task 4 closed out — analyze.py warmup exclusion, aggregate throughput fix, final results

**Date:** 2026-08-03

### What was left after the sweep

All five concurrency levels (1, 5, 10, 25, 50) had run clean — 0 errors
across 340 total requests. Remaining work: `analyze.py` computes p50/p95/p99
straight from raw `ttft_ms`, which still includes the warmup clusters
documented in the four entries above. Left as-is, concurrency=50's reported
p50 would land around 72 seconds (half that run's 100 requests sat in the
warmup cluster) instead of the real ~1.3 second steady-state figure —
unusable for the actual deliverable.

### Fix: statistical warmup detection, not a hardcoded threshold

Added `split_warmup_cluster()` to `analyze.py`. Deliberately avoided a fixed
time threshold (e.g. "exclude anything over 70,000ms") since that would
silently stop working the moment the model, hardware, or prompt set
changes. Detection instead requires two conditions together: near-zero
`queue_wait_ms` (this request got a semaphore slot instantly, meaning it
was among the first N submitted) AND `ttft_ms` more than 5x the file's own
steady-state median (computed from the *other* requests, so the cluster
can't skew its own baseline). Exclusion count is printed to the console
every run, never silent.

### Caught a bug in my own verification before it reached real data

Wrote a synthetic test first rather than running the new code straight
against the real CSVs. First version of the synthetic test used
`queue_wait_ms` values up to 170ms for simulated warmup rows (borrowed from
concurrency=50's real *TTFT* spread, mixed up with what queue_wait should
look like) against a 100ms detection threshold — only caught 33 of 50
simulated warmup rows. Root cause: the harness's own console output only
ever prints `queue_wait` when it exceeds 50ms (a threshold set when
`--repeat` was first built), and none of the real concurrency=50 run's 50
cluster rows had a printed queue_wait annotation — meaning real cluster
rows have `queue_wait_ms` well under 50ms, not up to 170ms. Fixed the test
to match that reality (0-5ms) rather than loosening the detection threshold
to paper over unrealistic test data. Re-ran: clean 50/50 split, confirmed
also against a concurrency=1 single-row edge case (no false positive when
there's nothing to compare against) and a pre-fix CSV missing the
`queue_wait_ms` column entirely (passes through as a no-op, doesn't crash).

### Ran against the real five CSVs

Exclusion counts matched exactly what had been manually verified by hand in
each of the four entries above: 1 excluded at concurrency=1 (even a single
request pays a first-ever-request compile cost), then 5, 10, 25, 50 —
precisely the cluster sizes already confirmed against the server's own log.

### Second bug caught before it reached the writeup: per-request vs aggregate throughput

The original `avg_tokens_per_sec` column is the *mean of each request's own*
tokens/sec — this naturally declines as concurrency rises (more requests
sharing one GPU means each one individually takes longer), even when the
server is doing more total work. Raw output showed 14.7 -> 6.1 -> 4.8 ->
4.5 -> 3.0 tokens/sec across the sweep, which reads like "throughput got 5x
worse," the opposite of what continuous batching is supposed to deliver.
Flagged this rather than let a misleading number carry into the eventual
portfolio writeup. Added `aggregate_tokens_per_sec` (total output tokens
across all steady-state requests / wall-clock span of the run) alongside
the existing per-request average, and updated the chart to plot both
explicitly labeled so neither can be mistaken for the other.

### Chart also needed a second fix: mismatched axis scales hid the actual deliverable

First chart version plotted TTFT p50/p95 and `total_p99_ms` on one shared
axis. `total_p99_ms` reaches ~121,000ms (it includes queueing time, which
scales up steeply with concurrency) while TTFT tops out around 2,200ms —
the TTFT lines, the actual metric Task 4 asks for, were visually flattened
to an invisible line at the bottom of the chart. Split into three panels:
TTFT alone, total request time alone, and throughput (aggregate + per-
request). Verified by rendering and inspecting the image directly rather
than assuming the code change fixed the visual problem.

### Final results

| Concurrency | TTFT p50 | TTFT p95 | TTFT p99 | Aggregate tok/s |
|---|---|---|---|---|
| 1  | 273ms   | 364ms   | 545ms   | 282.4 |
| 5  | 482ms   | 588ms   | 638ms   | 93.6  |
| 10 | 728ms   | 785ms   | 864ms   | 100.3 |
| 25 | 760ms   | 1,031ms | 1,557ms | 147.5 |
| 50 | 1,348ms | 2,147ms | 2,170ms | 80.2  |

TTFT climbs gradually through concurrency 25, then p95/p99 both bend
sharply upward at 50 — a real non-linear tail-latency inflection point, not
an artifact (this is the steady-state number, warmup cluster already
excluded). Aggregate throughput peaks at concurrency=25 (147.5 tok/s) and
drops at 50 (80.2 tok/s, below even the concurrency=5 level) — a second,
independent signal pointing to the same conclusion: this server, on this
hardware, with this model, has a practical concurrency ceiling somewhere
between 25 and 50, not at 50 itself. Task 4's stated deliverable
("saturation curve: ramp 1->5->10->25->50, record p50/p95/p99 + tokens/sec
at each step") is complete, with the added finding of exactly where the
curve bends.

## Meta-lesson for the writeup

This closing entry has the same shape as most of the entries before it,
compressed: build something, test it against synthetic data before trusting
it on real data, catch a bug in the test itself before it could produce a
wrong answer, and verify the final visual output by actually looking at it
rather than assuming a code fix produced the intended picture. The specific
bugs differed each time (semaphore/TTFT conflation, an unrealistic
synthetic test range, a misleading throughput average, a chart axis scale
mismatch) but the pattern catching all of them was identical: don't accept
a plausible-looking result without checking it against something concrete
-- a server log, a hand-computed expected value, or the rendered image
itself.
