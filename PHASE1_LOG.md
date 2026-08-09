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

## Entry: Task 6 — Grafana dashboard, and the DCGM gap resolved with a purpose-built exporter

**Date:** 2026-08-03

### What Task 6 asked for

Build a Grafana dashboard with 6 panels: TTFT heatmap, tokens/sec vs
concurrency, p99 trend, GPU SM utilization, HBM bandwidth, KV cache fill
level. Two of the six (GPU SM utilization, HBM bandwidth) are standard DCGM
panels in the master plan's reference design — DCGM is NVIDIA-only, already
flagged as a gap in `vllm-benchmark/README.md`'s "Known limitation" section
from earlier work on this project.

### Decision: build a real exporter, not a placeholder

Considered three options for the two DCGM-dependent panels: (1) build a
`powermetrics`-based exporter and get real (if differently-sourced) data
into those panels, (2) leave them as documented "no data" placeholders, (3)
swap in two unrelated-but-available metrics instead. Chose (1) — a real
substitute is a stronger interview story ("I adapted DCGM-equivalent
observability to a platform NVIDIA's tooling doesn't support") than either
alternative, and macOS's `powermetrics` genuinely does expose GPU
utilization-adjacent data, just not in DCGM's exact shape.

### Verifying `powermetrics`'s actual output format before writing a parser

Rather than write a regex against a guessed or documentation-described
format, found and fetched a real captured `powermetrics --samplers
gpu_power` text-format sample (from a small open-source Go wrapper project,
`BinSquare/powermetrics-go`, which ships an actual sample log used by its
own test suite). Confirmed directly from that sample:

- `GPU HW active frequency: <N> MHz` — real field, present
- `GPU HW active residency:  <N>.<N>% (...)` — real field, present, genuine
  analog to DCGM's SM-utilization percentage
- `GPU idle residency:  <N>.<N>%` — real field, present
- `GPU Power: <N> mW` — real field, present, but appears **twice** in one
  sample block (once in a top-level CPU/GPU/ANE power summary, once again
  inside the `**** GPU usage ****` section) — both values were numerically
  identical in the sample examined, but nothing guarantees that holds
  across `powermetrics` versions, so the regex was deliberately anchored to
  the second (GPU-usage-section) occurrence rather than relying on the
  first match being correct by luck. Verified this mattered with a stress
  test: manually forced the two occurrences to diverge (999 vs 28) and
  confirmed the parser still picked the correct one.
- **No HBM/memory-bandwidth field exists anywhere in the sample.** Confirmed
  by inspecting the real output rather than assuming the gap — Apple
  Silicon's unified memory architecture has no discrete VRAM bus for a
  bandwidth-utilization percentage to describe. Decision: rather than
  fabricate a number or leave the panel silently empty, the HBM bandwidth
  panel shows `GPU HW active frequency` instead, with the substitution
  stated explicitly in the panel description, the exporter's own metric
  HELP text, and the README — never silently swapped in.

### What got built

New `observability/` folder in this repo:
- `docker-compose.yml` + `prometheus.yml` — Prometheus + Grafana in Docker,
  scraping vLLM's existing `/metrics` (native process, unchanged from
  `scripts/serve.sh`) and a new exporter (also native, not Dockerized,
  since `powermetrics` needs direct host access and `sudo` that a
  container on macOS's Docker Desktop VM cannot reach)
- `powermetrics_exporter.py` — samples `powermetrics` continuously, parses
  the four fields above, serves them in Prometheus text format on `:9400`
- `grafana/provisioning/` — datasource + dashboard auto-provisioning, so
  Grafana has the 6-panel dashboard already loaded on first startup rather
  than requiring manual UI setup
- `README.md` — architecture, quick start, and an explicit
  verified/corroborated/unverified breakdown (see below)

### What could and could not be verified from this session

Working from a sandboxed Linux environment with no macOS, no Docker, no
`sudo`, and no live vLLM server reachable, verification was necessarily
partial — documented that honestly rather than presenting everything as
equally confirmed:

- **Directly verified**: the exporter's parsing logic, against the real
  captured sample and a divergent-value stress test; all config files
  parse as valid YAML/JSON; the dashboard JSON has all 6 panels.
- **Corroborated, not directly tested against this project's server**:
  `vllm:time_to_first_token_seconds` and `vllm:generation_tokens_total` —
  confirmed as real vLLM metric names via vLLM's official metrics
  documentation (web search), but not curl-verified against this project's
  actual running server the way `num_requests_running`,
  `num_requests_waiting`, and `kv_cache_usage_perc` were back in the Task 3
  entry. Flagged explicitly in the README with the exact curl command to
  run and which panels to fix if the names differ.
- **Cannot be verified from this environment at all**: the actual `sudo
  powermetrics` subprocess behavior, Docker Desktop's `host.docker.internal`
  resolution on this specific machine, and whether the heatmap panel
  renders meaningfully with real traffic. All flagged in the README under
  "What's verified vs. what needs your confirmation" as next steps for a
  live run, not silently assumed to work.

## Meta-lesson for the writeup

Two different verification postures were needed in the same task, and
conflating them would have been dishonest either direction: claiming full
confidence in things that were only corroborated by documentation (not
tested against this specific server) would overstate certainty, while
treating documentation-corroborated facts with the same suspicion as a
pure guess would understate it. The useful move was sorting every claim
into an explicit tier — directly verified, corroborated but not tested
here, or simply untestable from this environment — and saying so plainly,
rather than presenting a uniform "done" that papers over which parts still
need a human with the actual hardware to confirm.

## Entry: Task 6 live verification — the "No data" bug, a Grafana crash loop, and a working dashboard

**Date:** 2026-08-03

### The first failure: every panel showed "No data"

`docker compose up -d` came up clean (Prometheus and Grafana both started,
all three scrape targets showed `UP` on Prometheus's own targets page,
`curl localhost:9400/metrics` and `curl localhost:8000/metrics` both
returned real, correct data directly). Despite that, every one of the 6
dashboard panels showed "No data" in Grafana.

Root cause: every panel in `vllm-saturation-dashboard.json` referenced its
datasource as `{"type": "prometheus", "uid": "${DS_PROMETHEUS}"}`. That
`${DS_PROMETHEUS}` syntax is a Grafana template variable meant to be
resolved during the UI's dashboard-import flow — it only gets substituted
with a real datasource uid when a dashboard is uploaded through Grafana's
"Import" screen. This dashboard was file-provisioned instead (dropped
directly into `grafana/provisioning/dashboards/`), which never runs that
substitution step, so every panel was left pointing at a literal,
unresolved string that matched no real datasource. This was a real design
mistake in the original dashboard JSON — written as if it would be
UI-imported, then actually deployed via file provisioning, without checking
that both paths handle datasource references the same way (they do not).

Confusingly, manually opening a panel's Edit view and touching its
Data source dropdown made that specific panel start working immediately —
Grafana's query editor re-resolves the datasource live in that view. This
made the bug look intermittent/panel-specific at first, until recognizing
that only the panels someone had manually edited were fixed; the untouched
ones were still reading the broken reference from the dashboard's stored
state.

**Fix**: added an explicit `uid: prometheus-vllm` to the datasource
provisioning YAML (previously no uid was set, so Grafana auto-generated one
unpredictably), then updated all 6 panels in the dashboard JSON to
reference that fixed uid directly instead of the broken template variable.

### The second failure: Grafana crash-looping after the fix

Restarting with `docker compose restart grafana` didn't apply the fix, and
a full `docker compose down && up -d` made things worse: Grafana entered a
crash loop (`docker compose ps` showed `Restarting (1)` repeatedly, port
3000 unreachable). `docker compose logs grafana` showed the real error:
`"Failed to provision data sources" error="Datasource provisioning error:
data source not found"`, escalating to a hard startup failure across every
dependent module.

Root cause: Grafana persists dashboard/datasource state into its own
internal database, stored in the `grafana-data` Docker volume — it does not
purely re-read the provisioning files fresh on every restart. The first
time this stack ever started, Grafana loaded the dashboard with its broken
`${DS_PROMETHEUS}` reference and wrote *something* into its internal
database reflecting that broken state. After adding the new named
`prometheus-vllm` datasource uid, Grafana's persisted internal state and
the newly-provisioned config disagreed in a way it could not reconcile on
startup, and it failed hard rather than silently.

**Fix**: `docker compose down -v` (removing both named volumes, including
the stale `grafana-data`) followed by `docker compose up -d`. This gave
Grafana a genuinely clean slate with no conflicting persisted state to
reconcile against — it started cleanly on the first attempt afterward.
Losing Prometheus's ~10 minutes of scrape history in the same volume wipe
was an acceptable, deliberate tradeoff — there was nothing in it worth
preserving from this early testing phase.

### Live verification: 10-concurrent-user benchmark run against the dashboard

Ran a 10-user benchmark against the now-working dashboard and observed all
6 panels update with real, mutually-consistent data:

- Tokens/sec climbed from 0 to ~8 as `requests running (concurrency)`
  ramped to 10, tracked on the dual-axis Tokens/sec vs Concurrency panel
- GPU active residency rose from an idle dip to a sustained ~100% right as
  load hit, and GPU active frequency spiked from ~800 MHz to ~1,600 MHz in
  the same window — both GPU panels agreeing with each other and with the
  throughput panel that the GPU genuinely got busy, not just showing noise
- TTFT p99 trended downward across the run (~1.3 min down toward the end),
  consistent with the warmup-cluster cost documented extensively in the
  Task 4 entries being amortized as the batched execution path stayed warm
  through the run
- KV cache fill level stayed flat near 0% the entire run — plausible for
  only 10 short prompts against this model's context length, but the one
  panel that didn't show clear movement. Flagged as worth re-checking with
  a heavier or longer-context run later in the plan (a natural fit for
  Phase 6's TensorRT-LLM/quantization work, which will need this same
  dashboard) to confirm the panel is correctly wired rather than just
  quiet by coincidence.

Task 6's stated deliverable (Grafana dashboard, 6 panels: TTFT heatmap,
tokens/sec vs concurrency, p99 trend, GPU SM utilization, HBM bandwidth,
KV cache fill level) is complete and confirmed live, with the HBM bandwidth
panel's documented substitution (GPU active frequency, since Apple Silicon
exposes no bandwidth metric) holding up under real traffic the same way the
other five panels did.

## Meta-lesson for the writeup

Two failure modes stacked here, and each one only became visible by trying
the real thing rather than trusting the previous fix looked complete on
paper. The datasource-uid bug could not have been caught from a sandboxed
environment with no live Grafana to click through — this is exactly the
kind of gap the earlier entry's "what's verified vs. what needs your
confirmation" framing existed to flag honestly rather than paper over. The
crash-loop that followed the first fix is a good instance of a fix
introducing a *new*, different failure rather than resolving the original
one cleanly — worth remembering that "the config is now correct" and "the
running system reflects that correct config" are not the same claim,
especially for any tool (Grafana here) that persists state outside the
files you're editing.

## Entry: Task 7 — reframed to 4-bit vs 8-bit MLX quantization, and the idle-recovery investigation

**Date:** 2026-08-04

### Reframing the task

The master plan's Task 7 asks for a bf16-vs-fp8 comparison. Neither format
exists on this stack: vllm-metal (Apple Silicon via MLX) supports neither.
Reframed, with sign-off, to 4-bit vs 8-bit MLX quantization instead — the
comparison this hardware can actually run, using the already-established
4-bit Mistral-7B (Task 4/6) against a newly-added 8-bit build of the same
model.

### Two environment bugs fixed before any data could be collected

- `scripts/serve.sh`'s `VLLM_BIN` default pointed at a path that never
  existed (`.venv-metal`-style guess, not the real install location). Every
  prior session's use of the script had silently ridden on a server someone
  had already started by hand via `source ~/.venv-vllm-metal/bin/activate` +
  manual `vllm serve` — the script's own default had never actually been
  exercised. Fixed to `$HOME/.venv-vllm-metal/bin/vllm`, matching the real
  install path from Task 1's setup doc.
- A missing `xgrammar` dependency surfaced on the first real `serve.sh` run
  post-fix. Resolved via `pip install -r requirements-metal.txt`.

### Data collection and the first anomaly

Collected `8bit-concurrency1.csv/.jsonl` and `8bit-concurrency10.csv/.jsonl`
in `vllm-benchmark/runs/`, to compare against the existing 4-bit
`concurrency1.csv`/`concurrency10.csv` from Task 4. Mid-run, request `s2`
spiked to 36,978ms TTFT at concurrency=1. Checked the server log for the
window rather than pattern-matching to the already-documented cold-start or
batched-path-warmup stories from Task 3/4/6 (see those entries above) — this
one didn't fit either: it happened mid-run, on an already-warm server, at a
concurrency level with no new execution shape to compile. The one thing that
did line up: the server log showed `Running: 0 reqs, Waiting: 0 reqs` for a
stretch immediately before `s2` fired — the engine had gone fully idle. New
hypothesis: an **idle-recovery cost**, distinct from cold start (process
never restarted) and from batched-path warmup (concurrency never changed).

### Building an isolated repro: `idle_recovery_test.py`

Built `vllm-benchmark/scripts/idle_recovery_test.py`: send a warm-up request,
sleep a controlled idle duration, send a test request, report both TTFTs.
First pass at 10s/20s/35s looked reproducible but showed a counterintuitive
shrinking-penalty-with-longer-idle trend. Extended to 7 points
(5/10/15/20/35/45/60s) to check whether that was signal or noise.

### The confound: uninvited traffic invalidated the entire 7-point run

The extended run's own "warm-up" TTFTs were wildly unstable (some
100,000ms+, when a warm-up on an already-running server should be cheap),
with no consistent idle-duration relationship at all. Checked the server log
instead of re-running blind: it showed a steady ~90-second cycle of real
`POST /v1/chat/completions` traffic that `idle_recovery_test.py` itself never
sent. Something else was hitting the server for the whole test window,
invalidating all 7 data points. Killed and restarted the server clean, and
closed Grafana on the assumption it was the source — turned out to be a red
herring (Grafana's panel refresh only polls Prometheus, not vLLM directly);
the real second-order finding on this came later (see below).

### A detour: auth and a red-herring 404

Post-restart, every `idle_recovery_test.py` re-run failed identically with
`404 Not Found` on `/v1/chat/completions`, and `curl .../v1/models`
returned `{"error":"Unauthorized"}`. Resolved by finding the server's actual
`--api-key` from its running process args (`ps aux | grep vllm`):
`local-vllm-key`, which happens to already be the script's own default. With
the header present, both the "404" and the "Unauthorized" vanished in one
shot — the 404 was never a real routing bug, just what an unauthenticated
request off `/v1/chat/completions` looks like before it reaches real
route-dispatch. Not investigated further once auth was confirmed as the
single root cause for both symptoms.

### Getting a genuinely clean read: Grafana/Prometheus were still running

Restarted the server fresh on the 8-bit model (confirming, via the startup
log, identical memory-footprint numbers to the earlier 8-bit run:
`model_memory=7.70GB`, `metal_limit=12.71GB`, `kv_budget=3.28GB` — good
sanity check that nothing about the environment had silently changed).
Before trusting a "clean" sweep, checked for other traffic sources directly
rather than assuming the earlier Grafana-close had actually taken effect —
and it hadn't: `docker ps` showed `vllm-grafana`/`vllm-prometheus` still
`Up 24 hours`. Prometheus's own `/metrics` scrape (5s interval, confirmed
from `observability/prometheus.yml`) was still hitting the server directly,
independent of Grafana. Stopped both containers with `docker compose stop`
and confirmed via a quiet 15-second window of zero new log lines that the
server was genuinely idle before running anything.

### Four sweeps, one real finding and one debunked hypothesis

Ran the clean 7-point (and later, focused 4-point: 15/35/45/60s) sweep four
times total, cross-checking every run against the server's own log
(`Running: 0/Waiting: 0` gaps matching each script's printed idle window,
POST-line counts matching exactly `2 × number of idle durations tested` per
run) before trusting any of the TTFT numbers:

| Idle | Run 1 | Run 2 | Run 3 | Run 4 |
|---|---|---|---|---|
| 15s | — | 2,053ms | 2,122ms | 3,413ms |
| 35s | 28,565ms | 2,065ms | 3,146ms | 3,720ms |
| 45s | 16,350ms | 8,246ms | 5,737ms | 10,891ms |
| 60s | 3,888ms | 4,470ms | 7,494ms | 14,694ms |

Two conclusions, not one:

1. **The idle-recovery penalty itself is real and reproducible.** Every run
   shows a clear jump from a ~2,000ms short-idle baseline into a
   multi-second-to-tens-of-seconds penalty at longer idle, across four
   independent measurements.
2. **The exact idle duration that triggers the worst penalty is not fixed.**
   It moved (35s → 45s → no clear peak → 60s) across the four runs, which
   rules out a simple deterministic threshold tied purely to elapsed idle
   time.

### Testing the leading hypothesis with real telemetry, and disproving it

Leading theory going in: macOS/Metal steps the GPU down into a low-power
state during idle, and stepping back up is what costs the TTFT penalty.
Tested this directly rather than leaving it as speculation, using the
`powermetrics_exporter.py` built in Task 6 (GPU HW active frequency, `:9400`)
plus a simple 1Hz poller (`while true; do curl ...; sleep 1; done`) run
alongside two more replications of the sweep. Lined up each test's predicted
first-token timestamp (`send_time + ttft_ms`) against the frequency
timeline by hand, cross-checked across three independent spike events
(idle=15 test in run 3, idle=45 and idle=60 tests in run 4).

Result: **the hypothesis is wrong.** GPU frequency sits flat at its idle
baseline (~750-800 MHz) for nearly the entire wait, then jumps to its max
observed value (~1,576-1,578 MHz) in about one second — right at the moment
the first token is delivered, not gradually beforehand. For the worst case
observed (a 28.8s TTFT), elevated GPU activity was visible for only the
final ~2-3 seconds; the other ~25+ seconds showed no GPU activity at all
despite the request already being "in flight." The GPU is not slow to wake
up — something upstream of it (scheduling/dispatch layer: Python's asyncio
event loop, macOS process scheduling of an idle process, or MLX's lazy
dispatch) is sitting on the request before any GPU work starts. This is a
real, evidence-backed negative result, not just an unconfirmed guess: it was
tested and ruled out, which is worth more than never having tested it.

### A second, separate finding: session-level drift

Comparing the four sweeps as a whole (table above), most idle durations
climbed steadily run-over-run even where idle duration itself didn't change
— most visibly at 15s (1,567 → 2,053 → 2,122 → 3,413ms) and 60s (3,888 →
4,470 → 7,494 → 14,694ms). This isn't explained by idle duration at all,
since 15s idle should cost roughly the same amount every time. Candidate
explanation: thermal throttling on a Mac mini running repeated GPU bursts
over a ~35-minute session, but this was **not tested** — flagged as an open
question for a future task rather than chased further in this one, per the
"don't over-build beyond what the task asks" scope guardrail.

### Where this leaves Task 7's idle-recovery finding

Closed out deliberately at this point rather than continuing to dig, since
the marginal question left (root-causing the exact scheduling/dispatch stall,
or confirming thermal drift) would be its own investigation, not a
refinement of this one. Summary for the eventual writeup: the 8-bit model
exhibits a real, reproducible idle-recovery TTFT penalty triggered somewhere
above roughly 15-20 seconds of full engine idle, ranging from ~2x to ~37x
baseline TTFT depending on the run; direct GPU-frequency telemetry rules out
GPU power-state stepping as the cause and points instead at the
scheduling/dispatch layer; a separate, unexplained session-level latency
drift was also observed and is flagged as follow-up work.

## Meta-lesson for the writeup

This session is the clearest example yet in this log of the difference
between "a plausible mechanism" and "a mechanism confirmed against evidence."
The GPU-power-stepping theory was reasonable, specific, and testable — and
wrong. Building the actual telemetry to test it (reusing the Task 6
exporter rather than building something new) turned a plausible story into a
falsified one and pointed at a more specific, more useful "somewhere
upstream of the GPU" conclusion. Separately, this session is also the first
time this log has explicitly flagged a *replicated but still-shifting*
pattern (the moving idle-duration peak, and the run-over-run drift) as a
real, reportable finding in its own right, rather than something to keep
digging on until it stabilizes into a single clean threshold — sometimes the
honest, useful answer is "this is real and reproducible, but noisier and
more multi-causal than a single number," not a tidy inflection point.

## Entry: Task 7 closed out — memory footprint, comparison analysis, tradeoffs doc

**Date:** 2026-08-04

### Capturing the missing 4-bit memory numbers

The 8-bit memory breakdown had been captured earlier in the session; 4-bit's
had not. Restarted the server on `mlx-community/Mistral-7B-Instruct-v0.3-4bit`
and read the same startup log line. Result: `model_memory=4.08GB`,
`kv_budget=7.06GB`, `max_tokens_cached=53,856`, vs 8-bit's `model_memory=7.70GB`,
`kv_budget=3.28GB`, `max_tokens_cached=25,024`. Both draw from the same fixed
`usable_metal=11.70GB` pool, so 8-bit's heavier weights come directly out of
the KV cache budget — the practical effect is roughly half the concurrent
long-context capacity, not just "somewhat slower."

### Running the comparison analysis

`analyze.py` against all four CSVs (4-bit/8-bit × concurrency 1/10) in one
call, reusing the warmup-cluster exclusion logic from Task 4 unchanged.
Noticed immediately that 8-bit's concurrency=1 p99 TTFT (23,366ms) was an
outlier disproportionate to its p50/p95 (379ms/1,089ms) — recognized this as
very likely the same idle-recovery phenomenon just spent the whole session
characterizing, since a sequential 20-prompt run at concurrency=1 produces
exactly the kind of inter-request gaps that triggered it earlier. Flagged
this explicitly in the writeup rather than reporting the raw p99 at face
value, which would have misrepresented 8-bit's real tail latency using a
number this session had independently shown reason to distrust.

### Tradeoffs document

Found `LEARNING_AND_PORTFOLIO_STANDARD.md` already defines a required
structure for task deliverables (problem/constraints/architecture/live
proof/internals/engineering judgment/next experiment, plus an explicit
tradeoffs list) — followed it rather than inventing a new format. Wrote
`Task7_Quantization_Tradeoffs.md` covering the memory and latency/throughput
tables above, plus three caveats surfaced by this session's own work rather
than smoothed over: the contaminated 8-bit p99, the concurrency=10 throughput
comparison sitting inside a non-monotonic region already flagged in Task 4,
and the fact that output quality itself was never directly measured in this
task (the "8-bit is higher fidelity" side of the tradeoff rests on general
quantization theory, not a measurement taken here).

### Task 7 status

Core deliverable complete: memory footprint comparison, latency/throughput
comparison, tradeoffs document, and the idle-recovery investigation (with
its own dedicated entry above) are all written up. Follow-up items
explicitly deferred rather than silently dropped: root-causing the
idle-recovery scheduling stall, testing whether 4-bit shows the same
idle-recovery pattern, an output-quality comparison, and the untested
session-level thermal-drift hypothesis.

## Meta-lesson for the writeup

The most useful thing this closing stretch did wasn't the analysis itself —
it was noticing that a number produced by a completely different piece of
work (`analyze.py`, built back in Task 4) needed a caveat from *this*
session's separate investigation before it could be reported honestly. Two
pieces of work in the same session, seemingly unrelated on the surface, that
turn out to explain each other is exactly the kind of connection that's easy
to miss if each task is treated as a sealed box — worth actively checking
whether a new number "smells like" a pattern already confirmed elsewhere
before reporting it standalone.

## Entry: Master plan reconciliation — switching to the plan's own Phase 1 numbering, and Item #2 fully closed

**Date:** 2026-08-05/06

### Master plan finally reviewed directly

User supplied the actual plan PDF (`LLM_PE_Master_Plan_v4.pdf`) for the
first time this session — everything before this point had been
reconstructed from this log's own "Task N" labels, which turn out **not**
to be a 1:1 mapping to the plan's own Phase 1 checklist (17 numbered items,
tier T1 = "do not skip", T2 = "bonus after T1"). Comparing directly against
the plan surfaced two undone T1 items (item 8: burst test, spike 5→50 with
recovery-time measurement; item 9: `cost_per_1k_tokens`/`tokens_per_dollar`
+ Grafana cost panel) and six partially-done T1 items (2, 3, 10, 11, 15,
16), plus three undone T2 items (12, 13, 14: two papers + a prefix-caching
experiment) and the closing retrospective (17).

Decision, per explicit user direction: use the plan's own item numbers
going forward, not the "Task N" log labels used up through the Task 7
entries above. Close every partial item first, then the two net-new T1
items, then T2 items, then the retrospective — full Phase 1 completion
before Phase 2 starts, nothing skipped.

### Item #2 (core vLLM parameters) — closed with real evidence for all four flags

- `--max-num-seqs`: already closed in the original Task 2 entry above
  (`Running:3/Waiting:0` → `Running:1/Waiting:2` evidence).
- `--gpu-memory-utilization`: redone properly via a predict-then-verify
  cycle rather than accepted from incidental past logs. Restarted 4-bit at
  `fraction=0.5`; predicted (correctly, after one self-correction) that
  `usable_metal` shrinks proportionally but `kv_budget` shrinks by a much
  larger *relative* amount, since it's what's left after a fixed
  `model_memory` cost is subtracted from a smaller pool. Verified exactly:
  `usable_metal=6.36GB` (`12.71×0.5`), `kv_budget=1.57GB`
  (`6.36−4.08−0.71`), both matching the arithmetic precisely.
- Follow-up investigation into `overhead`: found and read the actual
  installed vllm-metal source
  (`~/.venv-vllm-metal/.../vllm_metal/v1/cache_policy.py` and
  `model_runner.py`). `overhead` is not a formula or constant — it's
  measured live every server startup by `profile_run()`, which runs one
  real dummy forward pass at `max_num_batched_tokens` size and measures how
  much MLX's buffer cache grows. The function's own docstring confirms this
  replaced "the historical 800 MB placeholder." Explains why `overhead`
  varies slightly run-to-run (`0.56GB` / `0.71GB` / `0.71GB` across three
  separate startups) — real memory telemetry, not a bug or inconsistency.
- `--tensor-parallel-size`: also grounded in real source
  (`vllm_metal/platform.py`). Confirmed `NotImplementedError` is raised for
  `tensor_parallel_size > 1` on Metal, for two stacked reasons stated
  directly in the code's own comments: a single GPU per Mac (hardware), and
  no cross-device collective (`mx.distributed`) wired up for this path yet
  (software — an unbuilt integration, not a fundamental Apple Silicon
  limitation). The same source file shows dense data-parallelism **is**
  supported on Metal, but only across multiple Macs each running a full
  model replica via Ray — a genuinely different topology (replication for
  throughput, not splitting for capacity/per-request latency). Correctly
  reasoned through, unprompted, why DP can't solve a
  doesn't-fit-on-one-GPU problem the way TP/PP can.
- `--max-model-len`: predicted (correctly) that `kv_budget`/
  `max_tokens_cached` are independent of this flag (confirmed absent from
  the `kv_budget` formula in source), and that "Maximum concurrency for N
  tokens per request" scales inversely with it. Verified via restart at
  `max-model-len=512`: concurrency ratio landed exactly on
  `52,704/512=102.94x`; the small deviation in `kv_budget`/
  `max_tokens_cached` from the 2048-baseline was fully explained by the
  *already-understood* overhead measurement noise (`0.15GB` difference in
  overhead ≈ exactly the `72`-block/`1,152`-token shortfall), not by any
  hidden dependency on `max-model-len`.

### Still open in Phase 1 (plan's own numbering)

Items 3, 10, 11, 15, 16 (remaining partials), 8, 9 (undone T1), 12, 13, 14
(T2), 17 (retrospective). Session paused here for a terminal restart
(troubleshooting a `/voice` dictation mic issue after switching headphones
mid-session, unrelated to the technical work) — next session should resume
at item #3 (add `gpu_util_pct` to the benchmark harness's own log schema).

## Meta-lesson for the writeup

Redoing `--gpu-memory-utilization` "properly" instead of accepting it as
already-understood surfaced two genuinely new, source-code-verified facts
(the live `profile_run()` overhead measurement, and the precise
TP-rejection reasoning) that hadn't been captured anywhere in this log
despite the flag having been used correctly for the entire project. Being
willing to re-open something already "working" and demand real evidence
for it — not just past behavior that happened to look right — found real,
previously-undocumented understanding, not just redundant confirmation.

## Entry: Item #3 — `gpu_util_pct` added to the harness's log schema

**Date:** 2026-08-06

### What item #3 asked for

Extend `vllm-benchmark`'s log schema with `gpu_util_pct`, documented since
Task 3 as a known gap: vLLM has no GPU-utilization metric on Apple Silicon
(no DCGM equivalent). The Task 6 Grafana work had already solved this exact
problem once, via `observability/powermetrics_exporter.py` — a sidecar that
wraps `powermetrics` and re-serves `GPU HW active residency` in Prometheus
format on its own port (`:9400`). Item #3's job was purely to make the
*harness* consume that already-existing exporter, not to build new GPU
monitoring from scratch.

### Session start: two access/environment blockers before any code

Picked up this session cold (new chat, `--resume`, only `PHASE1_LOG.md` and
`NOTES_PERSONAL.md` as context). Two blockers before real work could start:

- The `vllm-benchmark` repo (a sibling of this one under `~/Documents`, per
  the Task 3 entry's "new standalone repo" decision) was unreadable from
  this session — every read/list attempt returned `EPERM`, even with the
  Bash tool's sandbox explicitly disabled, which confirmed it was a real
  macOS permission issue (TCC/Full Disk Access) rather than a harness-level
  sandbox restriction. Resolved by the user granting the terminal app Full
  Disk Access in System Settings; access started working immediately after,
  no restart needed.
- Neither the vLLM server (port 8000) nor the powermetrics exporter (port
  9400) was running — expected, since nothing survives a terminal/session
  restart. Verified rather than assumed: `ps aux | grep -i vllm` showed only
  the `grep` command itself matching, confirming no server process was
  alive, before starting anything.

### The actual code change

`run_bench.py`'s `scrape_metrics()` previously had its regex dict
(`METRIC_PATTERNS`) hardcoded as a module-level global — fine for scraping
one fixed endpoint (vLLM's own `/metrics`), but not reusable for a second
endpoint with different metric names. Generalized it to accept a `patterns`
dict as a parameter, added a second pattern set (`GPU_METRIC_PATTERNS`) for
`powermetrics_gpu_active_residency_percent`, and added a second scrape call
in `run_one_request()` against a new `--gpu-metrics-url` (default
`http://localhost:9400/metrics`, skippable via `--no-gpu-metrics`). New
`gpu_util_pct` field added to `RequestResult`, threaded through to the CSV
output columns.

One deliberate detail: the new GPU regex pattern was written with the same
optional-label-group shape as the existing `vllm:` patterns
(`(\{[^}]*\})?\s+([0-9.eE+-]+)`), even though the exporter's actual output
has no labels on that line. This keeps the capture-group index (`group(2)`
= the value) identical across both pattern sets sharing one `scrape_metrics`
function — avoided a class of bug where a mismatched group index silently
returns the wrong field instead of an error.

### Verification, not just "it compiles"

`python3 -m py_compile` + `--help` confirmed the new flags exist and the
file is syntactically valid, but that only proves the code runs, not that
it does the right thing. Ran a real 3-request benchmark
(`--concurrency 1 --max-samples 3`) against the live server with the
exporter also running, and read the actual output CSV: `gpu_util_pct`
populated with real, varying values (96.92, 98.61, 83.36), not blank or a
constant.

### Interpreting the real numbers — one self-correction, one confirmed callback

Two questions came up reading the CSV, both worked through via prediction
first:

- **Why did `gpu_util_pct` jump from the ~6% idle baseline to 83-98%?**
  First instinct was "cold start" (request `3` in this run did pay the
  familiar MLX JIT-compile cost, 26,991ms TTFT). Wrong primary explanation,
  caught by checking the other two rows: requests `1` and `2` were *not*
  cold-start (716ms/915ms TTFT, normal) and still read 98.61%/83.36% GPU
  active residency. Corrected conclusion: active residency tracks whether
  the GPU is doing compute *right now* — generating tokens is
  compute-heavy work regardless of warmup state — not a one-time warmup
  signal. Cold start is a separate, additional cost that happened to land
  on one row in this particular run, not the mechanism behind the metric
  itself.
- **Why did `num_requests_running`/`kv_cache_usage_perc` still read `0.0`
  in the same run, despite the GPU clearly being busy?** This is the exact
  same finding the Task 3 log entry already confirmed on 2026-08-02 (curled
  the idle server directly, verified `0.0` was a real reading, not a
  parsing bug) — recognized as a repeat of an already-confirmed pattern
  rather than re-investigated from scratch. Precise mechanism: the scrape
  happens *after* each request completes (see the code comment directly
  above the scrape call), and at `--concurrency 1` nothing else is ever
  in flight at that instant by construction — not simply "too few total
  requests," which was the first-pass framing.

### Docs updated to close out the item cleanly

`vllm-benchmark/README.md`'s "Known limitation: GPU utilization on Apple
Silicon" section (which described this as an open gap since Task 3) was
rewritten to describe the resolution: what `gpu_util_pct` is, what has to
be running for it to be non-`None`, and the graceful-degradation behavior
if the exporter isn't up. `run_bench.py`'s own module docstring updated to
match. Item #3 is closed: real code change, real live-server verification,
documentation brought back in sync with actual behavior.

## Meta-lesson for the writeup

Two different threads from earlier in this log converged cleanly in one
session: the "verify state directly, don't infer it from a proxy signal"
discipline (used here to catch the wrong "cold start" hypothesis by
checking the *other* rows, not just the one that fit the story), and the
"recognize when a pattern is already confirmed, don't redundantly
re-investigate" discipline from the concurrency=10 entry (used here to
correctly identify the `0.0` scheduler fields as the same already-solved
Task 3 finding instead of opening a new investigation). Neither discipline
is new to this log, but this is the first entry where both were needed
back-to-back on two questions raised by the same single CSV.

## Entry: Item #2 — full hands-on redo, plus a prefix-caching and tensor/pipeline-parallelism detour

**Date:** 2026-08-07

### Why this session happened at all

Picked back up after a session restart (unrelated audio/dictation issue),
with an explicit concern that the depth built up in earlier sessions had
gone "foggy" after moving from a different Claude surface to Claude Code —
even with `PHASE1_LOG.md` and `NOTES_PERSONAL.md` available to read, the
understanding itself hadn't transferred just from re-reading notes about
it. Decision: redo all four item #2 flags from scratch via predict-then-
verify, live against the actual running server, rather than accept the
existing "closed" status at face value. A second, deliberate change this
session: the user drove every terminal command personally from this point
on, with Claude predicting/guiding/verifying rather than executing
directly — a working-style change, not just a content redo.

### Two infrastructure bugs hit before any flag testing could start

- **Backgrounding a server inside a monitored bash call is fragile.**
  First restart attempt launched `serve.sh` with `&` inside the same tool
  call that also ran a polling loop waiting for it to become ready. When
  the polling loop hit its own timeout, the entire process group —
  including the backgrounded server — was killed with it, mid-boot.
  Fixed by fully detaching the server (`nohup ... & disown`) in its own
  call, separate from any monitoring loop.
- **Long paths break when pasted into a wrapped terminal.** Commands built
  around the session's long scratchpad path repeatedly got mangled on
  paste — a multi-line wrap turned one path into three broken shell
  tokens, and a `curl` command's quotes were similarly split across wrapped
  lines, sending empty headers and a 400 from the server. Root cause both
  times was terminal line-wrapping, not the commands themselves. Fixed by
  writing throwaway helper scripts (`fire_requests.sh`,
  `fire_diverse_prompts.sh`, `fire_boundary_test.sh`, `tp_test.sh`) to the
  repo root and a short `serve.log` symlink pointing at the real log file
  in scratchpad, so every command actually typed or run was short enough
  to survive a terminal wrap.

### `--max-num-seqs` — re-confirmed live, plus a real gap in the first attempt

Restarted with `MAX_NUM_SEQS=1`, fired 3 concurrent requests, watched
`Running:`/`Waiting:` live via `tail -f`. First attempt reproduced the
exact same limitation the original Task 2 entry hit: with `max_tokens=60`,
requests finished faster than the server's ~10s periodic status tick, so
the log only ever showed `Running:1/Waiting:1`, never the expected
`Running:1/Waiting:2`. Re-ran with `max_tokens=400` (same fix as the
original entry) and got the clean sequence live in the user's own
terminal: `Running:1/Waiting:2` → `Running:1/Waiting:1` →
`Running:1/Waiting:0`, `Running` never once leaving 1 across the full run.

A second, smaller version of the same timing gap resurfaced later in the
session (see prefix-caching section below) — caught independently by the
user rather than assumed away, confirmed by reading the raw log's request-
completion line against the first status tick's timestamp rather than
guessing.

### Detour: prefix caching is block-granular, proven with a wrong prediction first

A side question ("would 3 completely different prompts still show a
nonzero prefix-cache hit rate, since they all go through the same chat
template wrapper?") turned into its own small experiment. Predicted
result: "way more than small," nonzero. Actual result, on a fresh-restart
server: a flat **0.0%** across the entire run.

Rather than accept an approximate explanation, used vLLM's own
`/tokenize` endpoint to get exact token counts instead of guessing from
word counts. Confirmed the server's own boot log already states
`block_size=16` (`cache_policy.py:846`) — prefix-cache hits require an
entire 16-token block to match exactly from position 0, not a partial or
approximate match. The shared chat-template wrapper alone is only ~3
tokens (`<s>[INST] `), nowhere near enough to fill one block before the
three prompts' actual content diverges — hence the flat 0%.

Built a second, precisely-engineered test to prove the mechanism
positively rather than just negatively: used `/tokenize` iteratively to
find a shared-prefix sentence landing at exactly 18 tokens before the
divergent ending word (comfortably past the 16-token boundary), then fired
3 requests sharing that prefix with different one-word endings ("cats",
"dogs", "birds"). Result: hit rate climbed from 0.0% baseline to **38.1%,
then 50.8%** — exactly the shape predicted (request 1 pays 0% since
nothing is cached yet; requests 2 and 3 reuse the now-cached first block).

Caught one more real gap mid-experiment: the log skipped from
`Waiting:1` straight to `Waiting:0`, missing the expected `Waiting:2`
moment. Traced directly in the raw log file: a request's `200 OK` line
appeared *before* the first periodic status tick fired — same root cause
as the `--max-num-seqs` timing gap above (short `max_tokens=200` here,
combined with the cache hit skipping some prefill work, made the request
fast enough to finish before the first ~10s tick).

### `--tensor-parallel-size` — live error, then real source, then a genuine correction mid-explanation

Predicted correctly at the PM level (capacity problem, splits weights
across GPUs rather than replicating). Triggered the real error live via a
direct `vllm serve --tensor-parallel-size 2` invocation (not wired into
`serve.sh`): `NotImplementedError` at config-validation time, before the
model even loads. Read the actual source
(`vllm_metal/platform.py:480-494`) rather than trusting the error message
alone — confirmed the two stacked reasons already documented in the
2026-08-05/06 entry (no cross-device collective wired up, and one GPU per
Mac to begin with), and found a new detail this pass: pipeline parallelism
**is** supported on Metal, using simple point-to-point `mx.distributed`
send/recv between stages rather than the collective all-reduce TP would
need.

Explaining PP conceptually surfaced a real gap, caught by a direct
question rather than self-review: the first explanation only described
PP's single-request behavior (one worker idle waiting for a handoff),
which implies almost no parallelism at all. Corrected: PP is only
genuinely parallel across a *continuous stream* of many requests, each
pipeline stage busy on a *different* request simultaneously (assembly-
line style) — for one isolated request it's close to sequential plus
handoff overhead. Also clarified, on request: PP as an algorithm is
platform-agnostic (used on CUDA clusters and TPU pods long before Metal
existed); only the transport is platform-specific (NCCL/NVLink on CUDA
vs. `mx.distributed` here).

### `--max-model-len` — confirmed independent of `kv_budget`, exact 4x math

Predicted, this time correctly on the first try, that `kv_budget` and
`max_tokens_cached` would be unaffected by `--max-model-len`, and that
concurrency would scale inversely. Restarted at `max-model-len=512`:
`kv_budget=6.91GB`, `max_tokens_cached=52704` — identical, digit for
digit, to the `2048` baseline captured earlier in this same session.
vLLM's own boot log confirmed the concurrency math directly:
`Maximum concurrency for 512 tokens per request: 102.94x`, versus
`25.73x` at `2048` — exactly a 4.0x ratio, matching the 4x reduction in
`--max-model-len` (2048→512) exactly.

One real communication gap surfaced here too: stating "`--max-model-len`
doesn't touch the KV budget" and "there's a 4x impact" back-to-back,
without flagging that those two sentences describe two *different*
quantities (the fixed-size pool vs. how many requests share it), read as
contradictory. Resolved by writing out the full four-step formula
explicitly (`usable_metal` → `kv_budget` → `max_tokens_cached` →
`concurrency = max_tokens_cached / max_model_len`) and pointing at the
exact step where the flag enters — the last one only. Worth remembering
for the eventual writeup: when a "doesn't affect X, but does affect Y"
claim lands as confusing, the fix is usually to separate X and Y
explicitly with real numbers, not to re-explain the same claim differently.

## Meta-lesson for the writeup

This session's most reusable lesson isn't new technical content — it's
that redoing already-"closed" material hands-on, with the learner driving
instead of the assistant, surfaced real gaps that passive review of the
same notes had not: a wrong prediction on prefix caching that led to a
precisely-engineered follow-up test, an incomplete pipeline-parallelism
explanation caught by a sharp question rather than self-correction, and a
confusing "doesn't affect X but affects Y" explanation that only got fixed
once separated into explicit, numbered steps. None of these were
mistakes in the underlying technical facts (item #2 was already correctly
closed in the 2026-08-05/06 entry) — they were gaps in how clearly that
understanding could be *reconstructed and re-explained from scratch*,
which is a different and arguably more interview-relevant skill than
having gotten it right once.

## Entry: Item #10 — PagedAttention + continuous batching (conceptual, with live verification)

**Date:** 2026-08-07

Worked through both halves of item #10 via predict-then-verify, with the user driving articulation and me pushing back on gaps rather than lecturing.

**PagedAttention.** Landed correctly on the two distinct failure modes non-paged KV cache allocation has: internal fragmentation (reserving each sequence's cache to `max_model_len` upfront wastes the entire unused tail for sequences that finish early) and external fragmentation (even sizing to current length, a growing sequence can be blocked by a neighbor sitting in adjacent physical memory, even with free memory elsewhere in the pool). PagedAttention's fix — fixed-size blocks (`block_size=16`, confirmed from this server's own boot log in the item #2 prefix-caching detour) plus a per-sequence block table decoupling logical from physical placement — solves both: on-demand block allocation bounds internal-fragmentation waste to under one block, and non-contiguous physical placement removes the adjacency constraint that causes external fragmentation.

**Continuous batching — live test, three iterations before the signature was actually observable.** Predicted correctly that static batching gates refills on the *whole* batch finishing, while continuous batching (iteration-level scheduling) refills a freed slot immediately. Verifying this live took three attempts, each failing for a distinct, diagnosable reason — a useful sequence in its own right:
1. First attempt used the server's own periodic log line (`tail -f`, ~10s tick). Result: only 3 log lines total, first one already showing `Running:2/Waiting:0` — the entire fire→short-finish→refill cycle happened inside the ~10s gap between ticks. Same root cause as two earlier timing gaps in this project (the original `--max-num-seqs` test, and the prefix-caching detour) — periodic print interval too coarse for fast requests.
2. Second attempt polled vLLM's `/metrics` Prometheus gauges (`vllm:num_requests_running`, `vllm:num_requests_waiting`) directly at 0.2s intervals instead of relying on the log's internal print timer. This produced a real transition but skipped the expected middle state (`Running:2/Waiting:0`) — went straight from `Running:2/Waiting:1` to `Running:1/Waiting:0`. Root cause, reasoned out before re-running: the two requests given identical `max_tokens=250` were processed in the same continuous-batching decode loop, one token per step for every running request together — so they likely hit their token cap on the *same* decode step, meaning the intermediate state may have had a true duration near zero, not just been under-sampled.
3. Third attempt used three deliberately distinct token budgets (150/450/900) so no matter which two requests happened to be admitted first, they couldn't tie. Result: a clean, fully-resolved four-state trace — `Running:2/Waiting:1` (11s) → `Running:2/Waiting:0` (held 3s) → `Running:1/Waiting:0` (6s) → `Running:0/Waiting:0`. The 3-second hold on the middle state is the actual proof: `Waiting` dropped to 0 a full 9+ seconds before `Running` reached 0, meaning the queued request was serviced without waiting for its original batch-mates to fully drain — the one behavior a static-batching scheduler could not produce (static batching would hold `Waiting` at 1 until `Running` also hits 0 simultaneously).

Helper scripts written to repo root for this: `fire_batching_test.sh` (concurrent curl firer, three iterations of tuned `max_tokens`), `poll_batching.sh` (0.2s `/metrics` poller, prints only on state change).

## Entry: Revisit session — items #3, #4, #7 re-derived hands-on (predict-then-verify)

**Date:** 2026-08-08

Following item #10's redo methodology, extended the same predict-then-verify articulation drill to three already-"closed" items whose understanding hadn't stuck: #3 (harness log-schema extension), #4 (saturation curve / `queue_wait_ms` bug), #7 (4-bit vs 8-bit quantization + idle-recovery investigation). Explicit reason given: earlier sessions on this material had Claude executing directly while the user "blindly" ran commands — the working-style change (user drives the terminal, effective from item #2's 2026-08-07 redo) was meant to fix exactly this, but items #3/#4/#7 predated that change and hadn't been redone since.

**Numbering ambiguity resolved before starting.** This repo has two parallel item-numbering schemes — old ad-hoc "Task N" labels vs. the master plan's own Phase 1 item numbers 1-17 — which don't map 1:1 (old Task 3 = harness-build broadly; new item #3 = specifically the `gpu_util_pct` schema extension). Installed `poppler` (`brew install poppler`) to actually read the master plan PDF's Phase 1 task table rather than guess: item #4 = old Task 4 (saturation curve) and item #7 = old Task 7 (quantization) do coincide 1:1; item #3 does not map to old Task 3.

**Item #3 (harness log schema).** Rebuilt the TPOT-vs-ITL distinction via a numeric example (99×20ms + one 500ms outlier token), establishing that percentiles must be taken of the right underlying data — p99 of ITL (raw per-token gaps) surfaces a single-token spike; p99 of TPOT (already-averaged per request) doesn't, since averaging erases the spike before any percentile runs. Connected `queue_depth` and `kv_cache_used_pct` directly to data already watched live in the item #10 session (the `Waiting:` count and `GPU KV cache usage: X%` log field) rather than introducing them as new abstractions. Quick pass on the DCGM-substitute story: no DCGM on Apple Silicon; `powermetrics_exporter.py` sidecar built as a from-scratch translator into Prometheus format; `run_bench.py`'s `scrape_metrics()` generalized to scrape both vLLM's own `/metrics` and the exporter's `:9400/metrics`.

**Item #4 (saturation curve) — retold, then reproduced live.** Retold the four-bug story (thin dataset → `--repeat` → real 15-prompt merge; the `s10` 36,948ms harness queue-wait leak; the real ~22s batched-path warmup at concurrency=5; the final flat-through-25-then-bends-at-50 finding) as narrative, then reproduced the core diagnostic live against the real running server and harness (`~/Documents/vllm-benchmark/scripts/run_bench.py`), restarted without the `--max-num-seqs` cap left over from item #10's testing to avoid confounding harness-side queueing with server-side capacity queueing. Real reproduction (`runs/refresher_c1.csv`, concurrency=1, 8 requests): `queue_wait_ms` climbed cleanly across requests 2-8 (52,100ms → 148,373ms) while their real `ttft_ms` stayed small (955-3,126ms) — the exact 2026-08-02 mechanism, live. Bonus, unplanned finding: the very first request (`s2`) showed a genuine 40,830ms TTFT with near-zero `queue_wait_ms` — correctly diagnosed live, unprompted, as a cold-start/JIT-compile cost (server had just been freshly restarted), applying the `queue_wait_ms`-first diagnostic rule just re-derived for item #3. Confirmed the dataset's `s`-prefix naming convention directly from `datasets/prompts.jsonl` (`s1`-`s15` = the 15 merged support prompts; `1`-`5` = the original synthetic set) rather than guessing — same file, coincidentally, as the original `s10` incident. Hit two environment snags along the way (system `python3` missing `httpx` — fixed by pointing at the repo's own `venv`; a wrapped multi-flag command splitting `--variant refresher_c5` into two shell tokens — fixed the same way as the item #10 session, by writing short throwaway scripts, `run_c1.sh`/`run_c5.sh`, instead of long inline commands).

**Item #7 (4-bit vs 8-bit MLX quantization + idle-recovery).** Rebuilt the core tradeoff from `Task7_Quantization_Tradeoffs.md`'s real numbers (model_memory 4.08GB vs 7.70GB; kv_budget 7.06GB vs 3.28GB; max concurrency 26.3x vs 12.2x; TPOT 65.9ms vs 104.7ms) rather than the generic "8-bit is higher quality" framing — flagged that output quality was never actually measured in this project (no perplexity/eval harness run), an intentional, documented gap rather than an oversight. Explained the TPOT gap via decode's memory-bandwidth-bound nature (half the weight bytes → roughly half the per-token time), connecting to the master plan's own Phase 7 preview material. Retold the idle-recovery investigation in full: the anomaly that didn't fit prior warmup stories, the mystery-traffic confound (later found to be Prometheus, not Grafana, still scraping the server directly), the four replicated sweeps showing a real-but-shifting penalty (no fixed idle-duration threshold), and the GPU-power-stepping hypothesis tested directly with real frequency telemetry and disproved (GPU sits flat at idle baseline for 25+ of 28 seconds, then snaps to full frequency in ~1 second right at token delivery — ruling out the GPU itself and pointing upstream, to an as-yet-unidentified scheduling/dispatch-layer stall). Root-causing that upstream stall was explicitly deferred as backlog, to be picked up after Phase 1 closes, rather than opened as new work mid-revisit-session.

## Meta-lesson for the writeup

Same core finding as item #10's redo, now confirmed across three more items: material that was correctly closed the first time can still fail to be *retained* if it was executed by Claude rather than driven by the learner. The specific gaps this session surfaced weren't in the original technical conclusions (all of which held up under live re-verification) but in recall and re-explanation from scratch — plus one genuinely new, unplanned confirmation live in the data itself (the `s2` cold-start artifact), a good reminder that hands-on redos don't just refresh memory, they can still turn up real, previously-unseen instances of already-understood phenomena.

## Entry: Item #8 — burst test (spike 5→50), and a much larger version of item #7's open mystery

**Date:** 2026-08-08

### Design

Three-phase test against `~/Documents/vllm-benchmark/scripts/run_bench.py`: baseline (`--concurrency 5`, 10 requests), burst (`--concurrency 50`, all 50 fired at once, not ramped), recovery (24 single-request canary probes, 5s apart, same fixed prompt each time). A continuous poller (`burst_poller.sh`, rewritten mid-session after the first version died under load — see below) logged `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc`, and GPU frequency (via the item #3/#6 `powermetrics_exporter.py` sidecar) once per second throughout.

### Discovery #1 — a new, sharper version of a first-batch stall, found before real testing could even start

Two consecutive baseline attempts both showed the *entire first batch* of admitted requests (matching whatever `--concurrency` was set to) stall uniformly for tens of seconds — 66,919-66,932ms across 5 requests in one run, 49,913-49,920ms across 5 in another, `queue_wait_ms` near-zero both times (ruling out a harness artifact). Critically, the second occurrence came only ~8.5 minutes after the first, breaking a pure idle-duration explanation (item #7's original finding needed at least 15-20s and was characterized up to 60s; a mundane 47-minute away-from-keyboard gap explained the very first occurrence, but 8.5 minutes clearly did not). Pattern across all observations today (including item #4's `s2` earlier): **the first batch of concurrent requests in any fresh `run_bench.py` process invocation incurs this stall, regardless of prior idle duration.** Worked around with a throwaway single-request warmup fired immediately before every real measurement (`burst_baseline.sh`, `burst_spike.sh` both updated to do this) — confirmed effective: the warmup absorbs the stall itself, and the real measurement immediately after comes back clean.

### Discovery #2 — the clean burst itself: moderate amplification, and a scoping gap

Once warmed, the real 50-request burst showed TTFT amplification from a 609-1,334ms baseline to a tight 4,395-4,700ms cluster (~3.5-4x) — real but moderate, consistent with operating over the ~25-request comfortable ceiling established in item #4. `kv_cache_usage_perc` only reached ~11-14% at peak, nowhere near its ceiling — because these test prompts are short relative to `max_model_len`, this particular burst design never actually exercised memory-based admission control the way a burst of longer-context requests would. Flagged as a real scoping limitation of this test, not glossed over: a future version testing near-max-length prompts would more directly stress `kv_cache_usage_perc` and could show real queueing (`Waiting > 0`) that this run never triggered.

### Discovery #3 — mid-burst, a live instance of item #7's exact signature

TTFT looked fine, but full completion of all 50 requests took over 4 minutes (the scheduler's `Running` count didn't return to 0 until ~4 minutes after the burst started) — TTFT and total completion time are very different things. During the tail of that drain, GPU frequency repeatedly dropped to idle baseline (789-790 MHz) for 70-90 second stretches while 25-36 requests were still marked `Running`. Verified this isn't consistent with "slowly grinding through decode under contention" (which would keep the GPU busy, just producing less aggregate throughput) — sustained idle frequency while requests are nominally in-flight is the same signature item #7 already characterized and ruled the GPU itself out for. New here: it can happen **mid-burst**, affecting a large in-progress batch, not just a single fresh connection.

### Discovery #4 — the headline finding: recovery never completed in 28 minutes

Caveat stated plainly: due to an operator mix-up, `burst_spike.sh` was accidentally run twice back-to-back (the second run showed TTFT ~10,380-10,412ms, roughly 2.2x worse than the first clean run — itself a notable data point about compounding load) before the recovery phase began. So what got measured was recovery after **two** consecutive 50-request bursts, not an isolated single burst — an honest scoping note for the record, not hidden.

Across 24 canary probes over ~28 minutes, TTFT never returned to baseline. Every probe measured between 20,825ms and 115,626ms (20x-115x baseline), no clear downward trend — if anything, the two worst readings (109,525ms, 115,626ms) came in the second half of the test. Probe 20 didn't just get slow, it hit a genuine `httpx.ReadTimeout` after 121.3 seconds of waiting (`queue_wait_ms=0.07ms`, `connect_wait_ms=961ms` — both normal, confirming this is 100% a server/upstream delay, not a harness artifact).

Cross-referencing the poller against each probe's real TTFT confirmed the mechanism directly: `Running:1` only ever appeared for ~2-4 seconds right before each probe finished, regardless of whether that probe's total TTFT was 20 seconds or 115 seconds. The overwhelming majority of every probe's delay happened in a phase where vLLM's own scheduler hadn't even admitted the request yet — the same "something upstream of the GPU/scheduler" conclusion item #7 reached, now shown to persist continuously for at least 28 minutes after real burst load, not just as a short blip after a controlled idle wait.

### Second infrastructure bug found and fixed mid-session

The first version of `burst_poller.sh` used `set -euo pipefail`; under load, a transient scrape failure (the exporter's 1s timeout being exceeded, or `grep` finding zero matches) caused `set -e` to kill the whole polling loop silently, mid-burst. Rewrote without `-e`/`pipefail`, with explicit `${var:-n/a}` fallbacks after every fetch, so the loop survives transient failures and logs `n/a` instead of dying.

### Where this leaves item #8 and the backlog

Item #8's own three deliverables (queue buildup, p99 amplification, recovery time) are answered, honestly: queue buildup wasn't observed the way expected (short prompts didn't stress KV capacity); p99 amplification was moderate (~3.5-4x) once the unrelated first-batch stall was controlled for; recovery time is the real story — **unbounded within the 28 minutes tested**, not a clean number. This substantially raises the stakes on item #7's already-deferred root-cause investigation: what looked like a minor, bounded (2x-37x, tested only to 60s idle) cold-start quirk there turns out to be capable of persisting for at least 28 minutes after real burst load. Still deliberately deferred per the earlier backlog decision — not reopened mid-item-8 — but now flagged as the single most operationally important open question in this project's Phase 1 work: a production system hitting this after a real traffic spike could be degraded for a very long time, not just briefly.

## Meta-lesson for the writeup

The most valuable result of this task wasn't the planned burst-test deliverables — it was catching, through the same verification discipline applied throughout this project (don't accept a plausible-looking result, cross-reference against real server/poller state), that a phenomenon previously scoped as a minor edge case (item #7's idle-recovery, bounded and rare) is actually much larger and more persistent than believed. Two operator mistakes this session (the accidental double-fire of `burst_spike.sh`, the 47-minute away-from-keyboard gap) turned out not to invalidate the findings — they became part of the evidence, because the discipline was to look at what actually happened in the logs rather than assume the intended experiment ran cleanly.

## Entry: Root-causing the async_eval stall — from three vague candidates to `mx.async_eval` → `eval_impl` → `condition_variable::wait`

**Date:** 2026-08-09

### Why this happened tonight, out of order

Item #7's original idle-recovery investigation named three untested candidate upstream causes (Python asyncio event loop, macOS scheduling of an idle process, MLX's lazy dispatch) and was deliberately deferred to after Phase 1 closes. Item #8's burst test escalated the severity of the same phenomenon (unbounded recovery, ≥28 minutes) but the deferral held. Tonight, while gathering a GPU power reading for item #9's cost model, an ad-hoc single `curl` request (not run through any harness) hit the same stall — 95 of ~115 total seconds at idle-level power, ~19 seconds of real compute — proof via a fourth independent signal that the GPU itself does zero work during the stall. Given the strength and immediacy of this evidence, the user made an explicit, informed decision to reopen the investigation immediately rather than wait for the originally-planned deferral point.

### Two-process architecture discovered first

`ps aux` revealed vLLM actually runs as two separate processes communicating over IPC: the API server (asyncio/HTTP layer) and a separate `VLLM::EngineCore` process (scheduling + MLX/Metal dispatch). Both needed independent stack-trace capture.

### `py-spy` — pinning the stall to one exact line

Installed `py-spy` (`brew install py-spy`) and wrote `stall_investigate.sh`: fires a request, then dumps both processes' Python stack traces every 5 seconds for as long as it runs. Result, across 33 consecutive dumps spanning 5.5 minutes (`00:24:05`-`00:29:43`) of a reproduced stall: the `EngineCore` process's MainThread sat at the **exact same line, never moving** — `vllm_metal/v1/model_runner.py:582`, inside `_submit_paged_forward_outputs`. Read the actual source: line 582 is `mx.async_eval(*eval_outputs)`. Despite the "async" name implying non-blocking behavior, the Python thread was blocked inside this call for the entire stall.

### `sample` (native) — a false start caught by checking timestamps, then a clean read

Installed nothing new (macOS's built-in `sample` tool) to capture native C/C++ stack frames beneath the Python call. First attempt (`stall_native_sample.sh`) produced a trace showing MainThread stuck in `PyThread_acquire_lock_timed` — looked plausible, but before trusting it, cross-checked the response file's mtime (`00:51:19`) against the sample report's own timestamp (`01:44:57`) — a 53-minute gap. Root cause: the request had actually finished fast (no stall that run), but a `kill -0`-based liveness check falsely reported it still running (classic zombie-PID race), and `sudo sample` then sat blocked on an expired sudo credential's password prompt for most of an hour before actually running — capturing ordinary idle state, not the stall. Caught and discarded before drawing any conclusion from it, per this project's standing verification discipline.

Fixed the script (`sudo -v` up front to prime credentials; `ps -p` instead of `kill -0` to avoid the zombie race; elapsed-time logging to self-verify future runs) and re-ran. Confirmed valid this time (104s total request, 45s sample fully inside that window). Result, unambiguous — **100% of 38,629 samples across the full 45 seconds**, on MainThread:
```
mlx::core::async_eval(...)
  mlx::core::eval_impl(...)
    std::condition_variable::wait(...)
```
A genuine C++ condition-variable wait inside MLX's own native code (`libmlx.dylib`), not the GIL, not `asyncio`, not a Python-level lock. Combined with the GPU-idle telemetry gathered earlier, this points at `eval_impl` waiting on MLX's own internal worker/stream threads (also observed idling on their own separate condition variables) to actually pick up and dispatch queued work — with the remaining open question being why the OS delays scheduling that worker thread after idle.

### Resolution-plan test: `caffeinate` — ruled out as a simple fix

Proposed a tiered resolution plan (keep-alive mitigation already validated via the warmup-request pattern; a fast `caffeinate` test; upstream bug report; deeper QoS/priority investigation if needed). Restarted the server fresh under `caffeinate -i ./scripts/serve.sh` and repeated the native-sample test against the new PIDs. Result: the stall still occurred (63s total), and the same `async_eval`/`eval_impl`/`condition_variable::wait` mechanism dominated the trace (~91.6% of samples) — **`caffeinate` alone does not fix this.** One minor difference noted honestly: this capture showed more movement through varied `eval_impl` code offsets than the two prior static captures, an ambiguous signal not treated as a win.

### Where this leaves the investigation

Root cause resolved to a specific, evidence-backed mechanism — `mx.async_eval()` blocking inside MLX's native `eval_impl` on a condition variable, most likely waiting on an MLX-internal worker thread that isn't being scheduled promptly by macOS after idle — a dramatically more specific finding than the three vague candidates item #7 started with, though the exact OS-level throttling mechanism remains one layer short of fully confirmed. Practical resolution: the throwaway-warmup-request pattern (already adopted project-wide tonight) remains the one validated mitigation. Next steps, not pursued tonight: check actual thread QoS/priority values directly (Instruments or `taskpolicy`), and file this as a bug report against `vllm-metal`/MLX given the unusually strong evidence trail (GPU power telemetry, Python stack traces, native C++ stack traces, all converging on the same call site).

## Meta-lesson for the writeup

This session is the clearest example yet of taking an already-deferred investigation, reopening it deliberately (not accidentally) when new evidence justified it, and pushing it all the way from three vague hypotheses to one exact line of code and a specific native synchronization primitive — using nothing but tools already available or trivially installable (`py-spy`, `sample`, `caffeinate`), each cross-checked against real timestamps before being trusted. The false-start `sample` capture is worth remembering on its own: a plausible-looking stack trace was caught and discarded specifically because the timestamps didn't add up, not because anything about the trace itself looked wrong — the same "verify against real state" discipline this project has applied to metrics and logs throughout, now applied to profiler output.

## Entry: Item #9 — cost model implemented (`cost_per_1k_tokens` / `tokens_per_dollar`)

**Date:** 2026-08-09 (resumed later same day)

Picked back up where item #9 was paused (formula designed by hand, code not yet written). Resuming after ~11 hours of server idle time, the first real request hit the same stall pattern from last night (~60s TTFT) — confirmed the throwaway-warmup pattern is now a genuine standing habit for this project, not a one-off fix.

**Implementation.** Added `cost_per_1k_tokens_usd`/`tokens_per_dollar` to `run_bench.py`, opt-in via `--cost-hardware-usd`. Design decisions carried over from the hand-derivation: hardware purchase price amortized over an assumed lifetime (`hardware_usd / (lifetime_years × 8760)`) into an hourly rate; electricity from a *continuously sampled* background task reading `powermetrics_gpu_power_milliwatts` throughout the run's entire wall-clock duration (not per-request snapshots) — deliberately including idle/queued gaps, matching how real cloud GPU billing charges for the hour regardless of utilization. Combined into one `$/hour`, scaled by the run's actual duration, divided by total output tokens produced. Written as a run-level summary (`{variant}_cost.json`) alongside the existing per-request CSV/JSONL, since cost is a shared-infrastructure economic quantity, not something that cleanly attributes to one request among many concurrent ones. `.gitignore` extended (`runs/*.json`) to match the existing CSV/JSONL exclusion pattern.

**Live validation.** First real run (`--concurrency 5 --max-samples 10 --cost-hardware-usd 2000`) produced `$0.005004/1K tokens`, `199,850.9 tokens/$`, `avg GPU power 3.69W` (189 samples over 251.1s) — verified by hand against the formula and consistent with the prediction from last night's manual derivation (electricity negligible next to hardware amortization on this hardware, ~200x smaller). The run itself was heavily affected by a recurrence of the first-batch stall (5 of 10 requests clustered at ~53.4-53.5s TTFT) despite the single-slot warmup already done — a real, useful data point that a warmup at one concurrency level may not fully "warm" a different concurrency level's first batch, and a concrete demonstration in dollar terms (not just latency) of why the stall matters: real wall-clock time burned without proportional token output directly inflates `$/1K tokens` under this project's own averaging-includes-idle design choice.

Also wrote a standalone math reference (`INTERVIEW_PREP.md`) at the user's request, framed explicitly for re-derivation practice before an interview rather than narrative recall — the user described their own learning style as preferring to keep re-working mathematical logic from scratch rather than just remembering a story about it.

**Still open:** a clean (non-stalled) comparison run, to quantify precisely how much the stall costs in dollar terms versus a smooth run at the same concurrency; wiring the cost summary into a Grafana panel per the master plan's own spec.
