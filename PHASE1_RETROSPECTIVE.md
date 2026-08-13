# Phase 1 Retrospective

Closes master plan item #17: "Phase retrospective: score yourself on skills
covered. What is still at 5/10 or below?" Grounded in the real work logged in
`PHASE1_LOG.md` (technical narrative) and `NOTES_PERSONAL.md` (skills
tracker, gitignored) — not reconstructed from memory.

## §1 — Item-by-item scorecard (Phase 1, 17 items)

| # | Item | Tier | Status |
|---|---|---|---|
| 1 | Deploy vLLM locally, serve via OpenAI-compatible API | T1 | **Closed** — Mistral-7B-Instruct-v0.3-4bit on M4 Mac mini, 16GB unified memory |
| 2 | Core vLLM params (`--tensor-parallel-size`, `--max-model-len`, `--gpu-memory-utilization`, `--max-num-seqs`) | T1 | **Closed** — done twice; fully redone hands-on 2026-08-07 after retention gap |
| 3 | Extend log schema (`gpu_util_pct`, etc.) | T1 | **Closed** — revisited hands-on 2026-08-08 |
| 4 | Saturation curve, concurrency 1→5→10→25→50 | T1 | **Closed** — revisited hands-on 2026-08-08; ceiling ~25 concurrent identified |
| 5 | DCGM exporter (GPU util, memory, SM active, bandwidth) | T1 | **Closed, adapted** — DCGM is NVIDIA-only; built a `powermetrics`-based exporter instead, gaps stated explicitly rather than faked |
| 6 | Grafana dashboard, 6 panels | T1 | **Closed** — verified live against real 10-user traffic, all 6 panels confirmed mutually consistent |
| 7 | bf16 vs fp8 comparison (reframed: 4-bit vs 8-bit MLX quantization) | T1 | **Closed** — revisited hands-on 2026-08-08; also surfaced the idle-recovery finding later escalated by #8 |
| 8 | Burst test, spike 5→50 | T1 | **Closed** — 2026-08-08; headline finding: recovery time unbounded within 28 min tested |
| 9 | Cost model (`cost_per_1k_tokens`, `tokens_per_dollar`) + Grafana panel | T1 | **Closed** — 2026-08-09; real dollar figures, stalled run costs ~3.6x clean run |
| 10 | PagedAttention + continuous batching (conceptual) | T1 | **Closed** — 2026-08-08; both mechanisms proven live, not just explained |
| 11 | Streaming API (SSE vs WebSocket vs HTTP), TTFT/TPOT streaming-vs-batch | T1 | **Closed** — 2026-08-09 |
| 12 | Read LLM-Emu 2026 paper, 1-page summary | T2 | **Not started** |
| 13 | Read Alibaba RTP-LLM paper, note industry alignment | T2 | **Not started** |
| 14 | Prefix caching experiment, measure TTFT delta | T2 | **Partial** — real experiment done as a detour during item #2's redo (`block_size=16` mechanism, 38.1%→50.8% hit rate with an 18-token shared prefix), but not confirmed against this item's exact wording |
| 15 | Interview prep: TTFT vs TPOT vs ITL, when each dominates (chatbot vs batch vs coding assistant) | T1 | **Partial** — a TTFT/TPOT/ITL cheat sheet exists in `INTERVIEW_PREP.md`, but framed around streaming-vs-batch (item #11), not this item's specific chatbot/batch/coding-assistant workload framing |
| 16 | Push to GitHub with README, architecture diagram, sample Grafana screenshot | T1 | **Partial** — repo is pushed and README now links the technical log (this session), but no architecture diagram or Grafana screenshot committed yet |
| 17 | Phase retrospective | T1 | **This document** |

**11 of 11 T1-required items with hands-on/conceptual work are closed** (1-4, 6-11); item #5 closed via a stated adaptation. Genuinely open: two T2 bonus papers (12, 13), one T2 partial (14), and two T1 wrap-up partials (15, 16) — all light, well-scoped remaining work, not new investigation.

## §2 — Highlights (3 most interesting entries, for a portfolio index)

1. **The burst test found something bigger than the burst test** (item #8, 2026-08-08). Set out to measure queue buildup, p99 amplification, and recovery time for a 5→50 concurrent spike. Queue buildup and amplification came back close to expected (~3.5-4x). Recovery didn't: 24 canary probes over 28 minutes never returned to baseline — every single one came back 20x-115x slower, one hit an outright timeout, with no downward trend. Cross-referencing against a live scheduler poller proved the delay sat *upstream* of vLLM's own scheduler, not inside it — turning what had been scoped as a minor, bounded cold-start quirk into the single most operationally important open question in the project.

2. **Root-causing it: `mx.async_eval` → `eval_impl` → `condition_variable::wait`** (2026-08-09). Chased the item #8 finding to ground truth using `py-spy` (pinned the exact Python line across 33 dumps) and macOS's native `sample` tool (100% of 38,629 samples over 45s landing on one C++ call). Along the way, caught and discarded a false-positive capture by checking real file timestamps rather than trusting a plausible-looking stack trace — the same verification discipline applied to profiling tools, not just metrics. Found independent corroboration on a different serving stack and different hardware (`ollama/ollama#16170`). Result: a well-evidenced, honestly-scoped upstream bug report, not an overclaimed fix.

3. **A concrete dollar figure for a latency curiosity** (item #9, 2026-08-09). Built a cost model (amortized hardware + continuously-sampled electricity) and used it to answer a question the burst-test/async_eval work could only describe qualitatively before: a stalled run costs **~3.6x more per 1K tokens** than a clean one on identical requests ($0.005004 vs $0.001394). Wired end-to-end into Grafana via a purpose-built exporter, verified exporter → Prometheus → dashboard rendering all agreed. Turns three separate investigations (reliability, root-cause, cost) into one throughline with a real number at the end of it.

## §3 — Skills self-assessment

Scored against the master plan's own five-layer hybrid role definition. Anchored in real evidence from this phase, not aspiration.

| Skill layer | Score | Evidence |
|---|---|---|
| **ML Systems Engineer** (serving, batching, KV cache, quantization, prefill/decode) | 8/10 | Deepest coverage this phase — core params (#2), quantization tradeoffs (#7), PagedAttention/continuous batching proven live (#10), prefix caching mechanism (block-granular, confirmed via `/tokenize`) |
| **AI FinOps Analyst** (cost/token, cost/request, concurrency sweet spot) | 7/10 | Full cost model + Grafana wiring (#9), real stalled-vs-clean dollar comparison. Not yet built: per-user/team budget attribution (that's Phase 5's scope, not this phase's) |
| **SRE / Reliability** (admission control, backpressure, incident analysis) | 6/10 | Genuine incident-analysis work (#8's burst test, the async_eval root-cause chase, a drafted upstream bug report) and real infra bugs caught mid-session (a `set -e` poller dying silently, a Grafana crash-loop from stale persisted state). Not yet done: a written runbook, or a dedicated admission-control/backpressure experiment |
| **Performance Engineer** (saturation curves, tail latency, SLOs, load modeling) | 7/10 | Saturation curve 1→50 with a real identified ceiling (~25 concurrent) (#4), TTFT/TPOT/ITL percentile reasoning including catching the "percentile of an average erases the spike" mistake (#3, #11). Not yet done: formal SLO threshold-setting, regression-gate work (that's Module 2/Phase 8 territory) |
| **GPU Platform Engineer** (CUDA mental model, DCGM, multi-GPU, MIG, Nsight) | **3/10 — flagged, at or below 5/10** | This phase is Apple Silicon end to end: no CUDA, DCGM adapted rather than used natively, Instruments/Nsight-equivalent tooling explicitly not installed (a real, stated gap in the async_eval investigation, not glossed over). By design, not failure — Phase 7 ("GPU-Aware Deep Dive: CUDA model, Triton, Nsight, multi-GPU") is where this is meant to be built |

**Cross-cutting meta-skill, not itemized above:** the predict-then-verify / verification discipline itself. Demonstrated repeatedly and consistently across the phase — catching a false-positive profiling capture via file timestamps, catching an operator double-fire of a test script from echo text alone, rejecting a "plausible-looking" 3-line log as insufficient evidence three separate times on the same experiment (#10), and refusing to accept a stalled-vs-clean cost comparison until warmup and real-test timing were airtight (#9). This is arguably the strongest single outcome of the phase and the one most worth naming explicitly in an interview.

## §4 — What's still open

- T2 bonus items #12, #13 (two papers) — not started, explicitly bonus tier
- Item #14 (prefix caching) — needs a wording check against the existing detour before deciding whether it's already satisfied or needs a dedicated re-run
- Item #15 — needs the specific chatbot/batch/coding-assistant TTFT/TPOT/ITL framing added to the existing cheat sheet
- Item #16 — architecture diagram and Grafana dashboard screenshot, for the GitHub README
- The async_eval upstream bug report (`ASYNC_EVAL_STALL_REPORT.md`) — drafted and reviewed, not yet filed; genuinely adjacent to Phase 1's numbered scope, not part of it (see `PHASE1_LOG.md`'s 2026-08-12 scope-decision note)

## §5 — Spoken interview version

Six findings, written as they'd actually come out in conversation, not as
something read aloud. First three are §2's highlights; the remaining three
are additional strong material pulled from the full log. Where the
underlying work is partial or unresolved, the spoken version stays honestly
partial too — no polishing an open question into a closed one.

### The burst test (item #8)

**How I'd say this out loud:** "I ran a pretty standard load test — spike from 5 to 50 concurrent requests, see how the server handles it and how long it takes to recover. The spike itself was fine, latency went up about 3.5 to 4x, which tracked with what I already knew about this server's comfortable ceiling. The surprising part was recovery. I fired single test requests every 5 seconds for almost half an hour afterward, and it never came back to baseline — every single one of them was 20 to over 100 times slower than normal, and one just timed out. I dug into the scheduler's own logs and found the request wasn't even being picked up for processing most of that time — so whatever's wrong is happening before the server even starts working on it, not during."

**If they push back or ask a follow-up:** *"Did you fix it?"* — No, not yet. I tracked it down to where it's happening at a code level (that's a separate piece of work I can walk through), but the actual fix — why the OS or the ML framework is delaying that handoff — is still an open investigation. Right now the only mitigation is a workaround in my test harness, firing a throwaway request before anything I actually want to measure.

### The async_eval root-cause chase (adjacent investigation, not a numbered plan item)

**How I'd say this out loud:** "That recovery-time bug I mentioned — I didn't want to leave it as just 'something's slow sometimes,' so I traced it. Used a Python-level profiler first, which pointed at one exact line calling into the ML framework's async execution code. Then I used a native system-level profiler to see what was happening underneath that call, and it showed the process was blocked on a genuine thread-synchronization wait — a condition variable — inside the framework's own compiled code, not my code and not vLLM's. I also caught myself almost being fooled by a bad trace capture at one point — the timestamps didn't line up, so I threw that result out and did it again properly. I found someone else hitting the exact same symptom on completely different hardware and a different serving stack, which made me more confident this is a real, general issue and not something specific to my setup."

**If they push back or ask a follow-up:** *"So did the maintainers respond, was it merged?"* — I haven't filed it yet. I've drafted the report and it's reviewed and ready, but submitting it upstream is still a decision I'm sitting on, not something that's happened. And even the root cause itself isn't 100% nailed down — I have a well-evidenced leading hypothesis about *why* the OS is delaying that thread, but I didn't get the one tool that would have confirmed it directly (it needed a large Xcode install I decided wasn't worth it for one more confirming data point on top of four independent ones I already had).

### The cost model (item #9)

**How I'd say this out loud:** "Once I had that stall characterized, I wanted to know what it actually costs, not just that it's slow. So I built a cost model — amortized hardware cost plus real electricity draw, sampled continuously off the machine's own power telemetry, not estimated. Then I ran the same set of requests two ways, once with the stall happening and once clean, and compared. The stalled run cost about 3.6 times more per thousand tokens than the clean one. I wired that straight into Grafana too, so it's not a one-off number in a spreadsheet, it updates with every run."

**If they push back or ask a follow-up:** *"Is that dollar figure realistic for production?"* — Not directly — it's amortized off a $2,000 Mac mini over a 2-year lifetime plus my own electricity rate, so the absolute numbers are specific to my hardware, not a cloud price. What I'd stand behind is the *ratio* — a 3.6x cost multiplier from an otherwise-invisible latency bug is the actual finding, and that relationship would hold directionally anywhere this stall shows up.

### The saturation curve and its ceiling (item #4)

**How I'd say this out loud:** "I ramped concurrent requests from 1 up to 50 and tracked p50, p95, and p99 latency plus total throughput at each step. Latency stayed pretty flat through about 10 concurrent requests, climbed gradually through 25, and then p95 and p99 both bent sharply upward at 50 — a real inflection point, not noise. Throughput told the same story from a different angle: it actually peaked at concurrency 25 and then dropped at 50, below even what I was getting at concurrency 5. So for this model on this hardware, 25 is roughly the practical ceiling — past that you're paying a latency tax without getting more work done."

**If they push back or ask a follow-up:** *"Why exactly does it fall over at 50 and not, say, 35?"* — Honestly, I don't have a sharper number than "somewhere between 25 and 50" — I never ran an intermediate point like 35 or 40 to narrow it down, and I flagged that as a gap rather than guessing at a precise threshold I didn't actually test.

### The warmup-cluster detection in the benchmark harness (part of item #4)

**How I'd say this out loud:** "Early on, my own analysis script was quietly wrecking the saturation numbers — the first batch of requests in any fresh run always eats a one-time warmup cost, tens of seconds, and that was dragging my p50 up to something like 72 seconds instead of the real 1.3 seconds. So I built a detector for it, but I deliberately didn't hardcode a time cutoff, because that breaks the moment you change models or hardware. Instead it flags a request as warmup only if two things are both true: it got processed instantly with no queue wait, and its latency is more than 5x the rest of that run's own median. I also wrote a synthetic test for the detector before trusting it on real data, and that test actually caught a mistake in my own test setup before it could hide a real bug."

**If they push back or ask a follow-up:** *"How do you know that 5x threshold isn't just as arbitrary as a hardcoded one?"* — It's relative to each run's own steady-state median, not a fixed number, so it self-adjusts if the model or hardware changes — that's the part that matters, more than the specific multiplier of 5. I picked 5x because it cleanly separated the two populations in every run I actually had data for; I haven't stress-tested it against a run where the gap is much narrower.

### The DCGM-substitute exporter (items #5/#6)

**How I'd say this out loud:** "Part of the Grafana dashboard spec called for GPU utilization and memory bandwidth panels, normally fed by NVIDIA's DCGM exporter — but I'm on Apple Silicon, so DCGM just isn't an option. Rather than leave those panels empty or fake a placeholder, I built my own exporter around macOS's `powermetrics` tool. I actually went and found a real sample output file for it before writing any parsing code, so I wasn't guessing at the format. One real gap I hit: there's no memory-bandwidth metric on Apple Silicon at all, because the unified memory architecture doesn't have a separate VRAM bus for that number to describe. So instead of leaving that panel blank, I substituted GPU clock frequency there — and I made sure that substitution is stated explicitly in three places: the panel description, the exporter's own metric documentation, and the README. It's not hidden."

**If they push back or ask a follow-up:** *"Isn't substituting a different metric kind of misleading on a dashboard?"* — Only if it's silent, which is why I was deliberate about labeling it everywhere rather than letting someone assume they're looking at real bandwidth data. And functionally it still answers the question the panel exists for — is the GPU under load — just through a different, honestly-disclosed signal.
