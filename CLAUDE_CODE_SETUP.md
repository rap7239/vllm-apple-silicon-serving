# Setting Up Claude Code for This Project

This is a reference doc for moving Phase 1 work from Cowork (the desktop chat app) to
Claude Code (a terminal tool). Written so you can also use it later if you forget how
any of this works — each section explains *why*, not just *what to type*.

---

## 1. Why we're switching tools (ELI5)

Cowork runs Claude in a sandboxed Linux box that is *not* your Mac. It can read/write
files you've explicitly shared with it, and run shell commands — but only inside that
sandbox. Your vLLM server runs on `localhost:8000` on your actual Mac mini. The sandbox
has no network path to your Mac's localhost, the same way your phone can't `curl`
something running on your laptop just because they're both on your desk.

Claude Code is different: it's a terminal program you run *directly on your Mac*, in
your project folder. When it runs `curl localhost:8000`, that's your Mac's own shell
doing it — no sandbox, no isolation. Same for git, same for editing `serve.sh`, same
for appending to `PHASE1_LOG.md`. This is why the rest of Task 7 (anything touching the
live server) needs to happen here instead.

---

## 2. Install Claude Code

You likely have Node.js already (needed for other tooling), but confirm:

```bash
node --version
```

If that fails, install Node first: `brew install node`.

Then install Claude Code globally:

```bash
npm install -g @anthropic-ai/claude-code
```

Verify it installed:

```bash
claude --version
```

## 3. Log in

Run `claude` from anywhere once to trigger the login flow — it'll open a browser tab
to authenticate with your Anthropic/Claude account (same account works for both
Cowork and Claude Code; no separate signup).

```bash
claude
```

Follow the browser prompt, then you'll land in an interactive terminal session.
Type `/exit` or press `Ctrl+C` twice to leave for now — we'll come back to it properly
in the right folder.

## 4. Start Claude Code in the right place

This part matters: Claude Code's filesystem/shell access is scoped to whatever
directory you launch it from (plus subdirectories). You want to launch it from your
actual project root, not your home directory or Desktop.

```bash
cd ~/Documents/vllm-apple-silicon-serving
claude
```

From inside that session, Claude Code can now see `PHASE1_LOG.md`, `scripts/serve.sh`,
and everything else in the repo — and can also reach `~/Documents/vllm-benchmark` if
you ask it to (just tell it the path; it can `cd` there or read across, since your
shell has access to your whole filesystem, not just the launch directory).

## 5. First message to send it

Paste the detailed context prompt below as your first message in the new Claude Code
session. This replaces the "opening prompt for a new chat" I gave you earlier — same
content, just handed to a tool that can actually act on it.

---

### Opening prompt to paste into Claude Code

> I'm continuing Phase 1 of a 22-week LLM Performance Engineering Master Plan, working
> toward a career transition into performance engineering. I keep a `PHASE1_LOG.md` in
> this repo as a running narrative log (decisions, bugs, lessons) — please read it
> first for full context before doing anything else. I want hands-on-first work, ELI5
> explanations where useful, rigorous verification against real evidence (server logs,
> direct testing) rather than assumptions, and proportionate scope — don't over-build
> beyond what a task actually asks for.
>
> **Current task: Task 7**, reframed (with my approval) from the plan's original "bf16
> vs fp8" comparison to "4-bit vs 8-bit MLX quantization comparison," since this
> project runs on Apple Silicon via vllm-metal, which supports neither bf16 nor fp8.
>
> **Where things stand:**
>
> 1. Fixed two real bugs this session: `serve.sh`'s `VLLM_BIN` default pointed at a
>    nonexistent path (fixed to `~/.venv-vllm-metal/bin/vllm`), and a missing
>    `xgrammar` dependency (fixed via `pip install -r requirements-metal.txt`).
>
> 2. Collected comparison sweep data: `8bit-concurrency1.csv/.jsonl` and
>    `8bit-concurrency10.csv/.jsonl` in `vllm-benchmark/runs/`, to compare against
>    existing 4-bit data (`concurrency1.csv`, `concurrency10.csv` from Task 4).
>
> 3. Mid-run, discovered an anomalous TTFT spike (request `s2`, 36,978ms) at
>    concurrency=1 on the 8-bit run. Confirmed via server logs it's a genuinely new
>    failure mode — an "idle-recovery cost": TTFT spikes on requests following a
>    period where the vLLM engine went fully idle (`Running: 0, Waiting: 0`), distinct
>    from the already-documented cold-start/warmup-cluster pattern from Task 4/6.
>
> 4. Built `vllm-benchmark/scripts/idle_recovery_test.py` to isolate this: send a
>    warm-up request, idle a controlled duration, send a test request, report both
>    TTFTs. Ran it at idle durations of 10s/20s/35s — results looked reproducible but
>    showed a counterintuitive pattern (penalty shrinking as idle time grew). Extended
>    to 7 points (5/10/15/20/35/45/60s) to check if that was a real trend or noise.
>
> 5. **Critical confound discovered**: the extended 7-point run showed wildly unstable
>    "warm-up" TTFTs (some 100,000ms+, when warm-up should be cheap) and no consistent
>    relationship between idle duration and post-idle TTFT. Server logs revealed a
>    steady ~90-second cycle of real `POST /v1/chat/completions` traffic during the
>    test window that the test script itself never sent — meaning something else was
>    hitting the server the whole time, invalidating **all 7 idle-recovery data
>    points** collected so far.
>
> 6. We killed and restarted the vLLM server clean, closed Grafana (to rule out its
>    traffic, though this turned out not to be the real cause — Grafana's 5s panel
>    refresh only hits Prometheus, not vLLM directly), meaning to redo the sweep with
>    nothing else running.
>
> 7. **Where it broke**: after restarting, all 7 re-run attempts at
>    `idle_recovery_test.py` failed identically with `404 Not Found` on
>    `/v1/chat/completions` (server responds, just doesn't recognize the route).
>    `curl http://localhost:8000/v1/models` returned `{"error":"Unauthorized"}` —
>    meaning the server is up and the `/v1/models` route exists, it just wants a valid
>    API key. Unclear yet whether this is related to the 404 or a separate issue.
>
> **Immediate next step**: run
> `curl -H "Authorization: Bearer local-vllm-key" http://localhost:8000/v1/models`
> and see whether it returns the model (auth is fine, 404 is a real separate routing
> bug to dig into) or still errors (server was started with a different
> `VLLM_API_KEY`, needs to be found and passed to the test script via `--api-key`).
> Once the server responds correctly to both endpoints, redo the clean
> `idle_recovery_test.py` sweep (5/10/15/20/35/45/60s) with nothing else running
> against the server, to get uncontaminated data on whether the idle-recovery TTFT
> penalty is real and how it scales.
>
> **Still pending after that**: finish analyzing the 4-bit vs 8-bit comparison sweep
> (`analyze.py` on the existing CSVs), write the memory footprint comparison (8-bit
> numbers already captured from server startup logs: model_memory=7.70GB,
> metal_limit=12.71GB, kv_budget=3.28GB — need equivalent 4-bit numbers), write Task
> 7's tradeoffs document, and log all of this session's Task 7 work to
> `PHASE1_LOG.md` (not yet updated with any of it — this is a hard requirement per the
> log file's own standing instruction).
>
> Please don't ask me multiple clarifying questions before starting — just pick up at
> the `curl` auth check above.

---

## 6. How this differs day-to-day from Cowork (things to expect)

- **No "connect a folder" step.** Claude Code already has access to everything under
  wherever you launched it (`~/Documents/vllm-apple-silicon-serving` and, by
  extension, anywhere else on your Mac if you point it there).
- **It can actually run your server, curl it, and read the live logs** — this is the
  entire reason we moved.
- **Git works properly here.** No more sandboxed permission errors or stale lock
  files — Claude Code runs git as your own user in your own repo.
- **Session memory resets between terminal sessions** the same way Cowork chats do —
  if you close the terminal and come back later, paste a fresh summary (or just say
  "read PHASE1_LOG.md and catch yourself up") rather than assuming it remembers.
- **Slash commands**: `/exit` to leave, `/clear` to reset context mid-session if it
  gets long, `/help` for the full list.

## 7. Coming back to this doc later

If you forget any of this: the short version is `cd` into the project folder, run
`claude`, and either paste a fresh status prompt or ask it to read `PHASE1_LOG.md`
first. Everything else (install, login) is one-time setup and won't need repeating
unless you're on a new machine.
