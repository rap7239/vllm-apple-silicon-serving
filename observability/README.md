# observability

Prometheus + Grafana stack for Phase 1 Task 6 of the master plan: a 6-panel
Grafana dashboard covering TTFT heatmap, tokens/sec vs concurrency, p99
trend, GPU SM utilization, HBM bandwidth, and KV cache fill level.

## The DCGM gap, and how this stack handles it

Two of those six panels — GPU SM utilization and HBM bandwidth — normally
come from NVIDIA's DCGM exporter. DCGM is NVIDIA-only. There is no Apple
Silicon equivalent (already documented in
[`../../vllm-benchmark/README.md`](../../vllm-benchmark/README.md#known-limitation-gpu-utilization-on-apple-silicon)
from earlier in this project). This stack closes that gap with a purpose-built
exporter (`powermetrics_exporter.py`) that wraps macOS's native `powermetrics`
tool and re-exposes what it can measure in Prometheus format.

**What `powermetrics` can and cannot measure**, confirmed against a real
captured `powermetrics` text-format sample (not assumed from documentation
alone — see the script's own docstring for the source):

| Dashboard panel | Real metric used | Notes |
|---|---|---|
| GPU SM Utilization | `powermetrics_gpu_active_residency_percent` (from `GPU HW active residency`) | Genuine analog to DCGM's SM-utilization percentage |
| HBM Bandwidth | `powermetrics_gpu_active_frequency_mhz` (from `GPU HW active frequency`) | **Not actually bandwidth.** `powermetrics` has no memory-bandwidth field at all — Apple Silicon's unified memory architecture has no discrete VRAM bus for a DCGM-style bandwidth percentage to describe. GPU active frequency was chosen as the closest available GPU-performance substitute, labeled explicitly in the dashboard panel description rather than silently swapped in. |

This is a deliberate, documented substitution, not a workaround pretending
to be the real thing — worth being able to explain clearly in an interview
if asked how GPU observability differs across hardware platforms.

## Architecture

```
┌─────────────────┐     scrape :8000/metrics      ┌──────────────┐
│  vLLM (native,   │ ─────────────────────────────▶│              │
│  scripts/serve.sh│                                │  Prometheus  │──▶ Grafana
│  in parent repo) │                                │  (Docker)    │   (Docker)
└──────────────────┘                                │              │
┌──────────────────┐    scrape :9400/metrics        │              │
│ powermetrics_     │────────────────────────────────▶              │
│ exporter.py       │                                └──────────────┘
│ (native, sudo)    │
└──────────────────┘
```

vLLM and the powermetrics exporter both run **natively** on the Mac, not in
Docker — vLLM because it already runs that way via `scripts/serve.sh` in the
parent repo, and the exporter because `powermetrics` requires direct access
to macOS kernel performance counters and `sudo`, neither of which a
Dockerized process on macOS (which runs inside a Linux VM under Docker
Desktop) can reach. Only Prometheus and Grafana run in Docker.

## Quick start

**1. Start vLLM** (if not already running), from the parent repo:
```bash
cd ..
./scripts/serve.sh
```

**2. Start the powermetrics exporter** (requires sudo):
```bash
sudo python3 powermetrics_exporter.py
# Serves GPU metrics on http://localhost:9400/metrics
```

**3. Start Prometheus + Grafana**:
```bash
docker compose up -d
```

**4. Open Grafana**: http://localhost:3000 (login: `admin` / `admin` — local
lab defaults only, change before exposing this beyond localhost). The
"vLLM Saturation & GPU Dashboard (M4 Mac mini)" dashboard should already be
provisioned and visible — Grafana auto-loads anything in
`grafana/provisioning/dashboards/` on startup.

**5. Verify Prometheus is actually scraping both targets**: http://localhost:9090/targets
should show `vllm` and `powermetrics_exporter` both `UP`. If either shows
`DOWN`, see Troubleshooting below.

## What's verified vs. what needs your confirmation

Built from a sandboxed Linux environment with no direct access to macOS,
`powermetrics`, Docker, or a live vLLM server — everything below was
verified as precisely as that allowed, but the honest state is:

**Verified directly:**
- `powermetrics_exporter.py`'s regex parsing logic, tested against a real
  captured `powermetrics --samplers gpu_power` text sample (not a guess) —
  all four fields (active residency, active frequency, idle residency,
  power) extract correctly, including a stress test for the case where
  `GPU Power:` appears twice in one sample block with different values.
- All JSON/YAML config files parse as valid.
- `vllm:num_requests_running`, `vllm:num_requests_waiting`,
  `vllm:kv_cache_usage_perc` — these three were already curl-verified
  directly against this project's live vLLM server in an earlier session
  (see `PHASE1_LOG.md`, Task 3 entry).

**Corroborated via vLLM's official docs, not directly tested against this
project's server:**
- `vllm:time_to_first_token_seconds` (histogram) and
  `vllm:generation_tokens_total` (counter) — confirmed as real vLLM metric
  names via vLLM's official metrics documentation, but not curl-verified
  against this project's specific vLLM 0.26.x server the way the three
  metrics above were. **Run `curl -s http://localhost:8000/metrics | grep -E
  "time_to_first_token|generation_tokens_total"` once vLLM is serving, and
  if the exact names differ, update the two panel queries in
  `grafana/provisioning/dashboards/vllm-saturation-dashboard.json`
  (panels 1, 2, 3) accordingly.**

**Cannot be verified from this environment at all, needs your testing:**
- The actual `sudo powermetrics` subprocess invocation and its live
  streaming/buffering behavior in `powermetrics_exporter.py` — this only
  runs on macOS. The parsing logic is solid; whether the process management
  around it (buffering per sample block, handling `powermetrics` exiting
  unexpectedly, etc.) behaves correctly under real conditions needs a live
  run.
- Whether Docker Desktop's `host.docker.internal` resolves correctly on
  your specific Docker Desktop for Mac version (this is standard behavior,
  but worth confirming via the Prometheus targets page in step 5 above).
- Whether the two-year-old-by-comparison heatmap panel type renders
  correctly with this project's actual TTFT bucket distribution — heatmap
  panels are sensitive to having enough buckets with enough traffic to look
  meaningful; worth checking after some real load (e.g. re-running the
  `vllm-benchmark` saturation sweep from Task 4 while this dashboard is open).

## Troubleshooting

**Prometheus target `powermetrics_exporter` shows DOWN**: confirm the
exporter is actually running (`sudo python3 powermetrics_exporter.py`
should print "Serving Prometheus metrics on :9400/metrics" to stderr) and
that `curl localhost:9400/metrics` returns data directly on the host before
suspecting the Docker networking layer.

**Prometheus target `vllm` shows DOWN**: confirm vLLM is running and
`curl localhost:8000/metrics` returns data directly.

**Dashboard panels show "No data"**: check the Prometheus targets page
first (http://localhost:9090/targets) — if both targets are UP, the next
most likely cause is a metric name mismatch (see the verification table
above for which panels rely on unverified-against-this-project's-version
metric names).

## Repo structure

```
observability/
├── docker-compose.yml           # Prometheus + Grafana (Docker); vLLM and the
│                                 # exporter run natively, see Architecture above
├── prometheus.yml                # scrape config: vllm:8000, powermetrics_exporter:9400
├── powermetrics_exporter.py       # DCGM substitute for Apple Silicon (needs sudo)
├── grafana/provisioning/
│   ├── datasources/prometheus.yml
│   └── dashboards/
│       ├── dashboards.yml
│       └── vllm-saturation-dashboard.json   # the 6-panel dashboard
└── README.md
```
