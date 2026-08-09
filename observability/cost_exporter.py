#!/usr/bin/env python3
"""
cost_exporter.py
================
A minimal Prometheus exporter that re-serves the cost model output
(Phase 1 item #9) from `run_bench.py`'s `{variant}_cost.json` summary files
so Grafana can chart it, the same way powermetrics_exporter.py bridges
macOS's `powermetrics` into the same stack.

Why this exists
----------------
Prometheus/Grafana scrape *continuously running* HTTP endpoints. run_bench.py
is a one-shot CLI script -- it runs a benchmark, writes
runs/{variant}_cost.json, and exits. There is nothing to scrape while it's
not running. This exporter bridges that gap: on every scrape, it finds the
most recently modified *_cost.json in the vllm-benchmark repo's runs/
directory and re-exposes its fields as Prometheus gauges, so the dashboard
always shows the latest completed run's cost figures, updating each time a
new benchmark run finishes -- no sudo required, no background sampling
thread needed (reading one small JSON file is cheap enough to do inline on
each scrape, unlike powermetrics_exporter.py's continuous subprocess).

Metrics exposed (Prometheus text format on :9500/metrics by default)
----------------------------------------------------------------------
    run_bench_cost_per_1k_tokens_usd   -- headline cost efficiency figure
    run_bench_tokens_per_dollar        -- inverse framing of the same number
    run_bench_avg_gpu_power_watts      -- what electricity term used
    run_bench_total_run_cost_usd       -- total $ for the whole run
    run_bench_hourly_hardware_usd      -- amortized hardware $/hour
    run_bench_hourly_electricity_usd   -- electricity $/hour
    run_bench_run_duration_seconds     -- wall-clock duration of that run
    run_bench_total_output_tokens      -- tokens produced in that run
    run_bench_cost_file_age_seconds    -- staleness of the underlying file

All metrics carry a `variant` label (the run_bench.py --variant that
produced them) so Grafana can distinguish which run is being shown.

Usage
-----
    python3 cost_exporter.py --runs-dir ~/Documents/vllm-benchmark/runs --port 9500
"""

import argparse
import glob
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional


def find_latest_cost_file(runs_dir: str) -> Optional[str]:
    """Return the most recently modified *_cost.json in runs_dir, or None."""
    candidates = glob.glob(os.path.join(runs_dir, "*_cost.json"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


class MetricsHandler(BaseHTTPRequestHandler):
    runs_dir: str = ""  # set by main()

    def do_GET(self):
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return

        lines = [
            "# HELP run_bench_cost_per_1k_tokens_usd Cost per 1,000 output tokens, "
            "from the most recently completed run_bench.py cost-tracked run.",
            "# TYPE run_bench_cost_per_1k_tokens_usd gauge",
            "# HELP run_bench_tokens_per_dollar Inverse framing of the same cost figure.",
            "# TYPE run_bench_tokens_per_dollar gauge",
            "# HELP run_bench_avg_gpu_power_watts Average GPU power sampled across "
            "that run's whole wall-clock duration (including idle/queued time).",
            "# TYPE run_bench_avg_gpu_power_watts gauge",
            "# HELP run_bench_total_run_cost_usd Total dollar cost for that run.",
            "# TYPE run_bench_total_run_cost_usd gauge",
            "# HELP run_bench_hourly_hardware_usd Amortized hardware cost, $/hour.",
            "# TYPE run_bench_hourly_hardware_usd gauge",
            "# HELP run_bench_hourly_electricity_usd Electricity cost, $/hour.",
            "# TYPE run_bench_hourly_electricity_usd gauge",
            "# HELP run_bench_run_duration_seconds Wall-clock duration of that run.",
            "# TYPE run_bench_run_duration_seconds gauge",
            "# HELP run_bench_total_output_tokens Total output tokens produced in that run.",
            "# TYPE run_bench_total_output_tokens gauge",
            "# HELP run_bench_cost_file_age_seconds Time since the underlying "
            "cost.json file was last written -- how stale this reading is.",
            "# TYPE run_bench_cost_file_age_seconds gauge",
        ]

        path = find_latest_cost_file(self.runs_dir)
        if path is None:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.end_headers()
            self.wfile.write(("\n".join(lines) + "\n").encode("utf-8"))
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            # Cost file mid-write or corrupted -- serve headers only rather
            # than crash the exporter over one bad scrape.
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.end_headers()
            self.wfile.write(("\n".join(lines) + "\n").encode("utf-8"))
            return

        variant = os.path.basename(path).removesuffix("_cost.json")
        label = f'{{variant="{variant}"}}'

        def emit(metric: str, value):
            if value is not None:
                lines.append(f"{metric}{label} {value}")

        emit("run_bench_cost_per_1k_tokens_usd", data.get("cost_per_1k_tokens_usd"))
        emit("run_bench_tokens_per_dollar", data.get("tokens_per_dollar"))
        emit("run_bench_avg_gpu_power_watts", data.get("avg_gpu_power_watts"))
        emit("run_bench_total_run_cost_usd", data.get("total_run_cost_usd"))
        emit("run_bench_hourly_hardware_usd", data.get("hourly_hardware_usd"))
        emit("run_bench_hourly_electricity_usd", data.get("hourly_electricity_usd"))
        emit("run_bench_run_duration_seconds", data.get("run_duration_s"))
        emit("run_bench_total_output_tokens", data.get("total_output_tokens"))
        emit("run_bench_cost_file_age_seconds", time.time() - os.path.getmtime(path))

        body = "\n".join(lines) + "\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, format, *args):
        pass  # silence per-scrape stderr noise, same as powermetrics_exporter.py


def main():
    parser = argparse.ArgumentParser(
        description="Prometheus exporter re-serving run_bench.py's cost-model output."
    )
    parser.add_argument("--runs-dir", required=True,
                         help="Path to vllm-benchmark's runs/ directory")
    parser.add_argument("--port", type=int, default=9500,
                         help="HTTP port to serve /metrics on (default: 9500)")
    args = parser.parse_args()

    runs_dir = os.path.expanduser(args.runs_dir)
    if not os.path.isdir(runs_dir):
        print(f"ERROR: {runs_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    MetricsHandler.runs_dir = runs_dir
    server = HTTPServer(("0.0.0.0", args.port), MetricsHandler)
    print(f"Serving Prometheus metrics on :{args.port}/metrics "
          f"(reading latest *_cost.json from {runs_dir})", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
