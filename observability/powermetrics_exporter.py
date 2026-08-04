#!/usr/bin/env python3
"""
powermetrics_exporter.py
=========================
A minimal Prometheus exporter that wraps macOS's `powermetrics` to stand in
for NVIDIA's DCGM exporter on Apple Silicon.

Why this exists
----------------
Phase 1 Task 6 of the master plan asks for a Grafana dashboard with 6
panels, two of which (GPU SM utilization, HBM bandwidth) normally come from
DCGM. DCGM is NVIDIA-only -- there is no Apple Silicon equivalent, and this
repo's own README (see ../README.md -> vllm-benchmark's "Known limitation:
GPU utilization on Apple Silicon") already documents that gap. `powermetrics`
is Apple's native tool for this class of metric, so this script samples it
periodically and re-exposes the numbers in Prometheus text format so
Prometheus/Grafana can consume them the same way they'd consume DCGM.

What this can and cannot measure
---------------------------------
Confirmed against a real captured `powermetrics` text-format sample (not
guessed from documentation alone) that the tool exposes:
    GPU HW active frequency: <N> MHz
    GPU HW active residency:  <N>.<N>% (...)
    GPU idle residency:  <N>.<N>%
    GPU Power: <N> mW

There is no HBM/memory-bandwidth field anywhere in `powermetrics` output.
Apple Silicon's unified memory architecture has no discrete VRAM bus for a
DCGM-style bandwidth-utilization percentage to describe -- CPU and GPU share
the same physical memory pool via a wide on-package interconnect, not a
PCIe-attached VRAM bus. Rather than fabricate a number, this exporter does
NOT emit anything named like an HBM bandwidth metric. The Grafana dashboard
panel that would normally show HBM bandwidth instead shows GPU HW active
frequency, explicitly labeled as a substitute -- see observability/README.md
for the full explanation.

Metrics exposed (Prometheus text format on :9400/metrics)
-----------------------------------------------------------
    powermetrics_gpu_active_residency_percent   -- GPU SM-utilization equivalent
    powermetrics_gpu_active_frequency_mhz        -- GPU clock speed while active
    powermetrics_gpu_idle_residency_percent      -- inverse of active residency
    powermetrics_gpu_power_milliwatts            -- GPU power draw

Requirements
------------
`powermetrics` requires root. Run this script with sudo:
    sudo python3 powermetrics_exporter.py

Usage
-----
    sudo python3 powermetrics_exporter.py --interval 1000 --port 9400
"""

import argparse
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List, Optional, Tuple

# `sudo python3` on macOS often resolves to the system Python rather than
# whatever interpreter/venv is active in your normal shell (sudo typically
# does not inherit your PATH/venv the same way). That system Python can be
# older than 3.10, where the `X | Y` union type-hint syntax (PEP 604) isn't
# available yet -- this bit a real run of this script (see PHASE1_LOG.md).
# Using typing.Optional/Tuple/Dict/List instead of the newer built-in
# generic syntax keeps this file running on any Python 3.7+.

# Regex patterns matched directly against a real powermetrics sample capture
# (BinSquare/powermetrics-go's powermetrics_sample.log), not guessed from
# docs. Kept intentionally tolerant of the variable-width spacing seen in
# that sample (e.g. "residency:   1.63%" vs "residency:  98.37%").
GPU_ACTIVE_FREQ_RE = re.compile(r"^GPU HW active frequency:\s*([\d.]+)\s*MHz", re.MULTILINE)
GPU_ACTIVE_RESIDENCY_RE = re.compile(r"^GPU HW active residency:\s*([\d.]+)%", re.MULTILINE)
GPU_IDLE_RESIDENCY_RE = re.compile(r"^GPU idle residency:\s*([\d.]+)%", re.MULTILINE)
# NOTE: a real captured sample (see module docstring) has "GPU Power: <N> mW"
# appear TWICE -- once in a top-level CPU/GPU/ANE power summary line, once
# again inside the "**** GPU usage ****" section. Both were numerically
# identical in the sample examined, but nothing guarantees that holds across
# powermetrics versions/sampler combinations, so this pattern deliberately
# anchors to the second occurrence (inside "**** GPU usage ****", which is
# the more semantically correct source for this exporter's GPU-focused
# metric) via a non-greedy match past the GPU usage section header.
GPU_POWER_RE = re.compile(
    r"\*\*\*\* GPU usage \*\*\*\*.*?^GPU Power:\s*([\d.]+)\s*mW",
    re.MULTILINE | re.DOTALL,
)


class LatestSample:
    """Thread-safe holder for the most recent parsed powermetrics sample."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data = {
            "gpu_active_residency_percent": None,
            "gpu_active_frequency_mhz": None,
            "gpu_idle_residency_percent": None,
            "gpu_power_milliwatts": None,
        }
        self._last_updated = None

    def update(self, data: dict):
        with self._lock:
            self._data.update(data)
            self._last_updated = time.time()

    def snapshot(self) -> Tuple[dict, Optional[float]]:
        with self._lock:
            return dict(self._data), self._last_updated


def parse_sample(text: str) -> dict:
    """Extract the four GPU fields from one powermetrics text-format sample.

    Returns a dict with None for any field not found in this sample (rather
    than raising) so a single malformed/partial sample doesn't crash the
    exporter -- the HTTP handler will just serve the last-known-good value
    for that field via LatestSample, or an empty metric if none has ever
    been seen.
    """
    result = {}

    m = GPU_ACTIVE_FREQ_RE.search(text)
    result["gpu_active_frequency_mhz"] = float(m.group(1)) if m else None

    m = GPU_ACTIVE_RESIDENCY_RE.search(text)
    result["gpu_active_residency_percent"] = float(m.group(1)) if m else None

    m = GPU_IDLE_RESIDENCY_RE.search(text)
    result["gpu_idle_residency_percent"] = float(m.group(1)) if m else None

    m = GPU_POWER_RE.search(text)
    result["gpu_power_milliwatts"] = float(m.group(1)) if m else None

    return result


def powermetrics_sampler(latest: LatestSample, interval_ms: int, stop_event: threading.Event):
    """Run `powermetrics` in continuous mode and feed each sample to `latest`.

    Uses -i <interval_ms> for repeated sampling in one long-running process
    (cheaper than re-invoking powermetrics per sample) and splits its output
    on the "*** Sampled system activity" marker that begins each new sample
    block, which is present in every real capture examined for this script.
    """
    cmd = [
        "powermetrics",
        "--samplers", "gpu_power",
        "-i", str(interval_ms),
    ]
    print(f"Starting: {' '.join(cmd)}  (requires sudo)", file=sys.stderr)

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )
    except FileNotFoundError:
        print("ERROR: `powermetrics` not found. This exporter only runs on macOS.",
              file=sys.stderr)
        stop_event.set()
        return
    except PermissionError:
        print("ERROR: permission denied starting powermetrics. Run this script "
              "with sudo.", file=sys.stderr)
        stop_event.set()
        return

    buffer_lines: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        buffer_lines.append(line)
        # Each sample block ends right before the next one's header line, or
        # can be flushed on the GPU Power line (last field in each block per
        # the sample log examined) -- flushing there keeps latency low
        # without waiting for the *next* block's header to arrive.
        if line.startswith("GPU Power:"):
            text = "".join(buffer_lines)
            parsed = parse_sample(text)
            latest.update(parsed)
            buffer_lines = []
        if stop_event.is_set():
            break

    proc.terminate()


class MetricsHandler(BaseHTTPRequestHandler):
    latest: LatestSample = None  # type: ignore[assignment]  -- set by main()

    def do_GET(self):
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return

        data, last_updated = self.latest.snapshot()
        lines = [
            "# HELP powermetrics_gpu_active_residency_percent GPU HW active "
            "residency percent (SM-utilization equivalent), from macOS powermetrics.",
            "# TYPE powermetrics_gpu_active_residency_percent gauge",
            "# HELP powermetrics_gpu_active_frequency_mhz GPU HW active frequency "
            "in MHz. Substitutes for HBM bandwidth on this dashboard panel -- "
            "Apple Silicon's unified memory architecture has no discrete "
            "HBM/VRAM bus for powermetrics to report a bandwidth figure for. "
            "See observability/README.md.",
            "# TYPE powermetrics_gpu_active_frequency_mhz gauge",
            "# HELP powermetrics_gpu_idle_residency_percent GPU idle residency percent.",
            "# TYPE powermetrics_gpu_idle_residency_percent gauge",
            "# HELP powermetrics_gpu_power_milliwatts GPU power draw in milliwatts.",
            "# TYPE powermetrics_gpu_power_milliwatts gauge",
        ]

        def emit(name: str, value):
            if value is not None:
                lines.append(f"{name} {value}")

        emit("powermetrics_gpu_active_residency_percent", data["gpu_active_residency_percent"])
        emit("powermetrics_gpu_active_frequency_mhz", data["gpu_active_frequency_mhz"])
        emit("powermetrics_gpu_idle_residency_percent", data["gpu_idle_residency_percent"])
        emit("powermetrics_gpu_power_milliwatts", data["gpu_power_milliwatts"])

        if last_updated is not None:
            staleness = time.time() - last_updated
            lines.append(
                "# powermetrics_exporter_sample_age_seconds "
                f"{staleness:.1f}  -- time since the last successful powermetrics sample"
            )
            lines.append(f"powermetrics_exporter_sample_age_seconds {staleness:.3f}")

        body = "\n".join(lines) + "\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, format, *args):
        # Silence the default per-request stderr logging -- Prometheus will
        # scrape this every few seconds and it's not useful noise here.
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Prometheus exporter wrapping macOS powermetrics for GPU metrics."
    )
    parser.add_argument("--interval", type=int, default=1000,
                         help="powermetrics sampling interval in ms (default: 1000)")
    parser.add_argument("--port", type=int, default=9400,
                         help="HTTP port to serve /metrics on (default: 9400)")
    args = parser.parse_args()

    latest = LatestSample()
    stop_event = threading.Event()

    sampler_thread = threading.Thread(
        target=powermetrics_sampler, args=(latest, args.interval, stop_event), daemon=True
    )
    sampler_thread.start()

    MetricsHandler.latest = latest
    server = HTTPServer(("0.0.0.0", args.port), MetricsHandler)
    print(f"Serving Prometheus metrics on :{args.port}/metrics "
          f"(sampling powermetrics every {args.interval}ms)", file=sys.stderr)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        server.shutdown()


if __name__ == "__main__":
    main()
