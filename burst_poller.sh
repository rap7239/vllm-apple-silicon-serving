#!/usr/bin/env bash
# Deliberately no -e / pipefail: this loop must survive transient scrape
# failures (a grep with zero matches exits non-zero) rather than die
# silently under load, which is what killed the first version of this
# script during the 50-request burst.
set -u

OUT="burst_timeseries.csv"
echo "timestamp,running,waiting,kv_cache_used_pct,gpu_freq_mhz" > "$OUT"

last=""
while true; do
  now=$(date +%H:%M:%S)
  metrics=$(curl -s -m 2 http://127.0.0.1:8000/metrics 2>/dev/null || true)
  running=$(echo "$metrics" | grep -E "^vllm:num_requests_running\{" | awk '{print $NF}')
  waiting=$(echo "$metrics" | grep -E "^vllm:num_requests_waiting\{" | awk '{print $NF}')
  kv=$(echo "$metrics" | grep -E "^vllm:kv_cache_usage_perc\{" | awk '{print $NF}')
  freq=$(curl -s -m 2 http://127.0.0.1:9400/metrics 2>/dev/null | grep -E "^powermetrics_gpu_active_frequency_mhz" | awk '{print $NF}')

  running="${running:-n/a}"
  waiting="${waiting:-n/a}"
  kv="${kv:-n/a}"
  freq="${freq:-n/a}"

  echo "${now},${running},${waiting},${kv},${freq}" >> "$OUT"

  cur="R:${running%.*} W:${waiting%.*} KV:${kv}"
  if [[ "$cur" != "$last" ]]; then
    echo "$now  Running:${running%.*} Waiting:${waiting%.*} KV_used:${kv} GPU_freq:${freq}MHz"
    last="$cur"
  fi
  sleep 1
done
