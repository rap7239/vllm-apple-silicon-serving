#!/usr/bin/env bash
set -euo pipefail

last=""
while true; do
  now=$(date +%H:%M:%S)
  running=$(curl -s http://127.0.0.1:8000/metrics | grep -E "^vllm:num_requests_running\{" | awk '{print $NF}')
  waiting=$(curl -s http://127.0.0.1:8000/metrics | grep -E "^vllm:num_requests_waiting\{" | awk '{print $NF}')
  cur="Running:${running%.*} Waiting:${waiting%.*}"
  if [[ "$cur" != "$last" ]]; then
    echo "$now  $cur"
    last="$cur"
  fi
  sleep 0.2
done
