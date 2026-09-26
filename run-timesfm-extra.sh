#!/bin/bash
cd /root/TS3
export PYTHONPATH=/root/TS3/src
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_ENDPOINT=https://hf-mirror.com
LOG=/root/autodl-tmp/TS3/outputs/probe-20260914/timesfm-single-domain-v1.log
{
for task in time_finland_15t time_eweld_15t time_nem_demand_5t time_solar_h; do
  /root/autodl-tmp/conda/envs/ts3-timesfm/bin/python -u -m rags.experiments.full_pool_exploration --config "configs/rags/timesfm-single-domain-v1/${task}.json" --out "/root/autodl-tmp/TS3/outputs/probe-20260914/timesfm-single-domain-v1/${task}"
  code=$?
  printf 'TASK=%s EXIT=%s\n' "$task" "$code"
  if [ "$code" -ne 0 ]; then exit "$code"; fi
done
printf 'BATCH_EXIT=0\n'
} >> "$LOG" 2>&1
