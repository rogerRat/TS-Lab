#!/bin/bash
probe_model="$1"; probe_python="$2"
cd /root/TS3
export PYTHONPATH=/root/TS3/src
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_ENDPOINT=https://hf-mirror.com
LOG=/root/autodl-tmp/TS3/outputs/probe-20260914/probe-expansion-v1-${probe_model}.log
{
for task in fev_epf_np fev_proenfo_gfc14 fev_solar_with_weather_1h time_darwin_water_15t time_ne_china_wind_h time_port_activity_d time_azure_i_cpu_5t; do
  "$probe_python" -u -m rags.experiments.full_pool_exploration --config "configs/rags/probe-expansion-v1/${probe_model}/${task}.json" --out "/root/autodl-tmp/TS3/outputs/probe-20260914/probe-expansion-v1/${probe_model}/${task}"
  code=$?
  printf 'MODEL=%s TASK=%s EXIT=%s\n' "$probe_model" "$task" "$code"
  if [ "$code" -ne 0 ]; then exit "$code"; fi
done
printf 'BATCH_EXIT=0\n'
} >> "$LOG" 2>&1
