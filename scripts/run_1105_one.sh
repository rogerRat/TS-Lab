#!/bin/bash
# usage: run_1105_one.sh <model_suffix> <run_dir> [config_prefix=1105]
set -e
PREFIX="${3:-1105}"
export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH=/root/TS3/src
export TS3_GRADIENT_CACHE_ROOT=/root/autodl-tmp/TS3/cache/grad-pool
cd /root/TS3
exec /root/miniconda3/envs/ts3/bin/python -m scripts.run_1105_budget_cut \
  --config "/root/TS3/configs/rags/exp_1105_budget_cut/${PREFIX}_$1.json" --run "$2"
