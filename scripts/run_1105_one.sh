#!/bin/bash
# usage: run_1105_one.sh <model_suffix> <run_dir> [config_prefix=1105]
set -e
PREFIX="${3:-1105}"
export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH=/root/TS-Lab/src
export RAGE_GRADIENT_CACHE_ROOT=/root/autodl-tmp/rage/cache/grad-pool
cd /root/TS-Lab
exec /root/autodl-tmp/rage/envs/rage/bin/python -m scripts.run_1105_budget_cut \
  --config "/root/TS-Lab/configs/rage/exp_1105_budget_cut/${PREFIX}_$1.json" --run "$2"
