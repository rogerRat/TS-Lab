#!/usr/bin/env bash
set -euo pipefail
# Fixed existing development pool; separate BF16 output, no live-cache writes.
cd /root/TS3
export PYTHONPATH=src OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
run_root=/root/autodl-tmp/TS3/outputs/anchor-budget-20260914
calibration_out="$run_root/epsilon-bf16-dev-p10"
/root/autodl-tmp/conda/envs/ts3-timemoe/bin/python -u -m rags.experiments.full_pool_calibration \
  --student timemoe-200m \
  --inputs /root/autodl-tmp/TS3/outputs/full-pool-exp-0914/epsilon-inputs \
  --out "$calibration_out" \
  --model-path /root/autodl-tmp/hf-cache/hub/models--Maple728--TimeMoE-200M/snapshots/794591bfeb1225fdf742cec0f4c71f20c3f3b87e
CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/ts3/bin/python -u -m rags.experiments.anchor_budget_linear \
  --run-root "$run_root" --calibration "$calibration_out/epsilon.json" \
  --out "$run_root/linear-p10-rank0"
