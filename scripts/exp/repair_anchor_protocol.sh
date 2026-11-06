#!/usr/bin/env bash
set -euo pipefail
# Fixed existing development pool; separate BF16 output, no live-cache writes.
cd /root/TS-Lab
export PYTHONPATH=src OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
run_root=/root/autodl-tmp/rage/outputs/anchor-budget-20260914
calibration_out="$run_root/epsilon-bf16-dev-p10"
/root/autodl-tmp/conda/envs/rage-timemoe/bin/python -u -m rage.experiments.full_pool_calibration \
  --student timemoe-200m \
  --inputs /root/autodl-tmp/rage/outputs/full-pool-exp-0914/epsilon-inputs \
  --out "$calibration_out" \
  --model-path /root/autodl-tmp/hf-cache/hub/models--Maple728--TimeMoE-200M/snapshots/794591bfeb1225fdf742cec0f4c71f20c3f3b87e
CUDA_VISIBLE_DEVICES= /root/autodl-tmp/rage/envs/rage/bin/python -u -m rage.experiments.anchor_budget_linear \
  --run-root "$run_root" --calibration "$calibration_out/epsilon.json" \
  --out "$run_root/linear-p10-rank0"
