#!/usr/bin/env bash
set -euo pipefail
cd /root/TS-Lab
export PYTHONPATH=src OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
run_root=/root/autodl-tmp/rage/outputs/anchor-budget-20260914
while [[ ! -f "$run_root/epsilon-bf16-dev-p10/epsilon.json" ]]; do
  if ! tmux has-session -t anchor-protocol-repair 2>/dev/null; then
    echo 'P10 calibration stopped without an output; inspect protocol-repair.log' >&2
    exit 1
  fi
  sleep 15
done
CUDA_VISIBLE_DEVICES= /root/autodl-tmp/rage/envs/rage/bin/python -u -m rage.experiments.anchor_budget_plot \
  --run-root "$run_root" --calibration "$run_root/epsilon-bf16-dev-p10/epsilon.json" \
  --out "$run_root/linear-p10-figures"
