#!/usr/bin/env bash
set -euo pipefail
cd /root/TS-Lab
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rage-timerxl
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/root/TS-Lab/src:/root/TS-Lab/third_parties/fev/src:/root/TS-Lab/third_parties/TIME/src:/root/TS-Lab:${PYTHONPATH:-}"
mkdir -p logs/v1-rage-ads-v2
exec > >(tee -a logs/v1-rage-ads-v2/ads-rev-pipeline.log) 2>&1
printf "%s-[ads-rev]-[selection-start]-[tasks=8 q0=32 q1=16 threshold_ge=0.1]
" "$(date "+%Y-%m-%d %H:%M:%S %Z")"
python -u scripts/exp/v1-rage-ads-v2/run_ads_rev_selection.py --task all
printf "%s-[ads-rev]-[train-eval-start]-[workers=12 tasks=8]
" "$(date "+%Y-%m-%d %H:%M:%S %Z")"
python -u scripts/exp/v1-rage-ads-v2/run_parallel_train_eval.py --workers 12 --arms ads-rev
printf "%s-[ads-rev]-[pipeline-complete]-[selection=8 train-eval=8]
" "$(date "+%Y-%m-%d %H:%M:%S %Z")"
