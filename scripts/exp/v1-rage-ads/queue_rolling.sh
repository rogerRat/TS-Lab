#!/usr/bin/env bash
set -euo pipefail
cd /root/TS-Lab
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rage-timerxl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/root/TS-Lab/src:/root/TS-Lab/third_parties/fev/src:/root/TS-Lab/third_parties/TIME/src:/root/TS-Lab:${PYTHONPATH:-}"
if [[ -f outputs/selection/v1-rage-ads/rolling/.complete ]]; then
    printf "%s-[queue_rolling]-[selection]-skip complete marker present\n" "$(date '+%Y-%m-%d %H:%M:%S')"
else
    python -u scripts/exp/v1-rage-ads/run_selection.py --task all
fi
if [[ -f outputs/selection/v1-rage-ads/rolling/audit/.complete ]]; then
    printf "%s-[queue_rolling]-[audit]-skip complete marker present\n" "$(date '+%Y-%m-%d %H:%M:%S')"
else
    printf "%s-[queue_rolling]-[audit]-workers=8 sample_count=4000 window_batch=32 vjp_batch=32\n" "$(date '+%Y-%m-%d %H:%M:%S')"
    python -u scripts/exp/v1-rage-ads/run_parallel_audit.py --workers 8 --sample-count 4000 --seed 42 --gradient-window-batch 32 --vjp-batch-size 32
fi
if [[ -f outputs/train/v1-rage-ads/rolling/.complete && -f outputs/eval/v1-rage-ads/rolling/.complete ]]; then
    printf "%s-[queue_rolling]-[train-eval]-skip complete markers present\n" "$(date '+%Y-%m-%d %H:%M:%S')"
else
    printf "%s-[queue_rolling]-[train-eval]-workers=12 cells=32\n" "$(date '+%Y-%m-%d %H:%M:%S')"
    python -u scripts/exp/v1-rage-ads/run_parallel_train_eval.py --workers 12
fi
python -u scripts/exp/v1-rage-ads/summarize_results.py
