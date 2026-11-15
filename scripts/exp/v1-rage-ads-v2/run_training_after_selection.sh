#!/usr/bin/env bash
set -euo pipefail
cd /root/TS-Lab
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rage-timerxl
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/root/TS-Lab/src:/root/TS-Lab/third_parties/fev/src:/root/TS-Lab/third_parties/TIME/src:/root/TS-Lab:${PYTHONPATH:-}"
mkdir -p logs/v1-rage-ads-v2
printf "%s-[v2-pipeline]-[waiting]-[selection completion event]\n" "$(date "+%Y-%m-%d %H:%M:%S %Z")" | tee -a logs/v1-rage-ads-v2/pipeline.log
if [[ ! -f outputs/selection/v1-rage-ads-v2/.complete ]]; then
    while IFS= read -r line; do
        case "$line" in
            *"-[complete]-"*) break ;;
            *"-[failed]-"*) printf "%s-[v2-pipeline]-[recovery]-[selection retry is handling failed tasks]\n" "$(date "+%Y-%m-%d %H:%M:%S %Z")" | tee -a logs/v1-rage-ads-v2/pipeline.log ;;
        esac
    done < <(tail -n0 -F logs/v1-rage-ads-v2/selection-queue.log)
fi
cells=$(python scripts/exp/v1-rage-ads-v2/experiment_config.py --print-cells)
printf "%s-[v2-pipeline]-[train-eval-start]-[workers=12 cells=%s]\n" "$(date "+%Y-%m-%d %H:%M:%S %Z")" "$cells" | tee -a logs/v1-rage-ads-v2/pipeline.log
python -u scripts/exp/v1-rage-ads-v2/run_parallel_train_eval.py --workers 12
printf "%s-[v2-pipeline]-[complete]-[train and rolling evaluation finished]\n" "$(date "+%Y-%m-%d %H:%M:%S %Z")" | tee -a logs/v1-rage-ads-v2/pipeline.log
