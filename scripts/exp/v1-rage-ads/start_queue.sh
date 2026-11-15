#!/usr/bin/env bash
set -u
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
mkdir -p logs/v1-rage-ads
started_at=$(date "+%Y-%m-%d %H:%M:%S %Z")
printf "%s status=started session=v1-rage-ads-rolling\n" "$started_at" > logs/v1-rage-ads/queue.status
bash scripts/exp/v1-rage-ads/queue_rolling.sh
queue_status=$?
finished_at=$(date "+%Y-%m-%d %H:%M:%S %Z")
printf "%s status=%s session=v1-rage-ads-rolling\n" "$finished_at" "$queue_status" > logs/v1-rage-ads/queue.status
while true; do
    sleep 3600
done
