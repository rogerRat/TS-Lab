#!/usr/bin/env bash
set -euo pipefail
cd /root/TS-Lab
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rage-timerxl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/root/TS-Lab/src:/root/TS-Lab/third_parties/fev/src:/root/TS-Lab/third_parties/TIME/src:/root/TS-Lab:${PYTHONPATH:-}"
mapfile -t tasks < <(python scripts/exp/v1-rage-ads-v2/experiment_config.py --print-tasks)
mkdir -p logs/v1-rage-ads-v2 outputs/selection/v1-rage-ads-v2
started=$(date "+%Y-%m-%d %H:%M:%S %Z")
printf "%s-[v2-queue]-[start]-[tasks=%s workers=%s source_pool=src-256]\n" "$started" "${#tasks[@]}" "${#tasks[@]}" | tee logs/v1-rage-ads-v2/selection-queue.log
pids=()
stop_children() {
    trap - INT TERM
    for pid in "${pids[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap stop_children INT TERM
failed=0
for task in "${tasks[@]}"; do
    python -u scripts/exp/v1-rage-ads-v2/run_selection.py --task "$task" \
        > "logs/v1-rage-ads-v2/selection-${task}.log" 2>&1 &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
done
if [[ "$failed" -ne 0 ]]; then
    printf "%s-[v2-queue]-[failed]-[selection stage had task failures]\n" "$(date "+%Y-%m-%d %H:%M:%S %Z")" | tee -a logs/v1-rage-ads-v2/selection-queue.log
    exit 1
fi
for task in "${tasks[@]}"; do
    test -f "outputs/selection/v1-rage-ads-v2/${task}/.complete"
done
printf "complete\n" > outputs/selection/v1-rage-ads-v2/.complete
printf "%s-[v2-queue]-[complete]-[elapsed=%s output=outputs/selection/v1-rage-ads-v2]\n" \
    "$(date "+%Y-%m-%d %H:%M:%S %Z")" "$started" | tee -a logs/v1-rage-ads-v2/selection-queue.log
