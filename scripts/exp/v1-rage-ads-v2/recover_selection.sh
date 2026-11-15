#!/usr/bin/env bash
set -euo pipefail
cd /root/TS-Lab
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rage-timerxl
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/root/TS-Lab/src:/root/TS-Lab/third_parties/fev/src:/root/TS-Lab/third_parties/TIME/src:/root/TS-Lab:${PYTHONPATH:-}"
mapfile -t all_tasks < <(python scripts/exp/v1-rage-ads-v2/experiment_config.py --print-tasks)
if [[ "${1:-}" != "--resume" ]]; then
    while IFS= read -r line; do
        case "$line" in
            *"-[failed]-"*) break ;;
            *"-[complete]-"*) exit 0 ;;
        esac
    done < <(tail -n0 -F logs/v1-rage-ads-v2/selection-queue.log)
fi
tasks=()
for task in "${all_tasks[@]}"; do
    if [[ ! -f "outputs/selection/v1-rage-ads-v2/${task}/.complete" ]]; then tasks+=("$task"); fi
done
printf "%s-[v2-recovery]-[start]-[missing=%s workers=2]\n" "$(date "+%Y-%m-%d %H:%M:%S %Z")" "${tasks[*]:-none}" | tee -a logs/v1-rage-ads-v2/selection-queue.log
if [[ "${#tasks[@]}" -gt 0 ]]; then
    pids=()
    failed=0
    for task in "${tasks[@]}"; do
        python -u scripts/exp/v1-rage-ads-v2/run_selection.py --task "$task" \
            > "logs/v1-rage-ads-v2/selection-${task}.log" 2>&1 &
        pids+=("$!")
        if [[ "${#pids[@]}" -ge 2 ]]; then
            for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
            pids=()
        fi
    done
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    if [[ "$failed" -ne 0 ]]; then
        printf "%s-[v2-recovery]-[failed]-[retry workers exited unsuccessfully]\n" "$(date "+%Y-%m-%d %H:%M:%S %Z")" | tee -a logs/v1-rage-ads-v2/selection-queue.log
        exit 1
    fi
fi
for task in "${all_tasks[@]}"; do
    test -f "outputs/selection/v1-rage-ads-v2/${task}/.complete"
done
printf "complete\n" > outputs/selection/v1-rage-ads-v2/.complete
printf "%s-[v2-recovery]-[complete]-[all_tasks=%s]\n" "$(date "+%Y-%m-%d %H:%M:%S %Z")" "${#all_tasks[@]}" | tee -a logs/v1-rage-ads-v2/selection-queue.log
if tmux has-session -t v1_rage_ads_v2_train 2>/dev/null; then
    tmux kill-session -t v1_rage_ads_v2_train
fi
tmux new-session -d -s v1_rage_ads_v2_train "cd /root/TS-Lab && bash scripts/exp/v1-rage-ads-v2/run_training_after_selection.sh"
