#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/TS-Lab
export PYTHONPATH=src:third_parties/fev/src:third_parties/TIME/src:.
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
python_bin=/root/autodl-tmp/rage/envs/rage-timerxl/bin/python
tasks=(fev-09 time-01 time-02 time-03 time-04)
batch=timerXL-ours-rage101-01-win-rr
root=outputs/train/v1-crb-redo/$batch
log() { printf '%s-[queue_v1_crb_redo_win]-[queue]-[%s]\n' "$(date -Is)" "$*"; }
child=
cleanup() { if [[ -n "$child" ]]; then kill -INT "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
log "config family=ours-rage101 tasks=${tasks[*]} budget=266631 score_dtype=bfloat16"
for task in "${tasks[@]}"; do
  if [[ -f "${root}/${task}/.complete" ]]; then log "skip complete ${batch} ${task}"; continue; fi
  log "select start ${batch} ${task}"
  "$python_bin" -u scripts/exp/v1-crb-redo/rr_select_win.py --family ours-rage101 --task "$task" & child=$!; wait "$child"; child=
  log "train-eval start ${batch} ${task}"
  "$python_bin" -u scripts/exp/v1-crb-redo/rr_train_eval.py --family ours-rage101 --granularity win --task "$task" & child=$!; wait "$child"; child=
  test -f "${root}/${task}/.complete"
  log "complete ${batch} ${task}"
done
for task in "${tasks[@]}"; do test -f "${root}/${task}/.complete"; done
touch "$root/.complete"
log "batch-complete ${batch} tasks=${tasks[*]}"
