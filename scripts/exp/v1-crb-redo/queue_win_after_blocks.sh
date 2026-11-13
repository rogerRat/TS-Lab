#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/TS-Lab
export PYTHONPATH=src:third_parties/fev/src:third_parties/TIME/src:.
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
python_bin=/root/autodl-tmp/rage/envs/rage-timerxl/bin/python
tasks=(fev-09 time-01 time-02 time-03 time-04 time-06 time-07 time-08)
log() { printf '%s-[queue_v1_crb_redo_win]-[queue]-[%s]\n' "$(date -Is)" "$*"; }
child=
cleanup() { if [[ -n "$child" ]]; then kill -INT "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for batch in timerXL-base-uni-01-traj-rr timerXL-base-uni-01-block-rr timerXL-ours-rage101-01-traj-rr timerXL-ours-rage101-01-block-rr; do
  marker="outputs/train/v1-crb-redo/${batch}/.complete"
  while [[ ! -f "$marker" ]]; do
    if ! tmux has-session -t v1-crb-redo-rr-queue 2>/dev/null; then
      log "upstream queue exited before ${batch} completed"
      exit 1
    fi
    sleep 30
  done
  log "upstream complete ${batch}"
done
for family in base-uni ours-rage101; do
  batch="timerXL-${family}-01-win-rr"
  root="outputs/train/v1-crb-redo/${batch}"
  for task in "${tasks[@]}"; do
    if [[ -f "${root}/${task}/.complete" ]]; then log "skip complete ${batch} ${task}"; continue; fi
    log "select start ${batch} ${task}"
    "$python_bin" -u scripts/exp/v1-crb-redo/rr_select_win.py --family "$family" --task "$task" & child=$!; wait "$child"; child=
    log "train-eval start ${batch} ${task}"
    "$python_bin" -u scripts/exp/v1-crb-redo/rr_train_eval.py --family "$family" --granularity win --task "$task" & child=$!; wait "$child"; child=
    test -f "${root}/${task}/.complete"
    log "complete ${batch} ${task}"
  done
  for task in "${tasks[@]}"; do test -f "${root}/${task}/.complete"; done
  touch "$root/.complete"
  log "batch-complete ${batch}"
done
log "all six RR batches complete"
