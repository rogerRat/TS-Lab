#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/TS-Lab
export PYTHONPATH=src:third_parties/fev/src:third_parties/TIME/src:.
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
python_bin=/root/autodl-tmp/rage/envs/rage-timerxl/bin/python
tasks=(fev-09 time-01 time-02 time-03 time-04 time-06 time-07 time-08)
log() { printf '%s-[queue_v1_crb_redo_rr]-[queue]-[%s]\n' "$(date -Is)" "$*"; }
child=
cleanup() { if [[ -n "$child" ]]; then kill -INT "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for pair in 'base-uni traj' 'base-uni block' 'ours-rage101 traj' 'ours-rage101 block'; do
  read -r family granularity <<< "$pair"
  batch="timerXL-${family}-01-${granularity}-rr"
  root="outputs/train/v1-crb-redo/${batch}"
  for task in "${tasks[@]}"; do
    train="${root}/${task}"
    if [[ -f "${train}/.complete" ]]; then log "skip complete ${batch} ${task}"; continue; fi
    if [[ ! -f "outputs/selection/v1-crb-redo/${batch}/${task}/.complete" ]]; then
      log "select start ${batch} ${task}"
      "${python_bin}" -u "scripts/exp/v1-crb-redo/rr_select.py" --family "${family}" --granularity "${granularity}" --task "${task}" & child=$!; wait "${child}"; child=
    fi
    log "train-eval start ${batch} ${task}"
    "${python_bin}" -u "scripts/exp/v1-crb-redo/rr_train_eval.py" --family "${family}" --granularity "${granularity}" --task "${task}" & child=$!; wait "${child}"; child=
    test -f "${train}/.complete"
    log "complete ${batch} ${task}"
  done
  for task in "${tasks[@]}"; do test -f "${root}/${task}/.complete"; done
  touch "${root}/.complete"
  log "batch-complete ${batch}"
done
log "traj and block batches complete"
