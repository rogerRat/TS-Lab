#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/TS-Lab
export PYTHONPATH=src:third_parties/fev/src:third_parties/TIME/src:.
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
python_bin=/root/autodl-tmp/rage/envs/rage-timerxl/bin/python
root=outputs/train/v1-crb-redo/12tasks-ref-grad-prepare
log() { printf '%s-[queue_v1_crb_redo_ref_feasible]-[queue]-[%s]\n' "$(date -Is)" "$*"; }
child=
cleanup() {
  if [[ -n "$child" ]]; then
    kill -INT "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
log "waiting for fev-09 reference gradients"
while [[ ! -f "$root/fev-09/.complete" ]]; do
  if ! tmux has-session -t v1-crb-redo-ref-grad-fev09 2>/dev/null; then
    log "fev-09 session ended without completion"
    exit 1
  fi
  sleep 10
done
for task in time-01 time-02 time-03 time-04 time-06 time-07 time-08; do
  test -f "data/ref/shard_dataset/v1-crb-redo-12tasks/$task/.complete"
  log "starting $task"
  "$python_bin" -u -m scripts/exp/v1-crb-redo/ref_gradients.py --task "$task" &
  child=$!
  wait "$child"
  child=
  test -f "$root/$task/.complete"
  log "completed $task"
done
log "all currently feasible reference tasks complete; four short-history tasks await rule"
