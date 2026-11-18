#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/TS-Lab
SESSION=main-crb-supp-chronos2-small
CURRENT_MARKER=/root/autodl-tmp/rage/outputs/eval/main-crb-supp-chronos2-small/run/time-azure2019-u-5t/.complete
PYTHON=/root/autodl-tmp/rage/envs/rage-chronos2/bin/python
QUEUE=scripts/exp/main-crb-supp-chronos2-small/run_queue.py
LOG=/root/TS-Lab/logs/main-crb-supp-chronos2-small/priority_watcher.log

cd "$PROJECT_ROOT"
mkdir -p "$(dirname "$LOG")"
printf '%s-[priority-watcher]-[waiting]-task=time-azure2019-u-5t\n' "$(date '+%F %T')" >> "$LOG"
while [[ ! -f "$CURRENT_MARKER" ]]; do
    if ! tmux has-session -t "=$SESSION" 2>/dev/null; then
        printf '%s-[priority-watcher]-[error]-current-session-exited-before-marker\n' "$(date '+%F %T')" >> "$LOG"
        exit 3
    fi
    sleep 1
done
printf '%s-[priority-watcher]-[boundary]-current-task-complete\n' "$(date '+%F %T')" >> "$LOG"
PID=$(ps -eo pid=,comm=,args= | awk '$2=="python" && $0 ~ /scripts\/exp\/main-crb-supp-chronos2-small\/run_queue.py --mode run/ {print $1; exit}')
if [[ -n "$PID" ]]; then
    kill -INT "$PID"
    for _ in $(seq 1 120); do
        if ! kill -0 "$PID" 2>/dev/null; then break; fi
        sleep 1
    done
fi
if tmux has-session -t "=$SESSION" 2>/dev/null; then
    tmux kill-session -t "=$SESSION"
fi
printf '%s-[priority-watcher]-[restart]-priority=time-sg-pm25-h\n' "$(date '+%F %T')" >> "$LOG"
tmux new-session -d -s "$SESSION" "cd $PROJECT_ROOT && $PYTHON -u $QUEUE --mode run"
printf '%s-[priority-watcher]-[started]-queue=sg-first-resume-completed-task-markers\n' "$(date '+%F %T')" >> "$LOG"
