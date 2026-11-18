#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/TS-Lab
MARKER=/root/autodl-tmp/rage/outputs/eval/main-crb-supp-24/run/.complete
UPSTREAM_SESSION=main-crb-supp
PYTHON=/root/autodl-tmp/rage/envs/rage-chronos2/bin/python
QUEUE=scripts/exp/main-crb-supp-chronos2-small/run_queue.py

cd "$PROJECT_ROOT"
printf '%s\n' 'Waiting for the TimerXL queue completion marker.'
while [[ ! -f "$MARKER" ]]; do
    if ! tmux has-session -t "=$UPSTREAM_SESSION" 2>/dev/null; then
        printf '%s\n' 'TimerXL queue exited without its completion marker; stopping dependent queue.' >&2
        exit 3
    fi
    sleep 20
done

printf '%s\n' 'TimerXL queue completed; converting its result indexes to MASE.'
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" -u scripts/exp/main-crb-supp/apply_mase_results.py
printf '%s\n' 'Preparing the Chronos-2-small queue.'
"$PYTHON" -u "$QUEUE" --mode prepare
"$PYTHON" -u "$QUEUE" --mode validate
"$PYTHON" -u "$QUEUE" --mode run
