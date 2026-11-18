#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/TS-Lab
PYTHON=/root/autodl-tmp/rage/envs/rage-timerxl/bin/python
QUEUE=scripts/exp/main-crb-supp/run_queue.py

cd "$PROJECT_ROOT"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
printf '%s\n' 'Preparing the 24-task TimerXL queue.'
"$PYTHON" -u "$QUEUE" --mode prepare
printf '%s\n' 'Validating the TimerXL queue.'
"$PYTHON" -u "$QUEUE" --mode validate
printf '%s\n' 'Running the 12-task TimerXL extension.'
"$PYTHON" -u "$QUEUE" --mode extend
