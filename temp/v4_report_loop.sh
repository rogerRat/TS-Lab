#!/bin/bash
cd /root/TS3 || exit 1
exec 9>/root/TS3/temp/v4_report_loop.lock
flock -n 9 || exit 0
RUN=/root/autodl-tmp/TS3/outputs/main-v4-three-seeds-20260918
PY=/root/miniconda3/envs/ts3/bin/python
while true; do
  PYTHONPATH=/root/TS3/src "$PY" -m rage.experiments.v4_report --run "$RUN"
  if "$PY" - "$RUN/results.json" <<'PYEOF'
import json,sys
try:
    complete=json.load(open(sys.argv[1]))['complete']
except (OSError,ValueError,KeyError):
    complete=False
sys.exit(0 if complete else 1)
PYEOF
  then
    exit 0
  fi
  sleep 1800
done
