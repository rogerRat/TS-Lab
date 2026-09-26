#!/bin/bash
# V3 三模型并行守护：
#  - 训练/评估阶段尽量并行（非 blocking）
#  - 无固定阶段互斥；RAM guard 排除可回收 file cache
#  - cgroup RAM 超限不起拉；flock 单例；错峰启动避 MPS 竞态
exec 9>/root/TS3/temp/v3_scheduler.lock
flock -n 9 || exit 0
RUN=/root/autodl-tmp/TS3/outputs/main-shared-nine-v3-20260917
LOG=/root/TS3/temp/v3_scheduler.log
COUNT=/root/TS3/temp/v3_scheduler.count
PY=/root/miniconda3/envs/ts3/bin/python
RAM_LIMIT_GB=92
EXPECT_CELLS=63
[ -f "$COUNT" ] || echo 0 > "$COUNT"

python_worker_pids() { pgrep -f -- 'bin/python -m rags.experiments.full_pool_worker'; }

state_of() {  # $1=student -> "stage dataset"
  $PY -c "import json;d=json.load(open('$RUN/students/$1/state.json'));print(d.get('stage','?'),d.get('dataset','?'))" 2>/dev/null
}
cells_of() {
  $PY - "$RUN" "$1" <<'PYEOF' 2>/dev/null
import sys, json
from pathlib import Path
run, s = sys.argv[1:3]
print(sum(1 for c in (Path(run)/'students'/s/'cells').glob('*.json')
          if json.loads(c.read_text()).get('experiment_version')=='v3'))
PYEOF
}
mem_gb() { awk '$1 == "anon" || $1 == "kernel_stack" || $1 == "pagetables" || $1 == "slab_unreclaimable" || $1 == "shmem" { sum += $2 } END { print int(sum / 1073741824) }' /sys/fs/cgroup/memory.stat; }

while true; do
  # Respect explicit V3 execution controls before restarting any worker.
  if ! "$PY" - "$RUN/execution_control.json" <<'CONTROL'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    ready = d.get('mode') == 'run' and all(d.get(k) is True for k in ['training_enabled','evaluation_enabled','selection_enabled'])
except Exception:
    ready = False
sys.exit(0 if ready else 1)
CONTROL
  then
    sleep 60; continue
  fi
  for entry in "chronos-base:/root/miniconda3/envs/ts3/bin/python" \
               "timesfm-2.5:/root/autodl-tmp/conda/envs/ts3-timesfm/bin/python" \
               "timemoe-200m:/root/autodl-tmp/conda/envs/ts3-timemoe/bin/python"; do
    student=${entry%%:*}; pybin=${entry#*:}
    pgrep -f -- "--student $student\$" >/dev/null && continue
    [ "$(cells_of $student)" -ge $EXPECT_CELLS ] && continue
    [ "$(mem_gb)" -gt $RAM_LIMIT_GB ] && { echo "$(date '+%H:%M:%S') hold $student (ram $(mem_gb)G)" >> "$LOG"; continue; }
    read stage dset <<< "$(state_of $student)"
    if [ "$(cat $COUNT)" -ge 30 ]; then echo "$(date '+%H:%M:%S') budget-exhausted $student" >> "$LOG"; continue; fi
    echo "$(date '+%H:%M:%S') launch $student stage=$stage/$dset ram=$(mem_gb)G" >> "$LOG"
    cd /root/TS3 && env PYTHONPATH=/root/TS3/src HF_HOME=/root/autodl-tmp/hf-cache \
      PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      nohup "$pybin" -m rags.experiments.full_pool_worker --run "$RUN" --student "$student" \
      >> "$RUN/$student.log" 2>&1 < /dev/null 9>&- &
    echo $(( $(cat $COUNT) + 1 )) > "$COUNT"
    sleep 45
  done
  sleep 60
done
