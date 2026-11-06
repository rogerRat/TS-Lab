#!/bin/bash
# v2-global 阶段感知调度器（替代看门狗）：
#  - 训练/评估阶段尽量并行（非 blocking）
#  - Azure 级大资产/选集同时只允许一个模型；azure 选集崩溃窗口内以 blocking 诊断模式拉起
#  - cgroup RAM 超限不起拉；flock 单例；错峰启动避 MPS 竞态
exec 9>/root/TS3/temp/phase_scheduler.lock
flock -n 9 || exit 0
RUN=/root/autodl-tmp/TS3/outputs/main-shared-nine-v2-global-20260917
LOG=/root/TS3/temp/phase_scheduler.log
COUNT=/root/TS3/temp/phase_scheduler.count
PY=/root/miniconda3/envs/ts3/bin/python
BIG=time_azure_i_cpu_5t
RAM_LIMIT_GB=78
EXPECT_CELLS=81
[ -f "$COUNT" ] || echo 0 > "$COUNT"

python_worker_pids() { pgrep -f -- 'bin/python -m rage.experiments.full_pool_worker'; }

state_of() {  # $1=student -> "stage dataset"
  $PY -c "import json;d=json.load(open('$RUN/students/$1/state.json'));print(d.get('stage','?'),d.get('dataset','?'))" 2>/dev/null
}
cells_of() {
  $PY - "$RUN" "$1" <<'PYEOF' 2>/dev/null
import sys, json
from pathlib import Path
run, s = sys.argv[1:3]
print(sum(1 for c in (Path(run)/'students'/s/'cells').glob('*.json')
          if json.loads(c.read_text()).get('experiment_version')=='v2'))
PYEOF
}
mem_gb() { echo $(( $(cat /sys/fs/cgroup/memory.current) / 1073741824 )); }

while true; do
  someone_big=0
  for s in chronos-base timesfm-2.5 timemoe-200m; do
    if pgrep -f -- "--student $s\$" >/dev/null; then
      read stage dset <<< "$(state_of $s)"
      if [ "$dset" = "$BIG" ] && { [ "$stage" = "assets" ] || [ "$stage" = "selection" ]; }; then
        someone_big=1
      fi
    fi
  done
  for entry in "chronos-base:/root/miniconda3/envs/ts3/bin/python" \
               "timesfm-2.5:/root/autodl-tmp/conda/envs/ts3-timesfm/bin/python" \
               "timemoe-200m:/root/autodl-tmp/conda/envs/ts3-timemoe/bin/python"; do
    student=${entry%%:*}; pybin=${entry#*:}
    pgrep -f -- "--student $student\$" >/dev/null && continue
    [ "$(cells_of $student)" -ge $EXPECT_CELLS ] && continue
    [ "$(mem_gb)" -gt $RAM_LIMIT_GB ] && { echo "$(date '+%H:%M:%S') hold $student (ram $(mem_gb)G)" >> "$LOG"; continue; }
    read stage dset <<< "$(state_of $student)"
    big_now=0
    if [ "$dset" = "$BIG" ] && { [ "$stage" = "assets" ] || [ "$stage" = "selection" ]; }; then big_now=1; fi
    if [ "$big_now" -eq 1 ] && [ "$someone_big" -eq 1 ]; then
      echo "$(date '+%H:%M:%S') gate $student (big-phase busy)" >> "$LOG"; continue
    fi
    if [ "$(cat $COUNT)" -ge 30 ]; then echo "$(date '+%H:%M:%S') budget-exhausted $student" >> "$LOG"; continue; fi
    echo "$(date '+%H:%M:%S') launch $student stage=$stage/$dset ram=$(mem_gb)G" >> "$LOG"
    cd /root/TS3 && env PYTHONPATH=/root/TS3/src HF_HOME=/root/autodl-tmp/hf-cache \
      PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      nohup "$pybin" -m rage.experiments.full_pool_worker --run "$RUN" --student "$student" \
      >> "$RUN/$student.log" 2>&1 < /dev/null 9>&- &
    echo $(( $(cat $COUNT) + 1 )) > "$COUNT"
    [ "$big_now" -eq 1 ] && someone_big=1
    sleep 45
  done
  sleep 60
done
