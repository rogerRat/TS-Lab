#!/bin/bash
# v2-global run worker watchdog: 单例(flock),每60秒检查,worker 缺失且仍有其他 worker 存活且该模型矩阵未完成时错峰补拉
exec 9>/root/TS3/temp/worker_watchdog.lock
flock -n 9 || exit 0
RUN=/root/autodl-tmp/TS3/outputs/main-shared-nine-v2-global-20260917
LOG=/root/TS3/temp/worker_watchdog.log
COUNT=/root/TS3/temp/worker_watchdog.count
PY=/root/miniconda3/envs/ts3/bin/python
[ -f "$COUNT" ] || echo 0 > "$COUNT"
while true; do
  alive=$(pgrep -f 'rags.experiments.full_pool_worker' | wc -l)
  for entry in "chronos-base:/root/miniconda3/envs/ts3/bin/python" \
               "timesfm-2.5:/root/autodl-tmp/conda/envs/ts3-timesfm/bin/python" \
               "timemoe-200m:/root/autodl-tmp/conda/envs/ts3-timemoe/bin/python"; do
    student=${entry%%:*}; pybin=${entry#*:}
    pgrep -f -- "--student $student\$" >/dev/null && continue
    done_cells=$($PY - "$RUN" "$student" <<'PYEOF' 2>/dev/null
import sys, json
from pathlib import Path
run, s = sys.argv[1:3]
cells = list((Path(run)/'students'/s/'cells').glob('*.json'))
print(sum(1 for c in cells if json.loads(c.read_text()).get('experiment_version')=='v2'))
PYEOF
)
    [ "$done_cells" -ge 81 ] && continue
    if [ "$alive" -ge 1 ] && [ "$(cat $COUNT)" -lt 20 ]; then
      echo "$(date '+%H:%M:%S') relaunch $student (alive=$alive cells=$done_cells)" >> "$LOG"
      cd /root/TS3 && env PYTHONPATH=/root/TS3/src HF_HOME=/root/autodl-tmp/hf-cache \
        PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
        nohup "$pybin" -m rags.experiments.full_pool_worker --run "$RUN" --student "$student" \
        >> "$RUN/$student.log" 2>&1 < /dev/null &
      echo $(( $(cat $COUNT) + 1 )) > "$COUNT"
      sleep 45   # 错峰,避开 MPS 初始化竞态
    fi
  done
  sleep 60
done
