#!/bin/bash
# V3 巡检远端采集脚本（由本地 v3_patrol.sh 经 ssh 调用）
RUN=/root/autodl-tmp/TS3/outputs/main-shared-nine-v3-20260917
PY=/root/miniconda3/envs/ts3/bin/python
echo "# V3 巡检 $(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "- 守护实例: $(pgrep -cf 'bash /root/TS3/temp/v3_scheduler.sh')  补拉预算余额: $((30 - $(cat /root/TS3/temp/v3_scheduler.count 2>/dev/null || echo 0)))"
echo "## Python worker"
ps -eo pid,etime,%cpu,rss,args | grep 'full_pool_worker' | grep 'v3-20260917' | grep -v grep | grep 'bin/python' | grep -v bash | awk '{printf "PID=%s %s CPU=%s%% RSS=%.1fGB %s\n", $1, $2, $3, $4/1048576, $NF}'
echo "## 资源"
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
echo "mem.current=$(( $(cat /sys/fs/cgroup/memory.current) / 1073741824 ))GB / $(( $(cat /sys/fs/cgroup/memory.max) / 1073741824 ))GB  oom_kill=$(grep oom_kill /sys/fs/cgroup/memory.events | awk '{print $2}')"
echo "## cells (v3 过滤, 每模型 63, 总 189)"
$PY - "$RUN" <<'PYEOF'
import sys, json
from pathlib import Path
run = Path(sys.argv[1])
total = 0
for s in ['chronos-base', 'timesfm-2.5', 'timemoe-200m']:
    cells = list((run/'students'/s/'cells').glob('*.json'))
    n = sum(1 for c in cells if json.loads(c.read_text()).get('experiment_version') == 'v3')
    print(f'{s}={n}')
    total += n
print(f'total={total}')
PYEOF
echo "## 各模型状态"
for s in chronos-base timesfm-2.5 timemoe-200m; do
  echo "$s: $(cat $RUN/students/$s/state.json 2>/dev/null | tr -d '\n' | cut -c1-130)"
done
echo "## 失败检测"
for s in chronos-base timesfm-2.5 timemoe-200m; do
  tail -3 $RUN/$s.log 2>/dev/null | grep -q 'Traceback' && echo "$s: 日志尾部 Traceback!"
done
echo "fail-scan-done"
echo "## 刷新 MAIN-v3"
cd /root/TS3 && $PY -m rage.experiments.v3_report --run "$RUN" 2>&1 | tail -2
