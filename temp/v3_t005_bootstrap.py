import json,os,signal,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/root/TS3');os.chdir(ROOT)
OLD='/root/autodl-tmp/TS3/outputs/main-shared-nine-v3-20260917'
NEW='/root/autodl-tmp/TS3/outputs/main-shared-nine-v3-t005-20260917'
run=Path(NEW)
def processes(needle):
 result=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:a=(p/'cmdline').read_bytes().split(b'\0');a=[x.decode() for x in a if x]
  except (FileNotFoundError,ProcessLookupError,PermissionError):continue
  if a and a[0].endswith('bin/python') and needle in a:result.append(int(p.name))
 return result
def detach(args,log):
 with open(log,'ab') as out:
  return subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=out,stderr=out,close_fds=True,start_new_session=True)
print('BOOTSTRAP',datetime.now(timezone.utc).isoformat(),flush=True)
while processes('temp/derive_t005.py'):time.sleep(3)
if not (run/'temperature_change.json').exists():
 print('Retrying interrupted CPU preparation',flush=True)
 subprocess.run(['/root/miniconda3/envs/ts3/bin/python','temp/derive_t005.py'],check=True)
c=json.loads((run/'protocol.json').read_text())
assert all(v==.05 for s in c['students'] for v in c['settings'][s]['temperature'].values())
for pid in processes(OLD):
 os.kill(pid,signal.SIGTERM)
for i in range(10):
 if not processes(OLD):break
 time.sleep(1)
for pid in processes(OLD):os.kill(pid,signal.SIGKILL)
from rags.experiments.v3_report import render
out=ROOT/'temp/MAIN-v3-T005-active.md'
render(run,out)
s=out.read_text().replace('<!-- v3-notes:end -->','\n### 温度切换\n所有涉及采样温度的默认方法统一 T005（0.05）：lp-mirror、less-main、rags-main、rags-traj。新方法沿用原定名称，旧版所有方法对照展示加 -T025 后缀；orig/rand/tsds无该温度，旧后缀仅标记历史批次。四温度臂重新训练；其余已完成同协议结果有显式复用来源。旧结果见 [T025快照](archive/MAIN-v3-t025-snapshot-20260917.md)。\n<!-- v3-notes:end -->')
out.write_text(s);out.replace(ROOT/'docs/MAIN-v3.md')
scheduler=(ROOT/'temp/v3_scheduler.sh').read_text().replace(OLD,NEW).replace('v3_scheduler.','v3_t005_scheduler.').replace('sleep 45','sleep 10')
scheduler=scheduler.replace('pgrep -f -- "--student $student\\$"','pgrep -f -- "rags.experiments.full_pool_worker --run $RUN --student $student\\$"')
p=ROOT/'temp/v3_t005_scheduler.sh';p.write_text(scheduler)
subprocess.run(['/bin/bash','-n',str(p)],check=True)
guard=detach(['/bin/bash',str(p)],ROOT/'temp/v3_t005_scheduler.launch.log')
(ROOT/'temp/v3_t005_scheduler.pid').write_text(str(guard.pid)+'\n')
loop=ROOT/'temp/v3_t005_report_loop.sh'
loop.write_text('#!/bin/bash\ncd /root/TS3\nwhile true; do\nPYTHONPATH=/root/TS3/src /root/miniconda3/envs/ts3/bin/python -m rags.experiments.v3_report --run '+NEW+'\nsleep 1800\ndone\n')
report=detach(['/bin/bash',str(loop)],ROOT/'temp/v3_t005_report.log')
(ROOT/'temp/v3_t005_report.pid').write_text(str(report.pid)+'\n')
with (ROOT/'docs/Progress.md').open('a') as f:f.write('\nT005远端后台已启动：'+NEW+'；guard PID'+str(guard.pid)+'，30分钟report PID'+str(report.pid)+'。三个模型10秒错峰并行，独立session/关闭继承fds，不依赖用户本机或Claude Cron。默认四温度方法全部.05，旧展示-T025，配置/选集及来源均已验证，31项回归通过；等待新训练成绩。\n')
(run/'remote_launch.json').write_text(json.dumps(dict(status='running',guard_pid=guard.pid,report_pid=report.pid,local_computer_required=False),indent=2))
print('REMOTE RUNNING guard',guard.pid,'report',report.pid,flush=True)
