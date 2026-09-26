from pathlib import Path
import json,subprocess,time,csv,os
from datetime import datetime,timezone,timedelta
from rags.experiments.students import atomic_json
run=Path('/root/autodl-tmp/TS3/outputs/main-rank-volume-20260918')
atomic_json(run/'telemetry-process.json',dict(pid=os.getpid(),interval_seconds=5))
history=[]
while True:
    try:
        supervisor=json.loads((run/'supervisor.json').read_text())
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw',
            '--format=csv,noheader,nounits'],text=True).strip().splitlines()[0]
        util,memory,power=[float(x.strip()) for x in raw.split(',')]
        now=datetime.now(timezone(timedelta(hours=8))).isoformat()
        item=dict(time=now,gpu_utilization_percent=util,gpu_memory_mib=memory,gpu_power_w=power)
        with (run/'gpu.csv').open('a') as f:csv.writer(f).writerow([now,util,memory,power])
        history.append(item);history=history[-24:]
        states={}
        for p in (run/'rounds').glob('seed*/students/*/state.json'):
            states[str(p.relative_to(run))]=json.loads(p.read_text())
        atomic_json(run/'telemetry.json',dict(**item,supervisor=supervisor,worker_states=states,
            recent_samples=len(history),recent_gpu_mean_percent=sum(x['gpu_utilization_percent'] for x in history)/len(history),
            recent_gpu_peak_percent=max(x['gpu_utilization_percent'] for x in history),
            recent_gpu_peak_memory_mib=max(x['gpu_memory_mib'] for x in history)))
        if supervisor['status'] in ('complete','failed'):break
    except Exception as e:
        with (run/'telemetry.log').open('a') as f:f.write(repr(e)+'\n')
    time.sleep(5)
