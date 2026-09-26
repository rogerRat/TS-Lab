"""Record compact live V5 supervision facts without changing execution."""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import shutil
import subprocess

RUN = Path('/root/autodl-tmp/TS3/outputs/main-rank-volume-20260918')
config = json.loads((RUN/'protocol.json').read_text())
supervisor = json.loads((RUN/'supervisor.json').read_text())
state = json.loads((RUN/'state.json').read_text())
telemetry = json.loads((RUN/'telemetry.json').read_text())
records = {}
for path in RUN.glob('rounds/seed*/students/*/cells/*.json'):
    row = json.loads(path.read_text())
    assert row.get('experiment_version') == 'v5', path
    key = tuple(row[k] for k in ('student', 'dataset', 'arm', 'seed'))
    records[key] = row
workers = {}
for path in RUN.glob('rounds/seed*/students/*/state.json'):
    workers[str(path.relative_to(RUN))] = json.loads(path.read_text())
failed = {k:v for k,v in workers.items() if v.get('status')=='failed'}
errors = {}
for path in list(RUN.glob('rounds/seed*/*.log'))+[RUN/'supervisor.log']:
    lines = path.read_text(errors='replace').splitlines()[-200:]
    matches = [x for x in lines if any(w in x for w in ('Traceback', 'OutOfMemoryError', 'CUDA out of memory'))]
    if matches:
        errors[str(path.relative_to(RUN))] = matches[-5:]
completed_tasks = []
for seed in config['seeds']:
    for student in config['students']:
        for task in config['datasets']:
            arms = {k[2] for k in records if k[0]==student and k[1]==task and k[3]==seed and k[2]!='orig'}
            if len(arms) == len(config['arms'])-1:
                completed_tasks.append(dict(seed=seed, student=student, task=task))
disk = shutil.disk_usage(RUN)
trained = [x for x in records.values() if x['arm']!='orig']
gpu_now = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw', '--format=csv,noheader,nounits'], text=True).strip()
stamp = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
facts = dict(time=stamp, supervisor=supervisor, state=state,
    completed_cells=len(records), trained_cells=len(trained), workers=workers,
    failed_states=failed, recent_error_markers=errors, completed_model_task_seeds=completed_tasks,
    gpu_now_util_memory_mib_power_w=gpu_now,
    rolling_gpu_mean_percent=telemetry['recent_gpu_mean_percent'],
    rolling_gpu_peak_percent=telemetry['recent_gpu_peak_percent'],
    rolling_gpu_peak_memory_mib=telemetry['recent_gpu_peak_memory_mib'],
    telemetry_time=telemetry['time'], free_disk_gib=round(disk.free/2**30,2))
note = f'\n### V5 scheduled personal supervision {stamp}\n- Live facts: '+json.dumps(facts,ensure_ascii=False)+'. MAIN refreshed with V4-layout result tables and all task/global sort checks. Normal progress if no failures/error markers; no restart, duplicate launch, scientific/execution change or manual deletion in this patrol. Continue current-thread supervision.\n'
with Path('docs/Progress.md').open('a') as handle:
    handle.write(note)
print(json.dumps(facts,ensure_ascii=False))
