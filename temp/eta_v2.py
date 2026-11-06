import json,re
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
import numpy as np
repo=Path('/root/TS3');run=repo/'outputs/main-shared-nine-v2-20260916'
config=json.loads((run/'protocol.json').read_text()); inputs=Path(config['inputs'])
out=repo/'temp/crb-half-20260916';out.mkdir(exist_ok=True,parents=True)
now=datetime.now(ZoneInfo('Asia/Shanghai'))
geometry={}
for task in config['datasets']:
    with np.load(inputs/f'{task}__reference.npz') as z:
        sizes=np.maximum(np.diff(z['offsets'])-575,0); fids=z['ids']
    geometry[task]=[(int(fid),int(n)) for fid,n in zip(fids,sizes) if n>0]

def budget(n):return min(n,max(min(2,n),round(.02*n))),round(.01*n),round(.01*n)
def coverage(n,k):
    anchors={0,n-1};intervals=[(0,n-1)]
    while len(anchors)<k:
        valid=[(a,b) for a,b in intervals if a<(a+b)//2<b]
        if not valid:break
        width=max(b-a for a,b in valid)
        frontier=sorted((a,b) for a,b in valid if b-a==width)[:k-len(anchors)]
        for a,b in frontier:
            m=(a+b)//2;anchors.add(m);intervals.remove((a,b));intervals.extend([(a,m),(m,b)])
    return anchors
det={}
for task,geo in geometry.items():
    rows=[]
    for fid,n in geo:
        c,e,s=budget(n); uniform=set(np.rint(np.linspace(0,n-1,c+e+s)).astype(int).tolist())
        rows.append((fid,n,coverage(n,c)|uniform,e+s,s))
    det[task]=rows
report=dict(snapshot=now.isoformat(),run=str(run),tasks_geometry={k:dict(trajectories=len(v),windows=sum(n for _,n in v)) for k,v in geometry.items()},models=[])
for student in config['students']:
    p=run/'students'/student
    state=json.loads((p/'state.json').read_text())
    cache=next(c for c in (run/'grad_cache').iterdir() if c.is_dir() and json.loads((c/'cache_manifest.json').read_text())['model']['student']==student)
    with np.load(cache/'window_index.npz') as z:
        names=z['datasets'];ids=z['window_ids'];ds=z['dataset_ids']
    cached=defaultdict(set)
    for did,name in enumerate(names):
        if str(name).startswith('reference:'):
            cached[str(name)]=set(map(tuple,ids[ds==did].tolist()))
    timing=json.loads((cache/'gradient_timing.json').read_text())['records']
    expected=dict(config['selection_execution']);expected.update(config['student_selection_execution'][student])
    matches=[r for r in timing if r['gradient_windows']>0 and all(r['identity']['execution'].get(k)==expected[k] for k in ['generation','gradient_batch','gradient_workers'])]
    rate=sum(r['gradient_seconds'] for r in matches)/sum(r['gradient_windows'] for r in matches)
    hist=[]
    for batch in ['main-shared-ten-20260916-211','main-shared-ten-20260916-211-t025','main-shared-ten-20260916-211-gateless']:
        for f in (repo/'outputs'/batch/'students'/student/'cells').glob('*.json'):
            r=json.loads(f.read_text())
            if r['arm']=='original' or r['dataset'] not in config['datasets']:continue
            recipe=r.get('training_recipe',{})
            if recipe.get('epochs')!=config['settings'][student]['epochs']:continue
            value=r.get('elapsed_seconds',r.get('elapsed_s'))
            if value:hist.append(value)
    quant=np.percentile(hist,[10,50,90]).tolist()
    cells=[json.loads(f.read_text()) for f in (p/'cells').glob('*.json')]
    trained=sum(r['arm']!='original' for r in cells)
    tasks=[]
    for task in config['datasets']:
        if (p/f'{task}__gradient_assets.json').exists():missinglo=missingmid=missinghi=0
        else:
            keys=cached['reference:'+task]
            missinglo=sum(sum((fid,a+1) not in keys for a in anchors) for fid,n,anchors,adapt,split in det[task])
            adapt=sum(v[3] for v in det[task])
            prior=repo/'outputs/main-shared-ten-20260916-211/students'/student/f'{task}__gradient_assets.json'
            central=sum(v[4] for v in det[task]) if prior.exists() else adapt
            missinghi=missinglo+adapt;missingmid=missinglo+.9*central
        tasks.append(dict(task=task,windows=sum(n for fid,n in geometry[task]),deterministic_missing=missinglo,
            central_missing=round(missingmid),maximum_missing=missinghi))
    assetcomplete=(p/f'{config["datasets"][0]}__gradient_assets.json').exists()
    if assetcomplete:source_remaining=0
    else:
        log=(run/f'{student}.log').read_text()
        progress=re.findall(r'\[pool CRB\] (\d+)/(\d+) trajectories.*?\((\d+)s\)',log)
        done,total,elapsed=map(int,progress[-1]); batchdone=min(total,((done-1)//256+1)*256)
        current_elapsed=max(elapsed,(now-datetime(2026,9,16,22,47,tzinfo=now.tzinfo)).total_seconds()-15)
        source_remaining=max(0,elapsed/batchdone*total-current_elapsed)
    # Warm source reconstruction/CPU asset overhead: observed current TimesFM 123 s/task.
    warm_overhead=sum(not (p/f'{t}__gradient_assets.json').exists() for t in config['datasets'])*180
    # Selection bodies: first cold task includes tsds pool features, subsequent tasks reuse them.
    measured={json.loads(line)['arm']:json.loads(line)['selection_seconds'] for line in (p/'selection_metrics.jsonl').read_text().splitlines() if json.loads(line)['arm']!='original'}
    # Scale task-dependent bodies with reference windows; trajectory GLS and LP are small.
    refs0=report['tasks_geometry'][config['datasets'][0]]['windows']
    sel=0
    for task in config['datasets']:
        scale=(report['tasks_geometry'][task]['windows']/refs0)**.5
        for arm in config['arms']:
            if (p/'selections'/f'{task}__{arm}.npz').exists():continue
            base=measured.get(arm,{'rage_gateless':120,'lessvg_t025':25,'random':1,'tsds':90,'lpv_t025':50,'rage_traj_gls':5}.get(arm,30))
            sel+=base*scale if arm in ['rage_gateless','lessvg_t025'] else base
    ntrain=162-trained
    gradlo=sum(t['deterministic_missing'] for t in tasks)*rate
    gradmid=sum(t['central_missing'] for t in tasks)*rate
    gradhi=sum(t['maximum_missing'] for t in tasks)*rate
    low=source_remaining*.7+gradlo*.8+warm_overhead+sel*.7+ntrain*quant[0]
    mid=source_remaining+gradmid+warm_overhead+sel+ntrain*quant[1]
    high=source_remaining*1.4+gradhi*1.3+warm_overhead*1.5+sel*2+ntrain*quant[2]
    row=dict(student=student,state=state,completed_cells=len(cells),completed_training_cells=trained,
        gradient_seconds_per_new_window=rate,timing_matched_windows=sum(r['gradient_windows'] for r in matches),
        historical_training_count=len(hist),training_cell_p10_median_p90_seconds=quant,
        reference_missing=tasks,source_remaining_seconds=source_remaining,warm_asset_overhead_seconds=warm_overhead,
        selection_remaining_seconds=sel,reference_gradient_seconds=[gradlo,gradmid,gradhi],training_remaining_seconds=ntrain*quant[1],
        remaining_hours=[low/3600,mid/3600,high/3600],estimated_finish=(now+timedelta(seconds=mid)).isoformat())
    report['models'].append(row)
report['batch_remaining_hours']=[max(r['remaining_hours'][i] for r in report['models']) for i in range(3)]
report['method']='Known coverage+LESS union cache misses exactly counted; adaptive extra probes bounded by error+split ceilings; central 90% of split ceiling for v1-complete refs, 90% of error+split otherwise. Future source gradient calls zero after first-task completion. Matching current gradient execution latency, historical recipe-matched training p10/median/p90. Shared GPU model wall ETA=max, not sum. Engineering range, not statistical CI; no outages.'
(out/'eta.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
