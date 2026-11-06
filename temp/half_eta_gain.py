import json
from pathlib import Path
from collections import defaultdict
import numpy as np
repo=Path('/root/TS3');run=repo/'outputs/main-shared-nine-v2-20260916'
config=json.loads((run/'protocol.json').read_text());old=json.loads((repo/'temp/crb-half-20260916/eta.json').read_text())
def cover(n,k):
    a={0,n-1}; spans=[(0,n-1)]
    while len(a)<k:
        good=[(l,r) for l,r in spans if l<(l+r)//2<r]
        if not good:break
        w=max(r-l for l,r in good)
        for l,r in sorted((l,r) for l,r in good if r-l==w)[:k-len(a)]:
            m=(l+r)//2;a.add(m);spans.remove((l,r));spans.extend([(l,m),(m,r)])
    return a
geometry={}
for task in config['datasets']:
    with np.load(Path(config['inputs'])/f'{task}__reference.npz') as z:
        geometry[task]=list(zip(z['ids'].tolist(),np.maximum(np.diff(z['offsets'])-575,0).tolist()))
for student in config['students']:
    prev=next(m for m in old['models'] if m['student']==student)
    cache=next(c for c in (run/'grad_cache').iterdir() if c.is_dir() and json.loads((c/'cache_manifest.json').read_text())['model']['student']==student)
    with np.load(cache/'window_index.npz') as z:
        names=z['datasets']; ids=z['window_ids']; ds=z['dataset_ids']
    cached={str(name):set(map(tuple,ids[ds==did].tolist())) for did,name in enumerate(names) if str(name).startswith('reference:')}
    savings=[0.,0.];detail=[]
    for task,geo in geometry.items():
        if (run/'students'/student/f'{task}__gradient_assets.json').exists():continue
        prior=(repo/'outputs/main-shared-ten-20260916-211/students'/student/f'{task}__gradient_assets.json').exists()
        counts=[]
        for half in [False,True]:
            deterministic=adaptive=split=0
            for fid,n in geo:
                if not n:continue
                # LESS remains at current budget; user only asks to cut RAGE CRB.
                current_c=min(n,max(min(2,n),round(.02*n)));e=round(.01*n);s=e
                uniform=set(np.rint(np.linspace(0,n-1,current_c+e+s)).astype(int).tolist())
                c=min(n,max(min(2,n),round((.01 if half else .02)*n)))
                anchors=uniform|cover(n,c)
                deterministic+=sum((fid,a+1) not in cached.get('reference:'+task,set()) for a in anchors)
                adaptive+=2*round((.005 if half else .01)*n)
                split+=round((.005 if half else .01)*n)
            counts.append((deterministic,adaptive,split))
        full,half=counts
        # v1-complete refs: full new split ~1%; half error geometry can change.
        cautious=(full[0]+.9*(full[2] if prior else full[1]))-(half[0]+.9*half[1])
        optimistic=(full[0]+.9*(full[2] if prior else full[1]))-(half[0]+.9*(half[2] if prior else half[1]))
        savings[0]+=cautious;savings[1]+=optimistic
        detail.append((task,round(cautious),round(optimistic)))
    rate=prev['gradient_seconds_per_new_window']
    grad=[v*rate/3600 for v in savings]
    # RAGE body only, leave LESS/TSDS/LP/random/training fixed.
    rage_body=sum(120*np.sqrt(sum(n for f,n in geo)/194405) for task,geo in geometry.items())/3600
    print(student,'Ref gradient hours saved',grad,'RAGE body plausible saved hours',[rage_body*.3,rage_body*.6],'details',detail)
