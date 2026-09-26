import json, re
from pathlib import Path
from collections import defaultdict
import numpy as np
root=Path('/root/TS3')
run=root/'outputs/main-shared-nine-v2-20260916'
config=json.loads((run/'protocol.json').read_text())
for student in config['students']:
    print('\nSTUDENT',student)
    times=[]
    for batch in ['main-shared-ten-20260916-211','main-shared-ten-20260916-211-t025','main-shared-ten-20260916-211-gateless']:
        p=root/'outputs'/batch/'students'/student
        rows=[json.loads(f.read_text()) for f in (p/'cells').glob('*.json')]
        vals=[r.get('elapsed_seconds',r.get('elapsed_s',0)) for r in rows if r['arm']!='original']
        if vals: print('train',batch,len(vals),np.percentile(vals,[10,50,90]).tolist()); times+=vals
        for f in p.glob('*__gradient_assets.json'):
            d=json.loads(f.read_text());print('asset',batch,f.name,{k:d.get(k) for k in ['preparation_seconds','preparation_gradient_seconds','computed_windows']},d.get('method_costs'))
        byarm=defaultdict(list)
        for f in p.glob('selection_metrics.jsonl'):
            for line in f.read_text().splitlines():
                d=json.loads(line)
                if d.get('selection_seconds',0)>1 and d.get('cache_state') not in ['warm','selection_cached']:
                    byarm[d['arm']].append(d['selection_seconds'])
        print('selection',batch,{k:[len(v),float(np.median(v)),max(v)] for k,v in byarm.items()})
    cache=next(p for p in (run/'grad_cache').iterdir() if p.is_dir() and json.loads((p/'cache_manifest.json').read_text())['model']['student']==student)
    with np.load(cache/'window_index.npz') as z:
        names=z['datasets']; ids=z['window_ids']; ds=z['dataset_ids']
    with np.load(Path(config['inputs'])/'pool.npz') as z:
        lens=dict(zip(z['ids'].tolist(),np.diff(z['offsets']).tolist()))
    complete=[]
    for did,name in enumerate(names):
        if name!='source':continue
        selected=ids[ds==did]
        fids,counts=np.unique(selected[:,0],return_counts=True)
        complete=[(int(fid),lens[int(fid)]-575) for fid,n in zip(fids,counts) if n==lens[int(fid)]-575]
    print('FULL_SOURCE',len(complete),sorted(complete,key=lambda x:x[1])[:5], sorted(complete,key=lambda x:x[1])[-10:])
