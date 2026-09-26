"""Replay current CRB on immutable cached observations; never manufacture gradients."""
import argparse, json, pathlib, dataclasses
import numpy as np
from rags.crb import CRBConfig, CRBEngine
from rags.core import TimeSeriesFragment
from rags.window_reconstruction import interpolate_window_gradients

def save(path, value):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,indent=2)); tmp.replace(path)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run', required=True); p.add_argument('--out',required=True); p.add_argument('--full',action='store_true'); a=p.parse_args()
    root=pathlib.Path(a.run); out=pathlib.Path(a.out); out.mkdir(parents=True,exist_ok=True)
    cfg=CRBConfig(cover_density=.04,error_density=.02,split_density=.02,epsilon_policy='trajectory_p10')
    records=[]
    for cache in sorted((root/'grad_cache').glob('*/window_index.npz')):
        folder=cache.parent; meta=json.loads((folder/'cache_manifest.json').read_text()); idx=np.load(cache)
        datasets=idx['datasets']; ids=idx['window_ids']; dsids=idx['dataset_ids']; sid=idx['shard_ids']; srow=idx['shard_rows']
        # Snapshot the index once. Shards referenced by it are immutable.
        mats=[]
        for name in idx['shards']:
            with np.load(folder/str(name)) as z: mats.append(z['gradients_bf16'])
        print('loaded',folder.name,len(ids),flush=True)
        for di,ds in enumerate(datasets):
            for fid in np.unique(ids[dsids==di,0]):
                rows=np.flatnonzero((dsids==di)&(ids[:,0]==fid)); starts=ids[rows,1]; order=np.argsort(starts);rows=rows[order];starts=starts[order]
                if a.full:
                    manifest=root/'sample_manifest.json'
                    if manifest.exists():
                        sample=next(x for x in json.loads(manifest.read_text())['samples'] if x['dataset']==ds and x['fragment_id']==int(fid))
                        n=int(sample['windows'])
                    else:
                        norms_file=root/f'{ds}-{fid}.npz'
                        n=len(np.load(norms_file)['starts'])
                elif ds=='source': n=int(meta['data']['pool']['lengths'][int(fid)])-575
                else: continue
                bits=np.stack([mats[int(sid[r])][int(srow[r])] for r in rows]); g=(bits.astype(np.uint32)<<16).view(np.float32)
                lookup={int(s):i for i,s in enumerate(starts)}
                frag=TimeSeriesFragment(np.zeros(n+575,dtype=np.float32),fragment_id=int(fid))
                def compute(f,ss):return g[[lookup[int(s)] for s in ss]]
                try:r=CRBEngine(cfg,compute).process_fragment(frag)
                except KeyError as e:
                    records.append(dict(model=folder.name,side=str(ds),fragment=int(fid),status='incomplete_cache',missing_start=int(e.args[0]),cached=len(starts),n=n));continue
                rec=dict(model=folder.name,side=str(ds),fragment=int(fid),status='complete_replay',n=n,cached=len(starts),budgets=list(cfg.budgets(n)),used=[r.cover_computations,r.error_computations,r.split_computations],epsilon=r.epsilon_audit['epsilon'],blocks=len(r.blocks),banned=r.banned_budget,no_signal=r.discarded_no_signal,antagonistic=r.discarded_antagonistic)
                rec['split_utilization']=r.split_computations/max(1,cfg.budgets(n)[2]); rec['split_fraction_used']=r.split_computations/r.computed_gradients
                if a.full and len(starts)==n:
                    stem=f'{folder.name.split("--")[0]}__{ds}__{fid}'
                    np.savez(out/(stem+'.npz'),gradients=g)
                    rec['dense_file']=stem+'.npz'
                records.append(rec)
        save(out/'scan.json',dict(run=str(root),config=dataclasses.asdict(cfg),records=records))
        del mats
    good=[r for r in records if r['status']=='complete_replay'];good.sort(key=lambda r:(r['used'][2],r['split_utilization']),reverse=True)
    save(out/'ranking.json',good)
    print(json.dumps(good[:20],indent=2),flush=True)
    print('completed',len(good),'incomplete',len(records)-len(good),flush=True)

if __name__=='__main__':main()
