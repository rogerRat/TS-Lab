"""CPU-only paired replay of complete archived TimeMoE trajectories under v2."""
import json, sys, time
from pathlib import Path
from dataclasses import asdict
import numpy as np
from rage.core import TimeSeriesFragment
from rage.crb import CRBConfig, CRBEngine
from rage.experiments.anchor_budget_simulate import ArrayGradientComputer
from rage.experiments.anchor_budget_linear import window_cosines
from assets.gradient.dataset import GradientDataset

repo=Path('/root/TS3')
arch=Path('/root/autodl-tmp/TS3/archive/outputs-cleanup-20260915/anchor-budget-20260914')
out=repo/'temp/crb-half-20260916'
out.mkdir(parents=True,exist_ok=True)
cache=next((arch/'grad_cache').glob('*/window_index.npz')).parent
dataset=GradientDataset(cache)
samples=json.loads((arch/'scan_status.json').read_text())['completed_samples']
configs={'current':CRBConfig(cover_density=.02,error_density=.01,split_density=.01,epsilon_policy='trajectory_p10'),
         'half':CRBConfig(cover_density=.01,error_density=.005,split_density=.005,epsilon_policy='trajectory_p10')}
protocol=dict(model=dataset.manifest['model'],cache=str(cache),samples=samples,
    configs={k:asdict(v) for k,v in configs.items()},model_calls=0,formal_exam_used=False,
    truth='4096-dimensional BF16 cached projected independent per-window gradients; not unprojected parameter vectors',
    metric='window-weighted cosine, negative values retained, zero norm contributes zero',
    caveat='Historical reference trajectories; only TimeMoE and five tasks; not formal selected-source distribution or downstream MASE')
(out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
rows=[]
loaded={}
for entry in samples:
    path=Path(entry['input_path'])
    if str(path) not in loaded:
        with np.load(path) as z:
            loaded[str(path)]=(z['values'],z['offsets'],z['ids'],z['entities'])
    values,offsets,ids,entities=loaded[str(path)]
    idx=np.flatnonzero(ids==entry['fragment_id'])
    assert len(idx)==1
    i=int(idx[0]); fragment=TimeSeriesFragment(values[offsets[i]:offsets[i+1]],entity_id=str(entities[i]),fragment_id=int(ids[i]))
    assert fragment.length==entry['length'] and fragment.entity_id==entry['entity_id']
    starts=fragment.valid_starts(512,64,1)
    assert len(starts)==entry['windows']
    truth=dataset.lookup(entry['dataset'],np.column_stack([np.full(len(starts),fragment.fragment_id),starts]))
    assert truth.ndim==2 and truth.shape==(len(starts),4096) and np.isfinite(truth).all()
    computer=ArrayGradientComputer(starts,truth,fragment.fragment_id)
    paired={}
    for label,config in configs.items():
        tic=time.perf_counter()
        result=CRBEngine(config,computer).process_fragment(fragment)
        cos=window_cosines(truth,result)
        block_sum=0.;rel_sum=0.;nrel=0
        for b in result.blocks:
            ix=b.start_positions-1
            gt=truth[ix].astype(np.float64); mean=b.mean_gradient.astype(np.float64)
            norm=np.linalg.norm(gt,axis=1); den=norm*np.linalg.norm(mean)
            block_sum+=np.divide(gt@mean,den,out=np.zeros(len(ix)),where=den>0).sum()
        for lo in range(0,len(truth),256):
            from rage.window_reconstruction import interpolate_window_gradients
            ix=np.arange(lo,min(lo+256,len(truth))); gt=truth[ix].astype(np.float64)
            norm=np.linalg.norm(gt,axis=1); est=interpolate_window_gradients(result,ix)
            good=norm>0
            rel_sum+=np.sum(np.linalg.norm(est[good]-gt[good],axis=1)/norm[good]);nrel+=int(good.sum())
        assert sum(b.size for b in result.blocks)==len(starts)
        paired[label]=dict(mean_cosine=float(cos.mean()),median=float(np.median(cos)),p10=float(np.percentile(cos,10)),
            frac_ge_09=float(np.mean(cos>=.9)),block_mean_cosine=float(block_sum/len(starts)),
            mean_relative_l2=float(rel_sum/nrel),anchors=len(result.anchors),
            used=[result.cover_computations,result.error_computations,result.split_computations],
            epsilon_audit=result.epsilon_audit,cpu_seconds=time.perf_counter()-tic)
        del result
    row=dict(task=entry['task'],fragment_id=entry['fragment_id'],rank=entry['rank'],windows=len(starts),**paired)
    row['delta_cosine']=paired['half']['mean_cosine']-paired['current']['mean_cosine']
    rows.append(row)
    temporary=out/'rows.json.tmp';temporary.write_text(json.dumps(rows,indent=2));temporary.replace(out/'rows.json')
    print(json.dumps({k:v for k,v in row.items() if k not in ['current','half']})+' '+str({k:paired[k]['mean_cosine'] for k in configs}),flush=True)
    del truth
summary=[]
for task in dict.fromkeys(r['task'] for r in rows):
    group=[r for r in rows if r['task']==task]; n=sum(r['windows'] for r in group)
    d=dict(task=task,trajectories=len(group),windows=n)
    for label in configs:
        d[label]={key:sum(r['windows']*r[label][key] for r in group)/n for key in ['mean_cosine','block_mean_cosine','mean_relative_l2','frac_ge_09']}
        d[label]['anchors']=sum(r[label]['anchors'] for r in group)
    summary.append(d)
total=sum(r['windows'] for r in rows)
allrow=dict(windows=total,trajectories=len(rows))
for label in configs:
    allrow[label]={key:sum(r['windows']*r[label][key] for r in rows)/total for key in ['mean_cosine','block_mean_cosine','mean_relative_l2','frac_ge_09']}
    allrow[label]['anchors']=sum(r[label]['anchors'] for r in rows)
(out/'summary.json').write_text(json.dumps(dict(tasks=summary,pooled=allrow),indent=2))
assert 'torch' not in sys.modules, 'CPU model-free replay unexpectedly imported torch'
print('COMPLETE',json.dumps(allrow),flush=True)
