"""Compare full/uniform/random gradient means and audit split thresholds."""
import argparse
from pathlib import Path
import numpy as np

from rags.crb import CRBConfig, CRBEngine
from .anchor_budget_simulate import (ArrayGradientComputer, FROZEN, atomic_write,
    load_sample_manifest, manifest_entries, load_gradient_dataset, check_rows_complete,
    load_entry_fragment, completed_keys)


def cosine(a,b):
    denominator = float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(a @ b / denominator) if denominator else 0.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    manifest=load_sample_manifest(args.run_root)
    entries=manifest_entries(manifest,{0})
    done=completed_keys(args.run_root)
    if len(entries)!=5 or any((e['task'],e['fragment_id'],e['wave']) not in done for e in entries):
        raise ValueError('Require all five completed rank-0 samples')
    dataset=load_gradient_dataset(manifest['cache'])
    check_rows_complete(dataset,entries)
    rows=[]
    for entry in entries:
        fragment=load_entry_fragment(entry)
        starts=fragment.valid_starts(512,64)
        truth=dataset.lookup(entry['dataset'],np.column_stack([np.full(len(starts),entry['fragment_id']),starts]))
        g=truth.astype(np.float64)
        norms=np.linalg.norm(g,axis=1)
        mean=g.mean(axis=0)
        unit_mean=np.divide(g,norms[:,None],out=np.zeros_like(g),where=norms[:,None]>0).mean(axis=0)
        def metrics(estimate):
            norm=float(np.linalg.norm(estimate))
            return dict(trajectory_mean_cosine=cosine(estimate,mean),
                        window_mean_cosine=float(unit_mean @ (estimate/norm)) if norm else 0.)
        record=dict(task=entry['task'],fragment_id=entry['fragment_id'],windows=len(starts),
                    full_mean_oracle=metrics(mean),full_mean_cost_windows=len(starts),budgets=[])
        for percent in (1,2,4,6,15):
            k=max(1,round(len(starts)*percent/100))
            uniform_rows=np.linspace(0,len(starts)-1,k,dtype=np.int64)
            trials=[]
            for seed in range(20):
                indices=np.random.default_rng(seed).choice(len(starts),k,replace=False)
                trials.append(dict(seed=seed,**metrics(g[indices].mean(axis=0))))
            record['budgets'].append(dict(percent=percent,k=k,actual_ratio=k/len(starts),
                uniform=metrics(g[uniform_rows].mean(axis=0)),random_trials=trials,
                random_summary={metric:dict(mean=float(np.mean([t[metric] for t in trials])),
                    sd=float(np.std([t[metric] for t in trials],ddof=1)),
                    minimum=float(min(t[metric] for t in trials)))
                    for metric in ('trajectory_mean_cosine','window_mean_cosine')}))
        result=CRBEngine(CRBConfig(cover_density=.02,error_density=.02,split_density=.02,**FROZEN),
                         ArrayGradientComputer(starts,truth,fragment.fragment_id)).process_fragment(fragment)
        block_norms=np.array([np.linalg.norm(b.mean_gradient.astype(np.float64)) for b in result.blocks])
        ratios=[]
        for block in result.blocks:
            left=result.anchors[block.left_index].astype(np.float64)
            right=result.anchors[block.right_index].astype(np.float64)
            denom=np.linalg.norm(left)+np.linalg.norm(right)
            ratios.append(float(np.linalg.norm(left+right)/denom) if denom else 0.)
        record['split_audit_222']=dict(mean_epsilon=FROZEN['mean_epsilon'],endpoint_epsilon=FROZEN['endpoint_epsilon'],
            min_window_gradient_norm=float(norms.min()),median_window_gradient_norm=float(np.median(norms)),
            min_block_mean_norm=float(block_norms.min()),median_block_mean_norm=float(np.median(block_norms)),
            min_cancellation_ratio=min(ratios),median_cancellation_ratio=float(np.median(ratios)),
            split_used=result.split_computations,banned=result.banned_budget,discarded_no_signal=result.discarded_no_signal)
        rows.append(record)
        atomic_write(args.out/(entry['task']+'.json'),record)
        print(entry['task'],record['full_mean_oracle'],record['split_audit_222'],flush=True)
    atomic_write(args.out/'summary.json',dict(complete=True,rows=rows,model_calls=0,
        interpretation='Random and uniform sampled gradients are averaged into one trajectory vector; '
                       'this is not random downstream training or random local interpolation.',
        random_seeds=list(range(20)),random_sampling='uniform without replacement',
        metric='raw gradient mean for trajectory target; unweighted mean of per-window cosines for reconstruction',
        frozen=FROZEN))


if __name__=='__main__':
    main()
