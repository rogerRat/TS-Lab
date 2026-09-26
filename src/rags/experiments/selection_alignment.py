"""Before/after gradient alignment under actual delivered-window sampling."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .arms import WindowIndex, streaming_random_select
from .calibration import build_student
from .development_inputs import load_fragments
from .native_probe import NativeProbe
from .regions import load_selection
from .students import atomic_json
from rags.execution import SelectionExecution


def alignment_metrics(source, reference):
    source, reference = np.asarray(source, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    sn, rn = np.linalg.norm(source, axis=1), np.linalg.norm(reference, axis=1)
    su = np.divide(source, sn[:, None], out=np.zeros_like(source), where=sn[:, None] > 0)
    ru = np.divide(reference, rn[:, None], out=np.zeros_like(reference), where=rn[:, None] > 0)
    cosines = su @ ru.T
    best = cosines.max(axis=1)
    sm, rm = source.mean(axis=0), reference.mean(axis=0)
    denom = float(np.linalg.norm(sm) * np.linalg.norm(rm))
    valid = (sn[:, None] > 0) & (rn[None, :] > 0)
    return dict(mean_pair_cosine=float(cosines.mean()),
        mean_best_reference_cosine=float(best.mean()),
        best_reference_quantiles=np.quantile(best, [0,.1,.5,.9,1]).tolist(),
        cosine_between_mean_raw_gradients=float(sm@rm/denom) if denom > 0 else None,
        positive_pair_fraction=float(np.mean(cosines > 0)),
        valid_pair_mean_cosine=float(cosines[valid].mean()) if valid.any() else None,
        source_zero_gradient_fraction=float(np.mean(sn == 0)),
        reference_zero_gradient_fraction=float(np.mean(rn == 0)),
        source_norm_quantiles=np.quantile(sn, [0,.1,.5,.9,1]).tolist(),
        reference_norm_quantiles=np.quantile(rn, [0,.1,.5,.9,1]).tolist(),
        best_cosine_sampling_standard_error=float(best.std(ddof=1)/np.sqrt(len(best))),
        zero_convention='zero-gradient pairs score 0 in inclusive summaries; valid-pair mean excludes them')


def direction_groups(reference, k):
    """Deterministic spherical clustering fitted only to reference gradients."""
    ref = np.asarray(reference, dtype=np.float64)
    norms = np.linalg.norm(ref, axis=1)
    ref = ref[norms > 0] / norms[norms > 0, None]
    centers = [ref[0]]
    for _ in range(1, k):
        centers.append(ref[np.argmin((ref @ np.asarray(centers).T).max(axis=1))])
    centers = np.asarray(centers)
    for _ in range(30):
        labels = (ref @ centers.T).argmax(axis=1)
        updated = centers.copy()
        for j in range(k):
            members = ref[labels == j]
            if len(members):
                mean = members.mean(axis=0)
                norm = np.linalg.norm(mean)
                if norm > 0:
                    updated[j] = mean / norm
        if np.allclose(updated, centers, atol=1e-10):
            break
        centers = updated
    return centers


def mixture_metrics(source, reference, centers):
    def assign(values):
        values = np.asarray(values, dtype=np.float64)
        norms = np.linalg.norm(values, axis=1)
        unit = np.divide(values, norms[:, None], out=np.zeros_like(values), where=norms[:, None] > 0)
        similarities = unit @ centers.T
        labels = similarities.argmax(axis=1)
        labels[norms == 0] = len(centers)  # Zero gradients are not silently assigned to group 0.
        shares = np.bincount(labels, minlength=len(centers)+1) / len(labels)
        return labels, shares, similarities.max(axis=1)
    labels, shares, best = assign(source)
    ref_labels, target, _ = assign(reference)
    return dict(source_shares=shares.tolist(), reference_shares=target.tolist(),
        share_difference=(shares-target).tolist(), total_variation=float(abs(shares-target).sum()/2),
        source_labels=labels.tolist(), reference_labels=ref_labels.tolist(),
        zero_gradient_group=len(centers), nonpositive_best_center_fraction=float(np.mean(best <= 0)),
        best_center_cosine_quantiles=np.quantile(best,[0,.1,.5,.9,1]).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--student',required=True,choices=['chronos-small','timesfm-2.5','timemoe-200m'])
    p.add_argument('--selection-run',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--arms',nargs='+',default=['rags','trajectory_level','less','tsds','lp_mirror','ltsv'])
    p.add_argument('--samples',type=int,default=512)
    p.add_argument('--reference-samples',type=int,default=64)
    p.add_argument('--direction-groups',type=int,nargs='+',default=[])
    p.add_argument('--seed',type=int,default=20260914)
    args=p.parse_args()
    torch.set_num_threads(4)
    run=Path(args.selection_run)
    protocol=json.loads((run/'protocol.json').read_text())
    inputs=Path(protocol['arguments']['inputs'])
    dataset=protocol['arguments']['dataset']
    for arm in args.arms:
        if not (run/'selection'/(arm+'.npz')).exists():
            raise FileNotFoundError(f'Incomplete selection: {arm}')
    def windows(path):
        fragments=load_fragments(path)
        index=WindowIndex(fragments,context_length=512,prediction_length=64,stride=1)
        return streaming_random_select(index,n_select=128).selected
    pool=windows(inputs/'pool.npz')
    refs=windows(inputs/(dataset+'.npz'))
    engine=build_student(args.student)
    engine.execution = SelectionExecution(gradient_batch=4, inference_batch=4)
    probe=engine if args.student=='chronos-small' else NativeProbe(engine,model_key=args.student)
    rng=np.random.default_rng(args.seed)
    ref_rows=rng.integers(0,len(refs),size=args.reference_samples)
    _,reference=probe.batch_gradient_sketch(refs[ref_rows])
    centers = {k: direction_groups(reference, k) for k in args.direction_groups}
    record=dict(arguments=vars(args),source_protocol=protocol,reference_rows=ref_rows.tolist(),
        purpose='Monte Carlo alignment in projected gradient space; not downstream efficacy or all-window exact result',
        sampling='uniform with replacement over delivered legal occurrences; repeats retained',arms={})
    before=None
    for name in ['before_random_pool']+args.arms:
        selected=pool if name=='before_random_pool' else load_selection(run/'selection'/(name+'.npz')).selected
        rows=rng.integers(0,len(selected),size=args.samples)
        started=time.monotonic()
        _,sketch=probe.batch_gradient_sketch(selected[rows])
        metrics=alignment_metrics(sketch,reference)
        mixtures = {str(k): mixture_metrics(sketch, reference, c) for k,c in centers.items()}
        if before is None:
            before=metrics
        delta={key:metrics[key]-before[key] for key in
               ['mean_pair_cosine','mean_best_reference_cosine','positive_pair_fraction']}
        record['arms'][name]=dict(metrics=metrics,delta_from_before=delta,sampled_rows=rows.tolist(),
                                 direction_mixtures=mixtures,
                                 seconds=time.monotonic()-started)
        atomic_json(Path(args.out),record)
        print(name,json.dumps(metrics),flush=True)
    print('SELECTION_ALIGNMENT_DONE',flush=True)


if __name__=='__main__':
    main()
