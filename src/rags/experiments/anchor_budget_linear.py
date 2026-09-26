"""Corrected cache-only window-start evaluation with frozen p10 classification."""
import argparse
import json
from pathlib import Path
import numpy as np
from rags.crb import CRBConfig, CRBEngine
from rags.epsilon_protocol import load_p10_calibration
from rags.window_reconstruction import interpolate_window_gradients
from .anchor_budget_simulate import (ArrayGradientComputer, atomic_write,
    check_rows_complete, load_entry_fragment, load_gradient_dataset,
    load_sample_manifest, manifest_entries, completed_keys)


def window_cosines(truth, result):
    scores = np.zeros(len(truth), dtype=np.float64)
    for lo in range(0, len(truth), 256):
        rows = np.arange(lo, min(lo+256, len(truth)))
        gt = truth[rows].astype(np.float64)
        estimate = interpolate_window_gradients(result, rows)
        denominator = np.linalg.norm(gt, axis=1)*np.linalg.norm(estimate, axis=1)
        scores[rows] = np.divide(np.einsum('ij,ij->i', gt, estimate), denominator,
            out=np.zeros(len(rows)), where=denominator > 0)
    return scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    manifest = load_sample_manifest(args.run_root)
    cache = Path(manifest['cache'])
    identity = json.loads((cache/'cache_manifest.json').read_text())['model']
    calibration = load_p10_calibration(args.calibration, identity)
    entries = manifest_entries(manifest, {0})
    done = completed_keys(args.run_root)
    if len(entries) != 5 or any((e['task'],e['fragment_id'],e['wave']) not in done for e in entries):
        raise ValueError('Five complete rank0 trajectories required')
    dataset = load_gradient_dataset(cache)
    check_rows_complete(dataset, entries)
    args.out.mkdir(parents=True, exist_ok=False)
    atomic_write(args.out/'protocol.json', dict(calibration=calibration,
        calibration_path=str(args.calibration), metric='mean per-window cosine',
        reconstruction='raw-vector piecewise linear at every legal window start, including anchors; independent of final block rejection',
        samples=entries, model_calls=0))
    # Uniform and risk share identical coverage and exactly the same error allocation count.
    conditions = [('coverage',2,0,0,'risk'), ('uniform_extra',2,2,0,'uniform'),
        ('error_extra',2,2,0,'risk'), ('error_split',2,2,2,'risk'),
        ('coverage6',6,0,0,'risk'), ('candidate15_1_1',15,1,1,'risk')]
    rows = []
    for entry in entries:
        fragment = load_entry_fragment(entry)
        starts = fragment.valid_starts(512,64)
        truth = dataset.lookup(entry['dataset'],np.column_stack([
            np.full(len(starts),entry['fragment_id']),starts]))
        computer = ArrayGradientComputer(starts,truth,entry['fragment_id'])
        for name,c,e,s,placement in conditions:
            result = CRBEngine(CRBConfig(cover_density=c/100,error_density=e/100,
                split_density=s/100,error_placement=placement,
                mean_epsilon=calibration['epsilon'],endpoint_epsilon=calibration['epsilon']),computer).process_fragment(fragment)
            scores = window_cosines(truth,result)
            row = dict(task=entry['task'],condition=name,budget_percent=[c,e,s],
                mean_cosine=float(scores.mean()),pass_mean=bool(scores.mean()>=.9),
                anchors=len(result.anchors),windows=len(starts),split_used=result.split_computations,
                no_signal=result.discarded_no_signal,antagonistic=result.discarded_antagonistic,
                banned=result.banned_budget)
            rows.append(row)
            atomic_write(args.out/'rows.json',rows)
            print(json.dumps(row),flush=True)


if __name__ == '__main__':
    main()
