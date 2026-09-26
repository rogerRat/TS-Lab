"""Model-free paired test of CRB interpolation and compression, archived refs only."""
import json
import sys
import time
from pathlib import Path
from dataclasses import asdict

import numpy as np

from rags.core import TimeSeriesFragment
from rags.crb import CRBConfig, CRBEngine
from rags.experiments.anchor_budget_simulate import ArrayGradientComputer
from assets.gradient.dataset import GradientDataset
from assets.gradient.variants import aggregate_retained

root = Path('/root/TS3')
archive = Path('/root/autodl-tmp/TS3/archive/outputs-cleanup-20260915/anchor-budget-20260914')
output = root / 'temp/representation-error-20260917'
output.mkdir(parents=True, exist_ok=True)
cache = next((archive / 'grad_cache').glob('*/window_index.npz')).parent
reader = GradientDataset(cache)
samples = json.loads((archive / 'scan_status.json').read_text())['completed_samples']
config = CRBConfig(cover_density=.02, error_density=.01, split_density=.01,
                   epsilon_policy='trajectory_p10')
names = ('linear', 'production_block', 'oracle_block', 'crb_trajectory', 'oracle_trajectory')
protocol = dict(config=asdict(config), cache=str(cache), model=reader.manifest['model'],
                samples=samples, model_calls=0, formal_exam_used=False,
                truth='independent per-window 4096-D projected BF16 gradients',
                limitations='historical TimeMoE reference samples only; not formal source pool or downstream training',
                normalized_squared_error='sum ||estimate-truth||^2 / sum ||truth||^2',
                integral_relative_error='||mean(estimate)-mean(truth)|| / ||mean(truth)||',
                oracle_block='exact truth mean over the production block owned windows',
                oracle_trajectory='exact truth mean over all legal windows')
(output / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
rows = []
inputs = {}

for entry in samples:
    tic = time.perf_counter()
    input_path = entry['input_path']
    if input_path not in inputs:
        with np.load(input_path) as data:
            inputs[input_path] = (data['values'], data['offsets'], data['ids'], data['entities'])
    values, offsets, ids, entities = inputs[input_path]
    selected = np.flatnonzero(ids == entry['fragment_id'])
    assert len(selected) == 1
    row_id = int(selected[0])
    fragment = TimeSeriesFragment(values[offsets[row_id]:offsets[row_id+1]],
                                  entity_id=str(entities[row_id]), fragment_id=int(ids[row_id]))
    assert fragment.length == entry['length'] and fragment.entity_id == entry['entity_id']
    starts = fragment.valid_starts(512, 64, 1)
    assert len(starts) == entry['windows']
    truth = reader.lookup(entry['dataset'], np.column_stack([
        np.full(len(starts), fragment.fragment_id), starts]))
    assert truth.shape == (len(starts), 4096) and np.isfinite(truth).all()
    result = CRBEngine(config, ArrayGradientComputer(starts, truth, fragment.fragment_id)).process_fragment(fragment)
    assert sum(block.size for block in result.blocks) == len(starts)
    truth_sum = np.sum(truth, axis=0, dtype=np.float64)
    truth_mean = truth_sum / len(starts)
    crb_mean = aggregate_retained(result).mean_gradient
    block_owner = np.empty(len(starts), dtype=np.int64)
    block_vectors, oracle_vectors = [], []
    for block_id, block in enumerate(result.blocks):
        indices = np.searchsorted(starts, block.start_positions)
        assert np.array_equal(starts[indices], block.start_positions)
        block_owner[indices] = block_id
        block_vectors.append(block.mean_gradient)
        oracle_vectors.append(np.sum(truth[indices], axis=0, dtype=np.float64) / len(indices))
    block_vectors = np.asarray(block_vectors, dtype=np.float64)
    oracle_vectors = np.asarray(oracle_vectors, dtype=np.float64)
    anchors = np.asarray(sorted(result.anchors), dtype=np.int64)
    anchor_starts = starts[anchors]
    anchor_vectors = np.stack([result.anchors[int(i)] for i in anchors]).astype(np.float64)
    metrics = {name: dict(squared_error_sum=0., cosine_sum=0., relative_l2_sum=0.,
                          estimate_sum=np.zeros(4096)) for name in names}
    truth_energy, truth_norm_sum, valid_norms = 0., 0., 0
    compression_energy, cross_term = 0., 0.
    for lo in range(0, len(starts), 256):
        hi = min(lo + 256, len(starts))
        gt = truth[lo:hi].astype(np.float64)
        norm = np.linalg.norm(gt, axis=1)
        truth_energy += np.einsum('ij,ij->', gt, gt)
        truth_norm_sum += norm.sum()
        valid_norms += int((norm > 0).sum())
        right = np.clip(np.searchsorted(anchor_starts, starts[lo:hi], side='right'), 1, len(anchors)-1)
        left = right - 1
        alpha = (starts[lo:hi] - anchor_starts[left]) / (anchor_starts[right] - anchor_starts[left])
        linear = (1-alpha[:, None])*anchor_vectors[left] + alpha[:, None]*anchor_vectors[right]
        estimates = dict(linear=linear, production_block=block_vectors[block_owner[lo:hi]],
                         oracle_block=oracle_vectors[block_owner[lo:hi]],
                         crb_trajectory=np.broadcast_to(crb_mean, gt.shape),
                         oracle_trajectory=np.broadcast_to(truth_mean, gt.shape))
        compression = estimates['production_block'] - linear
        compression_energy += np.einsum('ij,ij->', compression, compression)
        cross_term += 2*np.einsum('ij,ij->', linear-gt, compression)
        for name, estimate in estimates.items():
            difference = estimate - gt
            metric = metrics[name]
            metric['squared_error_sum'] += np.einsum('ij,ij->', difference, difference)
            metric['estimate_sum'] += estimate.sum(axis=0)
            denominator = norm * np.linalg.norm(estimate, axis=1)
            metric['cosine_sum'] += np.divide(np.einsum('ij,ij->i', gt, estimate), denominator,
                                             out=np.zeros(len(gt)), where=denominator > 0).sum()
            metric['relative_l2_sum'] += np.divide(np.linalg.norm(difference, axis=1), norm,
                                                  out=np.zeros(len(gt)), where=norm > 0).sum()
    for name, metric in metrics.items():
        estimate_mean = metric.pop('estimate_sum') / len(starts)
        metric['normalized_squared_error'] = float(metric['squared_error_sum'] / truth_energy)
        metric['mean_cosine'] = float(metric['cosine_sum'] / len(starts))
        metric['mean_relative_l2'] = float(metric['relative_l2_sum'] / valid_norms)
        metric['integral_relative_error'] = float(np.linalg.norm(estimate_mean-truth_mean) / np.linalg.norm(truth_mean))
        denominator = np.linalg.norm(estimate_mean)*np.linalg.norm(truth_mean)
        metric['mean_direction_cosine'] = float(estimate_mean @ truth_mean / denominator) if denominator > 0 else 0.
    assert np.isclose(metrics['production_block']['squared_error_sum'],
                      metrics['linear']['squared_error_sum'] + compression_energy + cross_term)
    record = dict(task=entry['task'], fragment_id=fragment.fragment_id, rank=entry['rank'],
                  windows=len(starts), blocks=len(result.blocks), anchors=len(anchors),
                  truth_energy=float(truth_energy), truth_rho=float(np.linalg.norm(truth_mean) / (truth_norm_sum/len(starts))),
                  block_compression_energy=float(compression_energy / truth_energy),
                  block_compression_cross_term=float(cross_term / truth_energy), metrics=metrics,
                  seconds=time.perf_counter()-tic)
    rows.append(record)
    temporary = output / 'rows.json.tmp'
    temporary.write_text(json.dumps(rows, indent=2))
    temporary.replace(output / 'rows.json')
    print(json.dumps(dict(task=record['task'], fragment_id=record['fragment_id'],
                         seconds=record['seconds'], mse={n: metrics[n]['normalized_squared_error'] for n in names})), flush=True)
    del truth, result, block_vectors, oracle_vectors, anchor_vectors

def summarize(group):
    windows = sum(row['windows'] for row in group)
    energy = sum(row['truth_energy'] for row in group)
    return dict(windows=windows, trajectories=len(group), metrics={name: dict(
        normalized_squared_error=sum(r['metrics'][name]['squared_error_sum'] for r in group)/energy,
        mean_cosine=sum(r['metrics'][name]['cosine_sum'] for r in group)/windows,
        trajectory_macro_integral_relative_error=float(np.mean([r['metrics'][name]['integral_relative_error'] for r in group])),
        trajectory_macro_mean_direction_cosine=float(np.mean([r['metrics'][name]['mean_direction_cosine'] for r in group]))) for name in names})

summary = dict(pooled=summarize(rows), tasks={task: summarize([r for r in rows if r['task'] == task])
                                            for task in dict.fromkeys(r['task'] for r in rows)})
(output / 'summary.json').write_text(json.dumps(summary, indent=2))
assert 'torch' not in sys.modules
print('COMPLETE', json.dumps(summary['pooled']), flush=True)
