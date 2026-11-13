"""Paired full-coverage LTSV execution pilot on source and safe Ref only.

Gates are fixed before measurement: window rank correlation >= .98,
rank-probability TV <= .02, top-decile overlap >= .90, Ref MSE drift <= .01.
These are engineering screening gates, not evidence of downstream MASE parity.
"""
import argparse
from dataclasses import replace
from pathlib import Path
import json
import time

import numpy as np
import torch
from scipy.stats import spearmanr

from baselines.common import reference_windows
from baselines.ltsv.fast import BlockScorer, block_rng, block_starts, reference_mse
from baselines.ltsv.algorithms import ltsv_window_scores
from baselines.ltsv.runner import _ltsv_training_loss
from rage.execution import SelectionExecution, prepare_contexts, selection_execution
from rage.experiments.development_inputs import load_fragments
from rage.experiments.full_pool_models import build_model
from rage.experiments.resampling import ranked_mass
from rage.experiments.students import atomic_json


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def compare(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n = max(1, int(np.ceil(len(a) * .1)))
    same = np.array_equal(a, b)
    return dict(rank_correlation=1. if same else float(spearmanr(a, b).statistic),
        probability_tv=float(np.abs(ranked_mass(a, temperature=1.) - ranked_mass(b, temperature=1.)).sum() / 2),
        top_decile_overlap=len(set(np.argsort(a)[-n:]) & set(np.argsort(b)[-n:])) / n,
        score_rmse=float(np.sqrt(np.square(a-b).mean())),
        score_relative_rmse=float(np.sqrt(np.square(a-b).mean()) / max(np.std(a), 1e-12)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--student', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fragments', type=int, default=8)
    p.add_argument('--blocks-per-fragment', type=int, default=4)
    p.add_argument('--tasks', type=int, default=3)
    args = p.parse_args()
    if args.output.exists() or args.run.resolve() in args.output.resolve().parents:
        raise ValueError('Use a new diagnostic output outside the formal run')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.run / 'protocol.json').read_text())
    pool = load_fragments(Path(config['inputs']) / 'pool.npz')
    c, h = config['context_length'], config['prediction_length']
    width, stride = c+h, (c+h)//2
    length = width + (args.blocks_per_fragment-1)*stride
    eligible = [f for f in pool if f.length >= length]
    chosen = [eligible[i] for i in np.linspace(0, len(eligible)-1, args.fragments, dtype=int)]
    datasets = [config['datasets'][i] for i in np.linspace(0, len(config['datasets'])-1, args.tasks, dtype=int)]
    torch.set_num_threads(4)
    _, probe, identity = build_model(args.student, model_path=config['model_paths'][args.student])
    original = dict(config['selection_execution'])
    original.update(config.get('student_selection_execution', {}).get(args.student, {}))
    original['gradient_workers'] = 1
    probe.execution = SelectionExecution(**original)
    native = callable(getattr(probe, 'forecast', None))
    # Current Chronos auto-batch contains all 64 reference windows on this GPU.
    modes = {'baseline': (probe.execution.inference_batch, 'eager', 32)}
    modes['candidate'] = (64, 'eager' if native else 'cuda_graph', 64)
    val_sets, ref_ids = {}, {}
    for dataset in datasets:
        refs = load_fragments(Path(config['inputs']) / f'{dataset}__reference.npz')
        val, ids = reference_windows(refs, context_length=c, prediction_length=h, stride=16, max_windows=64, seed=0)
        val_sets[dataset] = val if native else prepare_contexts(probe, val)
        ref_ids[dataset] = ids
    scorer = BlockScorer(probe, next(iter(val_sets.values())).windows if not native else next(iter(val_sets.values())),
        inference_batch=64, generation='eager')
    base_cpu = scorer.base.cpu().clone()
    cpu_views = list(base_cpu.split([v.numel() for v in scorer.parameters]))
    def restore_cpu():
        with torch.no_grad():
            for parameter, value in zip(scorer.parameters, cpu_views):
                parameter.copy_(value.view_as(parameter))
    def forecast(mode, dataset):
        batch, generation, adapter_batch = modes[mode]
        probe.execution = replace(probe.execution, inference_batch=batch, generation=generation)
        return reference_mse(probe, val_sets[dataset], adapter_batch)
    rows, base_losses, setup = [], {}, {}
    torch.cuda.reset_peak_memory_stats()
    with selection_execution(probe):
        for mode in modes:
            for dataset in datasets:
                value, seconds = timed(lambda: forecast(mode, dataset))
                base_losses[mode, dataset] = value
                setup[mode, dataset] = seconds
        # Warm-up above includes graph capture; timed rows use a reusable graph.
        ordinal = 0
        for fragment in chosen:
            for begin in block_starts(length, width, stride):
                scorer.restore()
                window = np.asarray(fragment.data[begin:begin+width], dtype=np.float32)[None]
                with block_rng(probe.device, 20260916, ordinal):
                    def update():
                        optimizer = torch.optim.Adam(probe.model.parameters(), lr=1e-5,
                            foreach=False, betas=(.9,.999), eps=1e-8, weight_decay=.1)
                        probe.model.train()
                        optimizer.zero_grad(set_to_none=True)
                        loss = _ltsv_training_loss(probe, window)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(probe.model.parameters(), 1.)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        probe.model.eval()
                        return optimizer
                    optimizer, update_seconds = timed(update)
                    try:
                        order = list(modes) if ordinal % 2 else list(reversed(modes))
                        for dataset in datasets:
                            for mode in order:
                                mse, seconds = timed(lambda: forecast(mode, dataset))
                                rows.append(dict(fragment_id=int(fragment.fragment_id), start=int(begin),
                                    ordinal=ordinal, dataset=dataset, mode=mode, mse=mse,
                                    value=base_losses[mode,dataset]-mse,
                                    forecast_seconds=seconds, update_seconds=update_seconds))
                    finally:
                        _, cpu_seconds = timed(restore_cpu)
                        _, gpu_seconds = timed(scorer.restore)
                        del optimizer
                    for row in rows[-2*len(datasets):]:
                        row['cpu_restore_seconds'] = cpu_seconds
                        row['gpu_restore_seconds'] = gpu_seconds
                ordinal += 1
            print(f'{args.student}: {ordinal} paired blocks across {len(datasets)} Ref tasks', flush=True)
            atomic_json(args.output.with_suffix('.progress.json'), dict(rows=rows, completed_blocks=ordinal))
    tasks = []
    for dataset in datasets:
        values = {}
        for mode in modes:
            scores = []
            for f in chosen:
                sums, counts = np.zeros(length+1), np.zeros(length+1, dtype=np.int64)
                for row in rows:
                    if row['dataset']==dataset and row['mode']==mode and row['fragment_id']==f.fragment_id:
                        begin = row['start']
                        sums[begin:begin+width] += row['value']
                        counts[begin:begin+width] += 1
                scores.extend(ltsv_window_scores(sums, counts, np.arange(length-width+1), width).astype(np.float32))
            values[mode] = scores
        stats = compare(values['baseline'], values['candidate'])
        baseline = [r for r in rows if r['dataset']==dataset and r['mode']=='baseline']
        candidate = [r for r in rows if r['dataset']==dataset and r['mode']=='candidate']
        drift = max(abs(a['mse']-b['mse'])/max(abs(a['mse']),1e-12) for a,b in zip(baseline,candidate))
        baseline_s = np.mean([r['update_seconds']+r['forecast_seconds']+r['cpu_restore_seconds'] for r in baseline])
        candidate_s = np.mean([r['update_seconds']+r['forecast_seconds']+r['gpu_restore_seconds'] for r in candidate])
        passed = stats['rank_correlation'] >= .98 and stats['probability_tv'] <= .02 and stats['top_decile_overlap'] >= .90 and drift <= .01
        tasks.append(dict(dataset=dataset, **stats, max_relative_ref_mse_drift=drift,
            baseline_seconds_per_block=float(baseline_s), candidate_seconds_per_block=float(candidate_s),
            speedup=float(baseline_s/candidate_s), passed=bool(passed)))
    report = dict(schema='ltsv-execution-pilot-v1', model=identity, student=args.student,
        source_fragments=[int(f.fragment_id) for f in chosen], source_prefix_length=length,
        reference_ids=ref_ids, blocks=ordinal, tasks=tasks, rows=rows,
        candidate=dict(inference_batch=64, generation=modes['candidate'][1]),
        baseline_execution=original, block_seed=20260916,
        gates=dict(rank_correlation_min=.98, probability_tv_max=.02,
            top_decile_overlap_min=.90, max_relative_ref_mse_drift=.01),
        passed=all(t['passed'] for t in tasks), peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
        baseline_seconds_per_block=float(np.mean([t['baseline_seconds_per_block'] for t in tasks])),
        forecast_setup_seconds={f'{mode}/{task}':sec for (mode,task),sec in setup.items()},
        limitation='Source-prefix screening only; shared deterministic updates isolate forecast numerics. No exam data or downstream quality claim. Live MPS contention.')
    scorer.close()
    atomic_json(args.output, report)
    print(json.dumps({k:report[k] for k in ('student','tasks','passed','peak_gpu_gib')}), flush=True)

if __name__ == '__main__':
    main()
