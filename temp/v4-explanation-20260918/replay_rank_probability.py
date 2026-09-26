"""Replay existing V4 selection scores; no model, training or cache mutation."""
import argparse
import json
from pathlib import Path

import numpy as np

from rags.experiments.rank_probability import REVISION, rank_probability


def atomic_json(path, value):
    partial = path.with_suffix(path.suffix + '.partial')
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    partial.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--temperature', type=float, default=.05)
    args = parser.parse_args()
    if args.out.resolve().exists():
        raise ValueError('Use a new output directory; do not overwrite a replay')
    args.out.mkdir(parents=True)
    protocol = json.loads((args.run / 'protocol.json').read_text())
    with np.load(Path(protocol['inputs']) / 'pool.npz') as pool:
        legal = np.diff(pool['offsets']) - (protocol['context_length'] + protocol['prediction_length'] - 1)
    pool_total = int(legal.sum())
    starts = np.cumsum(np.r_[0, legal[:-1]])
    rows, comparisons, skipped = [], [], []
    methods = ['less-main'] + ['rags-' + grain + '-' + mode
        for grain in ['block', 'traj'] for mode in ['linearSum', 'zeroClipSum', 'maxSum', 'expSum']]
    for model in protocol['students']:
        folder = args.run / 'students' / model / 'selections'
        tasks = sorted({path.name.split('__')[0] for path in folder.glob('*.npz')})
        for task in tasks:
            distributions = {}
            for method in methods:
                path = folder / (task + '__' + method + '.npz')
                if not path.exists():
                    skipped.append(dict(model=model, task=task, method=method, reason='selection_not_available'))
                    continue
                with np.load(path) as data:
                    meta = json.loads(str(data['resampling_metadata']))
                    ref = meta.get('reference_audit') or {}
                    if meta.get('competition', 'global') != 'global':
                        raise ValueError('A local-mixture diagnostic is not a final Softmax score')
                    score = np.asarray(ref.get('source_scores_for_softmax',
                        ref.get('source_scores', meta.get('native_scores', []))), dtype=float)
                    counts = np.diff(data['unit_offsets'])
                    first = data['unit_positions'][data['unit_offsets'][:-1]]
                    original = data['probabilities']
                if score.shape != counts.shape or int(counts.sum()) != pool_total:
                    raise ValueError(f'Incompatible scores or source coverage: {path}')
                trajectory = np.searchsorted(starts, first, side='right') - 1
                if len(counts) == len(legal) and not np.array_equal(counts, legal[trajectory]):
                    raise ValueError('Trajectory source geometry mismatch')
                for policy in ['equal_native_unit', 'legal_window_counts']:
                    p, audit = rank_probability(score, counts if policy == 'legal_window_counts' else None,
                        temperature=args.temperature)
                    q = counts / counts.sum() if policy == 'legal_window_counts' else np.full(len(p), 1 / len(p))
                    trajectory_mass = np.bincount(trajectory, weights=p, minlength=len(legal))
                    rank = np.asarray(audit.pop('percentiles'))
                    row = dict(model=model, task=task, method=method, policy=policy,
                        **audit, total_legal_windows=pool_total,
                        tv_to_value_deployment=float(np.abs(p - original).sum() / 2),
                        window_ess=float(1 / np.sum(p * p / counts)),
                        window_relative_ess=float(1 / (pool_total * np.sum(p * p / counts))),
                        trajectory_ess=float(1 / np.sum(trajectory_mass ** 2)),
                        top10pct_trajectory_mass=float(np.sort(trajectory_mass)[-int(np.ceil(len(legal) / 10)):].sum()))
                    rows.append(row)
                    if 'block' not in method:
                        distributions[(method, policy)] = (p, first)
                    out = args.out / model / policy
                    out.mkdir(parents=True, exist_ok=True)
                    target = out / (task + '__' + method + '.npz')
                    with target.with_suffix('.npz.partial').open('wb') as stream:
                        np.savez_compressed(stream, probabilities=p, unit_legal_counts=counts,
                            unit_first_positions=first, percentile_scores=rank,
                            metadata=json.dumps(dict(row, source_selection=str(path),
                                artifact_role='probability_overlay_not_training_result')))
                    target.with_suffix('.npz.partial').replace(target)
            for policy in ['equal_native_unit', 'legal_window_counts']:
                pair = [('less-main', policy), ('rags-traj-maxSum', policy)]
                if all(key in distributions for key in pair):
                    (a, aid), (b, bid) = [distributions[key] for key in pair]
                    if not np.array_equal(aid, bid):
                        raise ValueError('LESS/Traj source IDs differ')
                    comparisons.append(dict(model=model, task=task, policy=policy,
                        tv_less_traj_rank=float(np.abs(a - b).sum() / 2)))
    atomic_json(args.out / 'summary.json', dict(revision=REVISION,
        source_run=str(args.run), temperature=args.temperature, rows=rows,
        comparisons=comparisons, skipped=skipped,
        status='cached_score_replay_no_training', jobs_launched=0,
        caveat='Fixed percentile temperature does not guarantee equal concentration with ties or coarse units.'))
    print(json.dumps(dict(rows=len(rows), skipped=len(skipped), comparisons=comparisons), ensure_ascii=False))


if __name__ == '__main__':
    main()
