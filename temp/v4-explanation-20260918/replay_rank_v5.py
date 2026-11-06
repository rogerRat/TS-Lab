"""Replay prepared order keys only; no BP, model or training imports."""
import argparse
import json
from pathlib import Path

import numpy as np

from rage.experiments.rank_v5 import REVISION, DEFAULT_TEMPERATURE, rank_mass_v5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank-replay', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Use a new output directory')
    args.out.mkdir(parents=True)
    rows = []
    for model in ['chronos-base', 'timesfm-2.5']:
        for source in sorted((args.rank_replay / model / 'equal_native_unit').glob('*.npz')):
            task, method = source.stem.split('__')
            if method != 'less-main' and not method.startswith('rage-block-'):
                continue
            with np.load(source) as data:
                # This key is strictly monotone in the cached final score,
                # preserving every exact tie. No scores need recomputation.
                key = data['percentile_scores']
                counts = data['unit_legal_counts']
                first = data['unit_first_positions']
                old_meta = json.loads(str(data['metadata']))
            rank, final, audit = rank_mass_v5(key, counts, temperature=args.temperature)
            row = dict(model=model, task=task, method=method, **audit,
                source_original_selection=old_meta['source_selection'],
                order_source=str(source),
                artifact_role='cached_order_probability_replay_not_training_result',
                gradient_computations=0, jobs_launched=0)
            rows.append(row)
            out = args.out / model
            out.mkdir(exist_ok=True)
            path = out / source.name
            with path.with_suffix('.npz.partial').open('wb') as stream:
                np.savez_compressed(stream, rank_mass=rank, probabilities=final,
                    unit_legal_counts=counts, unit_first_positions=first,
                    metadata=json.dumps(row))
            path.with_suffix('.npz.partial').replace(path)
    summary = dict(revision=REVISION, temperature=args.temperature, rows=rows,
        gradient_computations=0, jobs_launched=0,
        status='cached_order_replay_no_v5_training')
    partial = args.out / 'summary.json.partial'
    partial.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    partial.replace(args.out / 'summary.json')
    print(json.dumps(dict(rows=len(rows), temperature=args.temperature,
        half_mass_min=min(r['rank_top_half_mass_fractional_boundary'] for r in rows),
        half_mass_max=max(r['rank_top_half_mass_fractional_boundary'] for r in rows),
        boundary_tie_rows=sum(r['half_boundary_inside_tie_group'] for r in rows))))


if __name__ == '__main__':
    main()
