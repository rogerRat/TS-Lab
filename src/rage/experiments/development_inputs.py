"""Prepare real-data native selection inputs without importing data readers in model environments."""
import argparse
import json
from pathlib import Path

import numpy as np

from rage.core import TimeSeriesFragment
from .students import atomic_json


def save_fragments(path, fragments):
    lengths = [len(f.data) for f in fragments]
    np.savez_compressed(path, values=np.concatenate([f.data for f in fragments]),
        offsets=np.cumsum([0] + lengths), ids=np.array([f.fragment_id for f in fragments]),
        entities=np.array([f.entity_id for f in fragments]))


def load_fragments(path):
    with np.load(path) as d:
        # NPZ indexing decompresses on every access; materialize the packed
        # values once, not once per trajectory in the full source pool.
        values, offsets, entities, ids = d['values'], d['offsets'], d['entities'], d['ids']
    return [TimeSeriesFragment(values[lo:hi].copy(), entity_id=str(entity), fragment_id=int(fid))
            for lo, hi, entity, fid in zip(offsets[:-1], offsets[1:], entities, ids)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--pool-cache', required=True)
    parser.add_argument('--source-trajectories', type=int, default=64)
    parser.add_argument('--reference-tail-points', type=int, default=2048)
    args = parser.parse_args()
    from .tasks import build_ratio_pool, build_task_specs, build_task_data
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pool, manifest = build_ratio_pool(context_length=512, prediction_length=64,
        equivalents=20480, seed=0, cache_dir=Path(args.pool_cache))
    rows = np.random.default_rng(20260913).choice(len(pool), args.source_trajectories, replace=False)
    chosen = [pool[i] for i in rows]
    save_fragments(out / 'pool.npz', chosen)
    protocol = dict(arguments=vars(args), source_entities=[f.entity_id for f in chosen],
        pool_points=sum(len(f.data) for f in chosen), source_sampling='uniform trajectories, target independent',
        validation_gap_horizons=16, validation_horizons=8, context=512, horizon=64, references={})
    for spec in build_task_specs(keywords=('boomlet_1676', 'boomlet_619')):
        task = build_task_data(spec, context_length=512, prediction_length=64, pool_fragments=[])
        refs, bounds = [], []
        for f in task.reference_fragments:
            stop = len(f.data) - (16 + 8) * 64 - 512
            start = max(0, stop - args.reference_tail_points)
            if stop - start < 576:
                raise ValueError(f'Insufficient reference prefix: {f.entity_id}')
            refs.append(TimeSeriesFragment(f.data[start:stop].copy(), entity_id=f.entity_id, fragment_id=f.fragment_id))
            bounds.append(dict(entity=f.entity_id, start=start, stop=stop))
        save_fragments(out / (spec.dataset + '.npz'), refs)
        protocol['references'][spec.dataset] = bounds
    atomic_json(out / 'protocol.json', protocol)
    print('REAL_DEVELOPMENT_INPUTS_READY', protocol['pool_points'], flush=True)


if __name__ == '__main__':
    main()
