"""Whole-trajectory exact quarter-pools. IDs never change between seeds."""
import numpy as np
from .exact_pool import ExactEquivalents

def training_pool(feasible, manifest, seed, *, target=16284):
    if len(feasible) != len(manifest['rows']) or len({f.fragment_id for f in feasible}) != len(feasible):
        raise ValueError('Feasible pool rows/IDs must be unique and aligned')
    if sum(f.length//576 for f in feasible) != manifest['total_equivalents']:
        raise ValueError('Feasible pool equivalent count differs')
    search = ExactEquivalents(target)
    for row in np.random.default_rng(seed).permutation(len(feasible)):
        if search.add(int(row), feasible[row].length//576):
            break
    selected = search.selected_rows()  # explicit failure if exact N is unreachable
    # Stable physical order, independent of randomized search/insertion order.
    selected.sort()
    pool = [feasible[i] for i in selected]
    result = dict(pool_protocol='v7-exact-whole-trajectory-seed-quarter-v1',
        feasible_pool=manifest, seed=int(seed), total_equivalents=sum(f.length//576 for f in pool),
        equivalents_target=target, window_length=576, source_load=manifest['source_load'],
        rows=[manifest['rows'][i] for i in selected],
        fragment_ids=[int(f.fragment_id) for f in pool], lengths=[f.length for f in pool],
        total_points=sum(f.length for f in pool), total_legal_windows=sum(f.length-575 for f in pool),
        equivalent_fraction=target/manifest['total_equivalents'],
        trajectory_fraction=len(pool)/len(feasible), cross_seed_overlap_allowed=True)
    if result['total_equivalents'] != target:
        raise ValueError('Training pool must exactly meet the exposure target')
    return pool, result
