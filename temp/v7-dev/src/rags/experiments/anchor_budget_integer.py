"""Exhaust positive integer budget totals on completed Ref caches, CPU only."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from rags.crb import CRBConfig, CRBEngine
from .anchor_budget_simulate import (
    ArrayGradientComputer, FROZEN, atomic_write, check_rows_complete,
    completed_keys, load_entry_fragment, load_gradient_dataset,
    load_sample_manifest, manifest_entries,
)


def positive_integer_budgets(total):
    if not isinstance(total, int) or not 3 <= total <= 30:
        raise ValueError('Integer total must be between 3 and 30 percent')
    return [(cover, error, total-cover-error)
            for cover in range(1, total-1) for error in range(1, total-cover)]


def unit_prefix(truth64, norms):
    unit = np.divide(truth64, norms[:, None], out=np.zeros_like(truth64),
                     where=norms[:, None] > 0)
    prefix = np.empty((len(unit)+1, unit.shape[1]), dtype=np.float64)
    prefix[0] = 0
    np.cumsum(unit, axis=0, out=prefix[1:])
    return prefix


def mean_block_cosine(truth64, norms, result, prefix=None):
    """Exact block-constant metric via sums of per-window unit gradients.

    Dropped and zero-norm windows contribute zero to the original denominator.
    This does not substitute a cosine of the trajectory's mean gradient.
    """
    n = len(truth64)
    if n != len(result.valid_starts) or n == 0:
        raise ValueError('Truth must cover every valid window')
    if n == 1:
        gradient = result.anchors[0].astype(np.float64)
        norm = float(np.linalg.norm(gradient))
        return float(truth64[0] @ gradient / (norms[0]*norm)) if norms[0] > 0 and norm else 0.
    prefix = unit_prefix(truth64, norms) if prefix is None else prefix
    total = 0.
    for block in result.blocks:
        start, stop = int(block.left_index), int(block.right_index)
        if stop == n-1:
            stop += 1
        if not np.array_equal(block.start_positions, result.valid_starts[start:stop]):
            raise ValueError('Block ownership differs from valid-start indexing')
        direction = block.mean_gradient.astype(np.float64)
        norm = float(np.linalg.norm(direction))
        if norm:
            total += float((prefix[stop]-prefix[start]) @ (direction/norm))
    return total/n


def main():
    raise RuntimeError("Retired block-constant budget protocol: use anchor_budget_linear with a matching frozen p10 calibration; historical budgets are invalid for window reconstruction")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'evaluations.jsonl').exists():
        raise ValueError('Use a new output directory; existing search evidence is preserved')
    manifest = load_sample_manifest(args.run_root)
    entries = manifest_entries(manifest, {0})
    done = completed_keys(args.run_root)
    if len(entries) != 5 or any((e['task'], e['fragment_id'], e['wave']) not in done for e in entries):
        raise ValueError('All five rank-0 trajectories must be complete')
    dataset = load_gradient_dataset(manifest['cache'])
    check_rows_complete(dataset, entries)
    state = {}
    # Historical hard examples first. Early rejection changes cost, not the
    # feasible candidate set: every triple at every lower total is checked.
    order = ['time_darwin_water_15t', 'time_finland_15t', 'time_nem_demand_5t',
             'time_azure_i_cpu_5t', 'time_eweld_15t']
    for entry in entries:
        fragment = load_entry_fragment(entry)
        starts = fragment.valid_starts(512,64)
        ids = np.column_stack([np.full(len(starts), entry['fragment_id']), starts])
        truth = dataset.lookup(entry['dataset'], ids)
        truth64 = truth.astype(np.float64)
        norms = np.linalg.norm(truth64, axis=1)
        state[entry['task']] = dict(fragment=fragment, truth64=truth64, norms=norms,
                                   prefix=unit_prefix(truth64,norms),
                                   computer=ArrayGradientComputer(starts,truth,entry['fragment_id']))
    caches = {task:{} for task in order}
    started = time.monotonic()
    checked = actual_calls = 0
    frontiers = []
    final_passes = []
    atomic_write(args.out/'protocol.json', dict(samples=entries, minimum_each_percent=1,
        maximum_total_percent=30, threshold=.9, frozen=FROZEN, cache=manifest['cache'],
        order=order, criterion='all five per-window block-constant mean cosines >= .9',
        split_reuse='only after actual CRB reports zero split, banned and antagonistic intervals',
        precision='BF16 cached gradients, FP64 metric', model_calls=0))
    with (args.out/'evaluations.jsonl').open('w', buffering=1) as log:
        for total in range(3,31):
            frontier = None
            passing = []
            for cover,error,split in positive_integer_budgets(total):
                checked += 1
                results = []
                for task in order:
                    data = state[task]
                    pair = (cover,error)
                    key = (cover,error,split)
                    cached = caches[task].get(pair) or caches[task].get(key)
                    if cached is None:
                        config = CRBConfig(cover_density=cover/100, error_density=error/100,
                                           split_density=split/100, **FROZEN)
                        result = CRBEngine(config,data['computer']).process_fragment(data['fragment'])
                        actual_calls += 1
                        cached = dict(task=task, mean_cosine=mean_block_cosine(
                            data['truth64'],data['norms'],result,data['prefix']),
                            anchors=len(result.anchors), windows=len(result.valid_starts),
                            split_used=result.split_computations,
                            split_invariant=(result.split_computations==0 and result.banned_budget==0
                                             and result.discarded_antagonistic==0))
                        caches[task][pair if cached['split_invariant'] else key] = cached
                    results.append(cached)
                    if cached['mean_cosine'] < .9:
                        break
                actual_usable = (sum(x['anchors'] for x in results)*10 <=
                                 sum(x['windows'] for x in results)*3)
                row = dict(cover=cover,error=error,split=split,total=total,
                           pass_all=len(results)==5 and all(x['mean_cosine']>=.9 for x in results)
                                    and actual_usable,
                           results=results, early_rejected=len(results)<5)
                log.write(json.dumps(row)+'\n')
                if row['pass_all']:
                    passing.append(row)
                # For rejected totals this is a bottleneck-only bound, not a
                # fabricated measurement of untested tasks.
                score = min(x['mean_cosine'] for x in results)
                if frontier is None or score > min(x['mean_cosine'] for x in frontier['results']):
                    frontier = row
            frontiers.append(frontier)
            progress = dict(total_percent=total,checked_triples=checked,actual_crb_calls=actual_calls,
                            elapsed_seconds=time.monotonic()-started,passing_count=len(passing),
                            best_measured_bound=frontier)
            atomic_write(args.out/'status.json',progress)
            print(json.dumps(progress),flush=True)
            if passing:
                final_passes = passing
                break
    # Same minimum nominal total: highest worst-task cosine, then actual cost.
    final_passes.sort(key=lambda x:(-min(y['mean_cosine'] for y in x['results']),
                                    sum(y['anchors'] for y in x['results']),
                                    x['cover'],x['error'],x['split']))
    summary = dict(complete=True,minimum_total_percent=final_passes[0]['total'] if final_passes else None,
                   recommendation=final_passes[0] if final_passes else None,
                   passing_at_minimum=final_passes,frontiers=frontiers,
                   checked_triples=checked,actual_crb_calls=actual_calls,
                   elapsed_seconds=time.monotonic()-started,model_calls=0,
                   scope='five rank-0 samples, positive integer percentages, fixed CRB thresholds; not full sample validation')
    atomic_write(args.out/'summary.json',summary)
    print('INTEGER_SEARCH_COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
