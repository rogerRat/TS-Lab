"""V7 preflight: explicit missing-shard failure, exact pool and native input reuse."""
import json
import os
import shutil
from pathlib import Path
import numpy as np
from assets.gradient.cache import atomic_json
from .data import load_bundle_index
from .exact_pool import build_exact_pool
from .development_inputs import save_fragments, load_fragments
from .v7_protocol import MODELS, WINDOWS, crb_for, exposure_audit, VERSION
from .v7_pools import training_pool

def preflight(config):
    source = Path(config['source_load'])
    bundle = load_bundle_index(source)
    shards = sorted({int(s) for s, dropped in zip(bundle.index['shard_id'], bundle.index['dropped']) if not dropped})
    missing = [str(source/'shards'/f'values_{s:06d}.bin') for s in shards
               if not (source/'shards'/f'values_{s:06d}.bin').is_file()]
    eligible_lengths = [int(n) for n,d in zip(bundle.index['length'],bundle.index['dropped']) if not d and n >= 576]
    return dict(source_load=str(source), indexed_shards=len(shards), missing_shards=missing,
        indexed_capacity_upper_bound=sum(n//576 for n in eligible_lengths),
        capacity_note='upper bound only; finite/nonconstant validity and exact subset still required',
        exposure=exposure_audit(config), free_bytes=shutil.disk_usage('/root/autodl-tmp').free)

def prepare(config, run):
    run = Path(run)
    out = run/'inputs'
    out.mkdir(parents=True, exist_ok=True)
    audit = preflight(config)
    atomic_json(run/'input_preflight.json', audit)
    pool_marker = out/'feasible_manifest.json'
    reusable_pool=pool_marker.is_file() and (out/'feasible.npz').is_file()
    if audit['missing_shards'] and not reusable_pool:
        raise FileNotFoundError(f"V7 source incomplete: {len(audit['missing_shards'])} missing shards; "
                                +audit['missing_shards'][0])
    if pool_marker.exists():
        manifest = json.loads(pool_marker.read_text())
        if any(manifest.get(k) != v for k,v in dict(total_equivalents=65136, seed=0,
                window_length=576, source_load=config['source_load']).items()):
            raise ValueError('Existing V7 pool identity differs')
        pool = load_fragments(out/'feasible.npz')
        if ([f.length for f in pool] != manifest['lengths'] or
                [f.fragment_id for f in pool] != list(range(len(pool))) or
                len(pool) != len(manifest['rows'])):
            raise ValueError('Cached complete feasible trajectories differ from manifest')
        audit['reused_feasible_snapshot']=True
    else:
        pool, manifest = build_exact_pool(load_dir=Path(config['source_load']), equivalents=65136, seed=0)
        temporary = out/'feasible.partial.npz'
        save_fragments(temporary, pool)
        os.replace(temporary, out/'feasible.npz')
        atomic_json(pool_marker, manifest)
    if sum(f.length//576 for f in pool) != 65136:
        raise ValueError('Delivered complete trajectories do not total 65136')
    actual = {}
    actual_pools = {}
    for seed in config['seeds']:
        p, m = training_pool(pool, manifest, seed)
        path = out/f'seed{seed}_manifest.json'
        if path.exists() and json.loads(path.read_text()) != m:
            raise ValueError('Existing seed pool differs; never overwrite a pool identity')
        atomic_json(path, m)
        actual[str(seed)], actual_pools[seed] = m, p
    union_ids = {f.fragment_id for p in actual_pools.values() for f in p}
    union = [f for f in pool if f.fragment_id in union_ids]
    side = {}
    for student in MODELS:
        folder = out/student
        folder.mkdir(exist_ok=True)
        references = {}
        for task in config['datasets']:
            old = Path(config['legacy_inputs'][student])
            fields = json.loads((old/f'{task}__manifest.json').read_text())
            c,h = WINDOWS[student]
            if fields.get('context_length') != c or fields.get('prediction_length') != h or not fields.get('formal_exam_excluded_from_reference'):
                raise ValueError('Legacy native input window/isolation fields differ')
            references[task] = fields
            for suffix in ('reference.npz', 'exam.npz', 'manifest.json'):
                src, dst = old/f'{task}__{suffix}', folder/f'{task}__{suffix}'
                if dst.exists():
                    if not os.path.samefile(src, dst):
                        raise ValueError('Existing input is not the verified native resource')
                else:
                    os.link(src, dst)
        # Full reference manifests are part of this model's cache data identity.
        shared_identity = dict(v7_protocol=VERSION, feasible_pool=manifest,
            actual_training_pools=actual, reference_identities=references)
        atomic_json(folder/'gradient_data_identity.json', shared_identity)
        source_bp = sum(sum(crb_for(student).budgets(f.length-575)) for f in union)
        source_bp_by_seed={str(seed):sum(sum(crb_for(student).budgets(f.length-575)) for f in p)
                           for seed,p in actual_pools.items()}
        ref_bp = sum(sum(sum(crb_for(student).budgets(n-575)) for n in m['reference_lengths'])
                     for m in references.values())
        side[student] = dict(source_bp_per_method=source_bp, reference_bp_per_method=ref_bp,
            source_bp_by_seed=source_bp_by_seed,
            gradient_union_upper_bound=2*(source_bp+ref_bp),
            gradient_bytes_upper_bound=2*(source_bp+ref_bp)*8192,
            source_crb_bytes_upper_bound=sum(source_bp_by_seed.values())*2*4096*8)
    legal = sum(m['total_legal_windows'] for m in actual.values())
    # Conservative uncompressed baseline score/probability budget, plus CRB raw representatives.
    estimate = sum(s['gradient_bytes_upper_bound'] for s in side.values())
    estimate += legal * 16 * 4 * 3 * 9  # 3 new arms plus LP-Mirror, each seed separately
    estimate += legal * 8 * 3 * 3 * 9  # persisted descending window order for new arms
    estimate += sum(s['source_crb_bytes_upper_bound'] for s in side.values())
    estimate += 12 * 2**30  # simultaneous optimizer/checkpoint headroom; retention checked separately
    overlaps={f'{a}-{b}':len(set(actual[str(a)]['fragment_ids']) & set(actual[str(b)]['fragment_ids']))
              for a,b in ((0,1),(0,2),(1,2))}
    audit.update(feasible_pool=manifest, actual_training_pools=actual,
                 cross_seed_trajectory_overlap=overlaps,
                 cross_seed_union_trajectories=len(union), sparse_budget=side,
                 estimated_new_bytes_upper_bound=estimate,
                 free_bytes=shutil.disk_usage(out).free, complete=True)
    audit['disk_ready']=estimate <= audit['free_bytes']
    atomic_json(run/'input_preflight.json', audit)
    if estimate > audit['free_bytes']:
        raise OSError('V7 conservative disk bound exceeds free space; expand/archive with user approval')
    return audit
