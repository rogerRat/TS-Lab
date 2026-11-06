"""Two explicit modes: sparse preparation, or fail-closed reconstruction with zero BP."""
from dataclasses import asdict
from functools import partial
import time
import numpy as np
from assets.gradient.cache import index_request
from assets.gradient.prepare import prepare_gradients
from assets.gradient.variants import trajectory_counts
from .arms import WindowIndex
from .v7_cache import V7Cache
from .v7_scores import uniform_anchors

def uniform_plan(index, counts):
    rows = []
    if set(index.fragment_lengths) != {r['fragment_id'] for r in counts}:
        raise ValueError('RAGE BP audit must cover every trajectory')
    for row in counts:
        fid = row['fragment_id']
        lo, hi = index.bounds(fid)
        k = row['gradients']
        if k != sum(row[s] for s in ('cover', 'error', 'split')):
            raise ValueError('Actual distinct BP does not match phase audit')
        p = uniform_anchors(hi-lo, k)
        rows.append(dict(row, positions=p.tolist(), global_offset=lo,
            uniform_actual_bp=len(p), endpoint_priority='left_if_K1_else_both'))
    return rows

def prepare(*, probe, run_dir, pool, index, refs, manifest, identity, crb, task,
            donors=(), readonly=False, shared_data=None):
    started = time.perf_counter()
    factory = partial(V7Cache, donors=donors, readonly=readonly, shared_data=shared_data)
    assets = prepare_gradients(['less', 'rage', 'rage_traj'], probe, run_dir=run_dir,
        reference_fragments=refs, pool=pool, index=index, pool_manifest=manifest,
        model_identity=identity, crb_config=crb, dataset_tag=task, cache_factory=factory)
    old_cache = assets.cache
    ref_index = WindowIndex(refs, context_length=index.context_length, prediction_length=index.prediction_length)
    plans = dict(source=uniform_plan(index, assets.source_trajectory_counts),
                 reference=uniform_plan(ref_index, [trajectory_counts(r) for r in assets.reference_results]))
    cache = factory(run_dir, probe, model_identity=identity, data_fields=dict(pool=manifest),
        context_length=index.context_length, prediction_length=index.prediction_length)
    try:
        for side, current, dataset in (('source', index, 'source'),
                                       ('reference', ref_index, assets.reference_dataset)):
            positions = np.concatenate([np.array(r['positions']) + r['global_offset'] for r in plans[side]])
            cache.prepare([index_request(dataset, current, positions)])
            # Audit finite BF16 values and norms without scanning non-budget windows.
            min_norm, max_norm = float('inf'), 0.
            for begin in range(0, len(positions), 1024):
                g = cache.get(index_request(dataset, current, positions[begin:begin+1024]))
                if g.ndim != 2 or g.shape[1] != 4096 or not np.isfinite(g).all():
                    raise ValueError('V7 requires finite single-checkpoint 4096-D gradients')
                norm = np.linalg.norm(g.astype(np.float64), axis=1)
                min_norm, max_norm = min(min_norm, float(norm.min())), max(max_norm, float(norm.max()))
            for row in plans[side]:
                if row['gradients'] != sum(crb.budgets(row['legal_windows'])):
                    raise ValueError('RAGE did not exhaust capped per-trajectory stage budgets')
            plans[side+'_audit'] = dict(required_bp=len(positions), legal_windows=len(current),
                coverage=len(positions)/len(current), norm_min=min_norm, norm_max=max_norm, nonfinite=0)
        new_bp = old_cache.computed_windows + cache.computed_windows
        imported = old_cache.imported_windows + cache.imported_windows
        if readonly and new_bp:
            raise RuntimeError('Selection performed additional BP')
        cache.freeze()
    except BaseException:
        cache.close()
        raise
    assets.cache = cache
    assets.selection_aggregations = {family: ['linearSum','zeroClipSum','maxSum','expSum']
                                    for family in ('rage','rage_traj')}
    source_ids={f.fragment_id for f in pool}
    required_union=sum(1 for ds,fid,_ in cache.rows if
        (ds=='source' and fid in source_ids) or ds==assets.reference_dataset)
    audit = dict(model=identity, pool=manifest, crb=asdict(crb), task=task,
        gradient_manifest=cache.manifest, path=str(cache.path), plans=plans,
        method_costs=assets.method_costs, new_bp=new_bp, imported_rows=imported,
        prepared_union_rows=required_union, cached_rows=required_union-new_bp-imported,
        gradient_seconds=old_cache.computed_gradient_seconds+cache.computed_gradient_seconds,
        elapsed_seconds=time.perf_counter()-started, reuse_audit=cache.reuse_audit,
        complete=True, scope='sparse_RAGE_LESS_uniform_union_not_full_window_scan')
    return assets, ref_index, audit
