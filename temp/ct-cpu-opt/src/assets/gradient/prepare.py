"""Prepare the union of static method requirements before weighting/training."""
from pathlib import Path
from dataclasses import dataclass, replace, field, asdict

import time

from rags.crb import CRBEngineMulti
from .cache import GradientCache, FragmentGradients, index_request


@dataclass
class PreparedGradients:
    cache: GradientCache
    reference_dataset: str
    source_blocks: object = None
    reference_blocks: object = None
    source_counters: object = None
    reference_results: object = None
    source_trajectories: object = None
    reference_trajectories: object = None
    source_trajectory_counts: object = None
    uniform: object = None
    pairing: object = None
    method_costs: dict = field(default_factory=dict)
    selection_aggregations: dict = field(default_factory=dict)
    selection_scores: dict = field(default_factory=dict)


def prepare_gradients(names, engine, *, run_dir, reference_fragments, pool, index,
                      pool_manifest, model_identity, crb_config, dataset_tag=""):
    """Uniform requirements are merged first; adaptive CRB rounds extend the union.

    No model update occurs here. The returned cache is frozen: scoring can only
    read prepared rows, never silently trigger model work.
    """
    from baselines.trajectory import trajectory_positions
    from rags.experiments.arms import WindowIndex
    from .blocks import compute_pool_blocks
    from .source_crb_cache import source_cache_directory
    reference_dataset = "reference:" + dataset_tag
    # Reference task IDs and fragment geometry accompany each task namespace;
    # the run's immutable inputs/protocol identify the actual data snapshot.
    cache = GradientCache(run_dir, engine, model_identity=model_identity,
        data_fields=dict(pool=pool_manifest), context_length=index.context_length,
        prediction_length=index.prediction_length)
    reuse_marker = Path(run_dir) / "gradient_reuse.json"
    if reuse_marker.exists():
        import json
        verification = json.loads((Path(run_dir) / "warm_cache_verification.json").read_text())
        expected_paths = {Path(row["cache"]).resolve() for row in verification["reports"]}
        if not cache.rows or cache.path.resolve() not in expected_paths:
            cache.close()
            raise ValueError("Required existing gradient cache did not match; refusing full recomputation")
    prepared = PreparedGradients(cache, reference_dataset)
    print(f"[gradient assets] task={dataset_tag} epsilon_policy={crb_config.epsilon_policy} "
          f"cached_windows={len(cache.rows)} path={cache.path}", flush=True)
    try:
        from .cache import atomic_json, readable
        marker = cache.path / ("reference-" + readable(dataset_tag) + ".json")
        fields = dict(task=dataset_tag, fragments=[dict(fragment_id=int(f.fragment_id),
            entity_id=str(f.entity_id), length=int(f.length)) for f in reference_fragments])
        import json
        if marker.exists() and json.loads(marker.read_text()) != fields:
            raise ValueError("reference fields changed; use a new run")
        atomic_json(marker, fields)
        def begin_cost():
            return time.perf_counter(), cache.computed_windows, cache.computed_gradient_seconds

        def end_cost(before, required_source, required_reference):
            elapsed = time.perf_counter() - before[0]
            gradient = cache.computed_gradient_seconds - before[2]
            return dict(required_source=int(required_source), required_reference=int(required_reference),
                preparation_new_windows=cache.computed_windows-before[1],
                preparation_seconds=elapsed, preparation_gradient_seconds=gradient,
                method_preparation_nongrad_seconds=max(0., elapsed-gradient),
                preparation_scope='method_delta_in_shared_union')

        less_before = begin_cost()
        requests = []
        ref_index = None
        for name in names:
            if name in ("less",):
                if ref_index is None:
                    ref_index = WindowIndex(reference_fragments,
                        context_length=index.context_length, prediction_length=index.prediction_length,
                        stride=index.stride)
                requests.extend([
                    index_request("source", index, trajectory_positions(index, crb_config)),
                    index_request(reference_dataset, ref_index, trajectory_positions(ref_index, crb_config))])
        cache.prepare(requests)
        if 'less' in names:
            # Deduplicate native requirements just like cache.prepare does.
            required = {side: set() for side in ('source', reference_dataset)}
            for request in requests:
                required[request.dataset].update(map(tuple, request.ids.tolist()))
            prepared.method_costs['less'] = end_cost(less_before,
                len(required['source']), len(required[reference_dataset]))
        if any(name in names for name in ("rags", "rags_traj", "rags_uniform")):
            rags_before = begin_cost()
            trajectory_assets = dict(trajectories=[], counts=[])
            prepared.source_blocks, prepared.source_counters = compute_pool_blocks(
                pool, index, config=crb_config, gradient_computer=FragmentGradients(cache, "source"),
                source_cache_dir=source_cache_directory(cache, crb_config, pool, index),
                trajectory_assets=trajectory_assets)
            prepared.reference_results = CRBEngineMulti(crb_config,
                FragmentGradients(cache, reference_dataset)).process_fragments(reference_fragments)
            prepared.reference_blocks = [b for result in prepared.reference_results for b in result.blocks]
            audit_root = Path(run_dir) / "crb_audits" / readable(model_identity["model_id"])
            audit_root.mkdir(parents=True, exist_ok=True)
            atomic_json(audit_root / ("crb-splits-" + readable(dataset_tag) + ".json"),
                dict(schema="min-kappa-split-v5", model=model_identity, crb=asdict(crb_config),
                     source=prepared.source_counters['split_audits'],
                     reference=[r.split_audit for r in prepared.reference_results]))
            from .variants import aggregate_anchors, trajectory_counts, pair_counts
            prepared.source_trajectories = trajectory_assets['trajectories']
            prepared.source_trajectory_counts = trajectory_assets['counts']
            prepared.reference_trajectories = [a for r in prepared.reference_results
                if (a := aggregate_anchors(r, expected_count=sum(crb_config.budgets(len(r.valid_starts))))) is not None]
            prepared.method_costs['rags'] = end_cost(rags_before, prepared.source_counters['gradients'],
                sum(r.computed_gradients for r in prepared.reference_results))
            prepared.method_costs['rags_traj'] = dict(prepared.method_costs['rags'],
                preparation_scope='shared_complete_RAGS_preparation_including_compact_aggregation')
            if 'rags_uniform' in names:
                uniform_before = begin_cost()

                uniform_config = replace(crb_config, error_placement='uniform')
                uniform = PreparedGradients(cache, reference_dataset)
                uniform_trajectories = dict(trajectories=[], counts=[])
                uniform.source_blocks, uniform.source_counters = compute_pool_blocks(
                    pool, index, config=uniform_config, gradient_computer=FragmentGradients(cache, 'source'),
                    source_cache_dir=source_cache_directory(cache, uniform_config, pool, index),
                    trajectory_assets=uniform_trajectories)
                uniform.reference_results = CRBEngineMulti(uniform_config,
                    FragmentGradients(cache, reference_dataset)).process_fragments(reference_fragments)
                uniform.reference_blocks = [b for r in uniform.reference_results for b in r.blocks]
                uniform.source_trajectory_counts = uniform_trajectories['counts']
                prepared.uniform = uniform
                prepared.pairing = pair_counts(prepared.source_trajectory_counts,
                    uniform.source_trajectory_counts,
                    [trajectory_counts(r) for r in prepared.reference_results],
                    [trajectory_counts(r) for r in uniform.reference_results])
                prepared.method_costs['rags_uniform'] = end_cost(uniform_before,
                    uniform.source_counters['gradients'], sum(r.computed_gradients for r in uniform.reference_results))
        cache.freeze()
        return prepared
    except BaseException:
        cache.close()
        raise
