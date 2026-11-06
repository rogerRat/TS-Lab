"""Production full-pool selection entry; no subset-size or top-K parameter.

Historical hard-delivery functions remain independently callable. This entry
always takes the actual shared manifest, model identity, and calibrated CRB
configuration; source cache namespaces cannot cross models or protocols.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import time

import numpy as np

from .v3_protocol import V3_MATRIX, V3_BASES
from .reference_opinions import (OPINION_ARMS, OPINION_BASES, selection_options,
                                 probability_revision, reference_marginal, weight_placement_for)
from .arms import ArmOutput, SelectionCost
from assets.gradient.prepare import prepare_gradients
from .resampling import (REVISION, ResampledWindows, rage_global_marginal,
                         rage_marginal, rage_benchmark_marginal, random_distribution)


ARMS = ("rage", "random", "less", "tsds", "lp_mirror", "ltsv")
VARIANTS = ("rage_traj", "rage_uniform")
TEMPERATURE_ARMS = ("rage_t025", "less_t025")
GATELESS_ARMS = ("rage_gateless", "rage_traj_gls")
GLOBAL_ARMS = ("rage_global", "rage_traj_global")
VALUE_ARMS = ("lessv_t025", "lessvg_t025", "lpv_t025")
LEGACY_RANK_ARMS = ("less", "less_t025")
TEMPERATURE_DEFAULT_FORM = ("rage_gateless", "random", "lessvg_t025", "tsds", "lpv_t025", "rage_traj_gls", "rage_global", "rage_traj_global")
TEMPERATURE_ARM_BASE = dict(rage_t025="rage", less_t025="less", lessv_t025="less", lessvg_t025="less", lpv_t025="lp_mirror", rage_gateless="rage", rage_traj_gls="rage_traj", rage_global="rage", rage_traj_global="rage_traj")
TEMPERATURE_ARM_BASE.update(V3_BASES)
TEMPERATURE_ARM_BASE.update(OPINION_BASES)
RAGE_PROBABILITY_REVISION = "rage-benchmark-reference-mean-anchor-traj-v2"
COSINE_NUMERICS = "float64-norm-exact-unit-v1"


def select_full_pool(name, engine, *, reference_fragments, pool, index, pool_manifest,
                     model_identity, crb_config, cache_dir=None, temperature=.1,
                     dataset_tag="", seed=0, run_dir=None, gradient_assets=None, allow_unmatched_uniform=False,
                     reference_aggregation=None, competition=None, reference_weight_placement=None):
    """Construct the named method's frozen static source distribution."""
    if name not in ARMS + VARIANTS + TEMPERATURE_ARMS + GATELESS_ARMS + GLOBAL_ARMS + VALUE_ARMS + tuple(V3_BASES) + OPINION_ARMS:
        raise ValueError(f"Unknown full-pool arm: {name}")
    base = TEMPERATURE_ARM_BASE.get(name, name)
    options = selection_options(name)
    aggregation = reference_aggregation or options["aggregation"]
    competition = competition or options["competition"]
    weight_placement = reference_weight_placement or weight_placement_for(name)
    started = time.perf_counter()
    metadata = dict(model=model_identity, temperature=temperature, crb=asdict(crb_config), cosine_numerics=COSINE_NUMERICS,
                    task=dataset_tag, scoring_revision=REVISION,
                    opinion_aggregation=aggregation, competition=competition,
                    opinion_weight_placement=weight_placement)
    # cache_dir remains the location for non-gradient baseline resources.
    cache = Path(cache_dir) if cache_dir is not None else None
    if base in ("rage", "less", "rage_traj", "rage_uniform") and gradient_assets is None:
        if run_dir is None:
            raise ValueError("gradient assets require run_dir (outputs/<run>)")
        gradient_assets = prepare_gradients([base], engine, run_dir=run_dir,
            reference_fragments=reference_fragments, pool=pool, index=index,
            pool_manifest=pool_manifest, model_identity=model_identity,
            crb_config=crb_config, dataset_tag=dataset_tag)
    if base == "random":
        selected = random_distribution(index, pool_manifest=pool_manifest)
        output = ArmOutput(name, selected, [], SelectionCost(), {"resampling": selected.coverage()})
    elif name == 'rage_global':
        output = _rage_global(engine, reference_fragments, pool, index, pool_manifest,
                              crb_config, gradient_assets, metadata, temperature)
    elif name == 'rage_traj_global':
        output = _rage_traj_global(index, pool_manifest, crb_config, gradient_assets,
                                   metadata, temperature)
    elif base == "rage":
        output = _rage(engine, reference_fragments, pool, index, pool_manifest,
                       crb_config, gradient_assets, metadata, temperature)
    elif base == 'rage_traj':
        output = _rage_traj(index, pool_manifest, crb_config, gradient_assets, metadata, temperature)
    elif base == 'rage_uniform':
        if gradient_assets.uniform is None or gradient_assets.pairing is None:
            raise ValueError('prepare_gradients must include rage_uniform for paired assets')
        if not gradient_assets.pairing['eligible_same_compute'] and not allow_unmatched_uniform:
            raise ValueError('RAGE-Uniform actual K pairing failed; diagnostic-only output requires allow_unmatched_uniform=True')
        uniform_config = replace(crb_config, error_placement='uniform')
        output = _rage(engine, reference_fragments, pool, index, pool_manifest,
            uniform_config, gradient_assets.uniform, dict(metadata, crb=asdict(uniform_config)), temperature)
        output.name = name
        output.diagnostics['actual_K_pairing'] = gradient_assets.pairing
        output.selected.metadata['actual_K_pairing'] = gradient_assets.pairing
    else:
        common = dict(reference_fragments=reference_fragments, index=index,
                      context_length=index.context_length, prediction_length=index.prediction_length,
                      pool_manifest=pool_manifest, cache_dir=cache, resampling=metadata)
        density = dict(cover_density=crb_config.cover_density, error_density=crb_config.error_density,
                       split_density=crb_config.split_density)
        if base == "less":
            from baselines.less.runner import less_select
            output = less_select(engine, gradient_assets=gradient_assets, **common, **density, seed=seed,
                                 mapping="value", gating=False,
                                 aggregation=aggregation, competition=competition, weight_placement=weight_placement)
        elif base == "tsds":
            from baselines.tsds.runner import tsds_select
            output = tsds_select(engine, **common, **density, seed=seed)
        elif base == "lp_mirror":
            from baselines.lp_mirror.runner import lp_mirror_select
            output = lp_mirror_select(engine, **common, seed=seed,
                                      mapping="value" if name in ("lpv_t025", "lp-mirror") else "rank")
        else:
            from baselines.ltsv.runner import ltsv_select
            output = ltsv_select(engine, **common, dataset_tag=dataset_tag, seed=seed)
    if name != base:
        output.name = name
    output.cost.wall_seconds = time.perf_counter() - started
    if "source" in output.cost.notes:
        output.cost.notes["pool_cached"] = output.cost.notes["source"]["pool_cached"]
    output.diagnostics["resampling_revision"] = REVISION
    if gradient_assets is not None:
        output.diagnostics["gradient_assets"] = dict(path=str(gradient_assets.cache.path),
            computed_windows=gradient_assets.cache.computed_windows,
            format=gradient_assets.cache.manifest["format"])
    return output


def _opinion_marginal(refs, sources, weights, reference_ids, temperature, metadata):
    aggregation = metadata.get("opinion_aggregation", "linearSum")
    competition = metadata.get("competition", "global")
    if (aggregation, competition) == ("linearSum", "global"):
        return rage_benchmark_marginal(refs, sources, weights, reference_ids=reference_ids,
            temperature=temperature)
    beta, audit = reference_marginal(refs, sources, weights, aggregation=aggregation,
        competition=competition, temperature=temperature,
        weight_placement=metadata.get("opinion_weight_placement", "after_transform"))
    audit["reference_ids"] = reference_ids
    return beta, audit


def _benchmark_block_marginal(assets, temperature, metadata=None):
    metadata = metadata or {}
    legal = {int(r.fragment_id): len(r.valid_starts) for r in assets.reference_results}
    owned = {fid: 0 for fid in legal}
    ref = assets.reference_blocks
    for block in ref:
        owned[int(block.fragment_id)] += len(block.start_positions)
    if not legal or any(n <= 0 or owned[fid] != n for fid, n in legal.items()):
        raise ValueError("Benchmark RAGE requires every reference legal start to be retained")
    weights = [len(b.start_positions) / (len(legal) * legal[int(b.fragment_id)]) for b in ref]
    beta, audit = _opinion_marginal(
        [b.direction for b in ref], [b.direction for b in assets.source_blocks], weights,
        [[int(b.fragment_id), int(b.start_positions[0]), int(b.start_positions[-1])]
         for b in ref], temperature, metadata)
    audit["reference_trajectory_weights"] = [dict(fragment_id=fid, weight=1./len(legal), legal_starts=n)
                                             for fid, n in legal.items()]
    audit["reference_count"] = len(legal)
    return beta, audit


def _rage(engine, references, pool, index, manifest, config, assets, metadata, temperature):
    source, counters = assets.source_blocks, assets.source_counters
    reference = assets.reference_blocks
    results = assets.reference_results
    reference_counters = dict(gradients=sum(result.computed_gradients for result in results))
    cached = reference_cached = True
    reference_path = assets.cache.path
    ref = reference
    if not ref or not source:
        raise ValueError("CRB retained no usable reference/source blocks; full-pool distribution infeasible")
    beta, audit = _benchmark_block_marginal(assets, temperature, metadata)
    audit["reference_coverage"] = ([dict(fragment_id=int(f.fragment_id),
        legal_starts=len(result.valid_starts),
        retained_starts=sum(len(b.start_positions) for b in result.blocks),
        discarded_no_signal=result.discarded_no_signal,
        discarded_antagonistic=result.discarded_antagonistic, banned_budget=result.banned_budget)
        for f, result in zip(references, results)] if results is not None else [])
    audit["reference_cache"] = {"cached": reference_cached,
                                 "blocks": len(ref),
                                 "path": str(reference_path) if reference_path else None}
    selected = ResampledWindows.from_index(index, [b.start_positions for b in source], beta,
        pool_manifest=manifest, metadata=dict(metadata, scoring_unit="CRB_block", reference_audit=audit,
            rage_probability_revision=probability_revision(RAGE_PROBABILITY_REVISION,
                metadata.get("opinion_aggregation", "linearSum"), metadata.get("competition", "global"),
                metadata.get("opinion_weight_placement", "after_transform")), source_weighting="equal_block"))
    return ArmOutput("rage", selected, [], SelectionCost(
        gradient_computations=0,
        index_bytes=sum(b.direction.nbytes for b in source) + beta.nbytes,
        notes={"pool_cached": cached, "reference_cached": reference_cached,
               "pool": counters, "reference": reference_counters}),
        {"reference_audit": audit, "resampling": selected.coverage()})


def _rage_traj(index, manifest, config, assets, metadata, temperature):
    """Same arithmetic sampled-gradient mean as LESS; CRB chooses the anchors."""
    from baselines.less.algorithms import unit_rows
    refs, source = assets.reference_trajectories, assets.source_trajectories
    if not refs or not source:
        raise ValueError('RAGE-Traj requires reference/source anchor-mean assets')
    if len(refs) != len(assets.reference_results):
        raise ValueError('Every reference trajectory must have an anchor-mean judge')
    normalize = lambda t: unit_rows(np.asarray(t.mean_gradient)[None, :])[0]
    beta, audit = _opinion_marginal(
        [normalize(t) for t in refs], [normalize(t) for t in source],
        np.full(len(refs), 1./len(refs)), [int(t.fragment_id) for t in refs],
        temperature, metadata)
    audit["source_weighting"] = "equal_trajectory"
    audit.update(support='all-legal-trajectory-starts',
        trajectory_representation='arithmetic_mean_of_distinct_raw_CRB_anchors',
        reference_count=len(refs), reference_trajectory_weights=[1./len(refs)]*len(refs),
        source_legal_windows=sum(len(t.start_positions) for t in source),
        reference_legal_windows=sum(len(t.start_positions) for t in refs),
        source_trajectory_counts=assets.source_trajectory_counts,
        independent_gradient_requirements=int(assets.source_counters['gradients'] +
            sum(r.computed_gradients for r in assets.reference_results)))
    selected = ResampledWindows.from_index(index, [t.start_positions for t in source], beta,
        pool_manifest=manifest, metadata=dict(metadata, scoring_unit='RAGE_trajectory',
            reference_audit=audit, rage_probability_revision=probability_revision(RAGE_PROBABILITY_REVISION,
                metadata.get("opinion_aggregation", "linearSum"), metadata.get("competition", "global"),
                metadata.get("opinion_weight_placement", "after_transform")),
            source_weighting='equal_trajectory',
            trajectory_representation='arithmetic_mean_of_distinct_raw_CRB_anchors'))
    return ArmOutput('rage_traj', selected, [], SelectionCost(gradient_computations=0,
        index_bytes=sum(t.mean_gradient.nbytes for t in source) + beta.nbytes,
        notes=dict(pool_cached=True, aggregation_incremental_only=True)),
        dict(reference_audit=audit, resampling=selected.coverage()))


def _rage_global(engine, references, pool, index, manifest, config, assets, metadata, temperature):
    # The revised main block method itself uses global competition.
    output = _rage(engine, references, pool, index, manifest, config, assets, metadata, temperature)
    output.name = "rage_global"
    output.selected.metadata["scoring_unit"] = "CRB_block_global"
    return output


def _rage_traj_global(index, manifest, config, assets, metadata, temperature):
    output = _rage_traj(index, manifest, config, assets, metadata, temperature)
    output.name = 'rage_traj_global'
    output.selected.metadata['scoring_unit'] = 'RAGE_trajectory_global'
    return output
