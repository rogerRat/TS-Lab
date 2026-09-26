"""Adapted trajectory-level LESS: uniform window means, SGD, one checkpoint."""
from __future__ import annotations
import numpy as np
from rags.experiments.arms import ArmOutput, SelectionCost, WindowIndex
from baselines.trajectory import budget_config, trajectory_features
from rags.experiments.regions import REVISION, deliver_trajectories
from .algorithms import less_influence_scores, unit_rows
from rags.experiments.reference_opinions import reference_marginal, probability_revision


PROBABILITY_REVISION = "less-benchmark-reference-mean-v2"

def assert_probability_revision(metadata, aggregation="linearSum", competition="global", weight_placement="after_transform"):
    expected = probability_revision(PROBABILITY_REVISION, aggregation, competition, weight_placement)
    if (metadata.get("less_probability_revision") != expected
            or metadata.get("source_weighting") != "equal_trajectory"):
        raise ValueError("LESS trajectory weighting changed; use a new selection/run directory")


def less_select(engine, *, reference_fragments, index: WindowIndex,
                context_length: int, prediction_length: int, n_select: int | None = None,
                cover_density=.02, error_density=.02, split_density=.005,
                cache_dir=None, pool_manifest=None, seed=0, resampling=None, gradient_assets=None,
                mapping="value", gating=False, aggregation="linearSum", competition="global", weight_placement="after_transform") -> ArmOutput:
    config = budget_config(index, cover_density, error_density, split_density)
    source, source_info = trajectory_features(engine, index, config, kind="less",
                                             cache_dir=cache_dir, pool_manifest=pool_manifest,
                                             gradient_assets=gradient_assets)
    ref_index = WindowIndex(reference_fragments, context_length=context_length,
                            prediction_length=prediction_length, stride=index.stride)
    reference, ref_info = trajectory_features(engine, ref_index, config, kind="less",
        gradient_assets=gradient_assets,
        dataset=gradient_assets.reference_dataset if gradient_assets else "reference")
    # Normalize each trajectory mean separately, then average reference cosines.
    if not len(reference):
        raise ValueError("LESS requires reference trajectories")
    native_mass = False
    audit = None
    if (aggregation, competition) == ("linearSum", "global"):
        scores = less_influence_scores(source, reference) / len(reference)
    else:
        beta, audit = reference_marginal(unit_rows(reference), unit_rows(source),
            np.full(len(reference), 1./len(reference)), aggregation=aggregation,
            competition=competition, temperature=(resampling or {}).get("temperature", .1),
            weight_placement=weight_placement)
        native_mass = competition == "local"
        scores = beta if native_mass else np.asarray(audit["source_scores_for_softmax"])
    if resampling is not None:
        # Benchmark adaptation: each source trajectory is one competing unit.
        # Keep within-trajectory draws uniform; do not multiply mass by length.
        from rags.experiments.resampling import trajectory_distribution
        selected = trajectory_distribution(index, source_info["trajectory_ids"], scores,
            pool_manifest=pool_manifest, metadata=dict(resampling, reference_aggregation="mean_reference_cosines" if aggregation=="linearSum" else aggregation,
                opinion_aggregation=aggregation, competition=competition, opinion_weight_placement=weight_placement, reference_audit=audit, reference_count=len(reference),
                less_probability_revision=probability_revision(PROBABILITY_REVISION, aggregation, competition, weight_placement)),
            temperature=resampling.get("temperature", .1), mapping=mapping, gating=gating,
            source_weighting="equal_trajectory", native_mass=native_mass)
        if native_mass:
            selected.metadata.update(probability_mapping="weighted_reference_row_softmax_mixture",
                                     temperature=resampling.get("temperature", .1))
        return ArmOutput("less", selected, [], SelectionCost(
            gradient_computations=source_info["executed_windows"] + ref_info["executed_windows"],
            index_bytes=source.nbytes, notes={"source": source_info, "reference": ref_info}),
            {"source_sampling": source_info, "reference_sampling": ref_info,
             "resampling": selected.coverage()})
    order = np.lexsort((np.random.default_rng(5).random(len(scores)), -scores))
    fids = np.asarray(source_info["trajectory_ids"])
    selected = deliver_trajectories(index, fids[order], n=n_select)
    return ArmOutput(
        name="less", selected=selected, window_ids=[],
        cost=SelectionCost(
            gradient_computations=source_info["executed_windows"] + ref_info["executed_windows"],
            index_bytes=source.nbytes,
            notes={"pool_cached": source_info["pool_cached"],
                   "pool_gradients": source_info["executed_windows"],
                   "pool_feature_windows": source_info["sampled_windows"],
                   "reference_gradients": ref_info["executed_windows"]}),
        diagnostics={"selection_revision": REVISION, "delivery": selected.coverage(),
                     "source_sampling": source_info, "reference_sampling": ref_info,
                     "trajectory_scores": scores.tolist(),
                     "adaptation": "trajectory uniform-sample mean; bilateral SGD full-parameter "
                                   "Count-Sketch projections; single pretrained checkpoint; "
                                   f"{aggregation}/{competition}; whole trajectory point-budget delivery"})
