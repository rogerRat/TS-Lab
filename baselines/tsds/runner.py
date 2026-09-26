"""Trajectory-level TSDS adaptation with upstream IVF/KDE/OT kernels."""
from __future__ import annotations
import numpy as np
from rags.experiments.arms import ArmOutput, SelectionCost, WindowIndex
from baselines.trajectory import budget_config, trajectory_features
from rags.experiments.regions import REVISION, deliver_trajectories
from .algorithms import tsds_probability_assignment
from .retrieval import trajectory_neighbours


def tsds_select(engine, *, reference_fragments, index: WindowIndex,
                context_length: int, prediction_length: int, n_select: int | None = None,
                cover_density=.02, error_density=.02, split_density=.005,
                sample_size=None, max_k=64, kde_k=16, sigma=4., alpha=.1, c_param=5.,
                cache_dir=None, pool_manifest=None, seed=0, resampling=None) -> ArmOutput:
    config = budget_config(index, cover_density, error_density, split_density)
    source, source_info = trajectory_features(engine, index, config, kind="tsds",
                                             cache_dir=cache_dir, pool_manifest=pool_manifest)
    ref_index = WindowIndex(reference_fragments, context_length=context_length,
                            prediction_length=prediction_length, stride=index.stride)
    reference, ref_info = trajectory_features(engine, ref_index, config, kind="tsds")
    distances, neighbours, kdes, retrieval = trajectory_neighbours(
        source, reference, max_k=max_k, kde_k=kde_k, sigma=sigma)
    probabilities = tsds_probability_assignment(
        distances, neighbours, kdes, candidate_count=len(source), alpha=alpha, c_param=c_param)
    if resampling is not None:
        from rags.experiments.resampling import trajectory_distribution
        selected = trajectory_distribution(index, source_info["trajectory_ids"], probabilities,
            pool_manifest=pool_manifest, metadata=resampling, native_mass=True)
        return ArmOutput("tsds", selected, [], SelectionCost(
            feature_computations=source_info["executed_windows"] + ref_info["executed_windows"],
            index_bytes=source.nbytes, notes={"source": source_info, "reference": ref_info, **retrieval}),
            {"source_sampling": source_info, "reference_sampling": ref_info,
             "parameters": dict(max_k=max_k, kde_k=kde_k, sigma=sigma, alpha=alpha, c=c_param),
             "parameter_origin": "experiment_frozen_not_upstream_defaults",
             "resampling": selected.coverage(), **retrieval})
    # Upstream NumPy multinomial RNG, local instance so other arms are unaffected.
    counts = np.random.RandomState(seed).multinomial(sample_size or n_select, probabilities)
    # Preserve repeated trajectories; uniformly order copies before the new
    # whole-unit budget cutoff, avoiding upstream numeric-ID prefix bias.
    copies = np.repeat(np.asarray(source_info["trajectory_ids"]), counts)
    np.random.default_rng(seed).shuffle(copies)
    selected = deliver_trajectories(index, copies, n=n_select)
    return ArmOutput(
        name="tsds", selected=selected, window_ids=[],
        cost=SelectionCost(
            feature_computations=source_info["executed_windows"] + ref_info["executed_windows"],
            index_bytes=source.nbytes,
            notes={"pool_cached": source_info["pool_cached"],
                   "pool_feature_windows": source_info["sampled_windows"],
                   "pool_features": source_info["executed_windows"],
                   "reference_features": ref_info["executed_windows"], **retrieval}),
        diagnostics={"selection_revision": REVISION, "delivery": selected.coverage(),
                     "parameters": {"max_k": max_k, "kde_k": kde_k, "sigma": sigma,
                                    "alpha": alpha, "c": c_param,
                                    "sample_size": sample_size or n_select, "seed": seed},
                     "parameter_origin": "experiment_frozen_not_upstream_defaults",
                     "source_sampling": source_info, "reference_sampling": ref_info,
                     "trajectory_probabilities": probabilities.tolist(),
                     "trajectory_sample_counts": counts.tolist(), **retrieval,
                     "deployment": "shuffled_trajectory_copies_until_point_budget",
                     "probe_model": str(getattr(engine.config, "model_path", "unspecified")),
                     "adaptation": "trajectory uniform-sample mean of frozen probe context "
                                   "features; upstream IVF/KDE/OT/multinomial repeats; "
                                   "whole repeated trajectories until point budget; uniform legal "
                                   "window occurrence sampling during training"})
