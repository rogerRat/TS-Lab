"""Opt-in rank deployment; never changes frozen value-score protocols.

Reference aggregation produces the scores before this function is called.
Legal-window counts are a data-volume prior, not trajectory equalization.
"""
from __future__ import annotations

import numpy as np

REVISION = "weighted-midrank-fixed-temperature-v1"


def rank_probability(scores, counts=None, *, temperature=.05):
    """Return mass and audit from tie-preserving weighted percentile ranks.

    counts=None retains equal native-unit priors for a pure mapping control.
    Otherwise counts are the number of distinct owned legal window starts.
    Exact ties share one percentile and receive mass proportional to counts.
    All finite scores participate; there is no sign gate or top-k truncation.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Scores must be a nonempty finite vector")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    base = np.ones(len(scores)) if counts is None else np.asarray(counts, dtype=np.float64)
    if base.shape != scores.shape or not np.isfinite(base).all() or np.any(base <= 0):
        raise ValueError("Counts must be matching positive finite values")
    # Scaling first avoids overflow in the total for large finite counts.
    q = base / base.max()
    q /= q.sum()
    if np.any(q == 0):
        raise ValueError("Source prior underflows")
    _, inverse = np.unique(scores, return_inverse=True)
    group_q = np.bincount(inverse, weights=q)
    upper = np.cumsum(group_q)
    percentile = (upper - .5 * group_q)[inverse]
    logits = (percentile - percentile.max()) / temperature + np.log(q)
    p = np.exp(logits - logits.max())
    p /= p.sum()
    positive = p > 0
    rho = float(1. / np.sum(p * (p / q)))
    audit = dict(rank_probability_revision=REVISION, temperature=float(temperature),
        source_prior="equal_native_unit" if counts is None else "legal_window_counts",
        ranking="ascending_weighted_midpercentile", tie_policy="exact_group_proportional_to_prior",
        gating=False, score_role="final_pre_outer_softmax_score",
        units=len(scores), unique_scores=len(group_q), largest_tie_prior=float(group_q.max()),
        relative_ess=rho, kl_to_prior=float(np.sum(p[positive] * np.log(p[positive] / q[positive]))),
        percentiles=percentile.tolist())
    return p, audit


def rank_resampled_windows(selected, *, temperature=.05, source_prior="legal_window_counts"):
    """Build an independent trainable view; reuse source arrays, not its mass.

    The caller must save/use a new run identity. Frozen V4 never calls this.
    This maps final global scores, not local-mixture diagnostic scores or OT mass.
    """
    from .resampling import ResampledWindows
    if source_prior not in ("legal_window_counts", "equal_native_unit"):
        raise ValueError("Unknown source prior")
    if selected.metadata.get('competition', 'global') != 'global':
        raise ValueError("Local-mixture diagnostics are not final Softmax scores")
    if selected.metadata.get('rank_deployment'):
        raise ValueError("Selection is already rank deployed")
    ref = selected.metadata.get('reference_audit') or {}
    scores = ref.get('source_scores_for_softmax', ref.get('source_scores'))
    if scores is None:
        scores = selected.metadata.get('native_scores')
    if scores is None:
        raise ValueError("Final selection scores are required; do not rank native probability mass")
    counts = np.diff(selected.unit_offsets)
    if np.asarray(scores).shape != counts.shape:
        raise ValueError("Final scores do not match source units")
    p, audit = rank_probability(scores, counts if source_prior == 'legal_window_counts' else None,
        temperature=temperature)
    meta = dict(selected.metadata,
        rank_deployment=audit, probability_mapping=REVISION,
        pre_rank_probability_mapping=selected.metadata.get('probability_mapping'),
        pre_rank_temperature=selected.metadata.get('temperature'),
        temperature=temperature,
        source_weighting=source_prior if source_prior == 'legal_window_counts'
            else selected.metadata.get('source_weighting', 'equal_native_unit'))
    # Ensure legacy cache checks cannot accept this as a value-score selection.
    for key in ('less_probability_revision', 'rage_probability_revision'):
        if key in meta:
            meta[key] += ':' + REVISION + ':' + source_prior
    result = ResampledWindows(selected.values, selected.window_locations, selected.window_ids,
        selected.unit_offsets, selected.unit_positions, p,
        window_length=selected.window_length, pool_n=selected.pool_n, metadata=meta)
    result.native_scores = np.asarray(scores, dtype=np.float64)
    return result
