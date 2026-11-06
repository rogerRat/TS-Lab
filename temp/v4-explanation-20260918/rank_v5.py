"""V5: object-only ranks, exponential interval mass, then source-volume prior.

Array-only probability construction. No model, gradient accessor or BP path.
"""
from __future__ import annotations

import numpy as np

REVISION = "object-rank-exponential-interval-v5"
DEFAULT_TEMPERATURE = 1. / (2. * np.log(4.))


def frozen_rank_protocol_v5():
    """Approved formal defaults; protocol changes require a new identity."""
    return dict(protocol_id="v5-rank-probability-frozen-20260918",
        status="approved_frozen", rank_probability_revision=REVISION,
        ranking_measure="equal_object_count", score_information="order_and_exact_ties_only",
        curve="F(x)=(1-exp(-x/T))/(1-exp(-1/T)); best_rank_at_x=0",
        rank_allocation="interval_integral", temperature=float(DEFAULT_TEMPERATURE),
        temperature_expression="1/(2*ln(4))", target_top_object_fraction=.5,
        target_rank_mass_before_source_prior=.8,
        tie_policy="whole_interval_then_equal_per_object",
        boundary_tie_policy="preserve_ties_and_report_deviation",
        reference_weights_stage="score_aggregation",
        source_prior="legal_window_counts_after_rank_mapping",
        source_length_enters_ranking=False, source_length_applied_once=True,
        normalization_after_source_prior="required", post_rank_softmax=False,
        gating=False, within_unit_sampling="uniform_owned_legal_start",
        additional_bp_allowed=False, missing_prepared_assets="error_without_bp")


def validate_frozen_rank_protocol_v5(protocol):
    expected = frozen_rank_protocol_v5()
    if protocol != expected:
        keys = sorted(set(protocol) | set(expected)) if isinstance(protocol, dict) else []
        differences = [key for key in keys if protocol.get(key) != expected.get(key)]
        raise ValueError("Frozen V5 rank protocol mismatch: " + ', '.join(differences))


def rank_mass_v5(scores, legal_counts, *, temperature=DEFAULT_TEMPERATURE):
    """Return intermediate rank mass, final source mass, and diagnostics.

    Higher scores rank first. Original gaps are discarded; exact ties share
    their entire rank interval equally. Counts never affect the ranks.
    Default curve assigns 80% to the top half before the volume multiplier,
    provided that the half boundary does not bisect a tie group.
    """
    scores = np.asarray(scores, dtype=np.float64)
    counts = np.asarray(legal_counts, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Scores must be a nonempty finite vector")
    if (counts.shape != scores.shape or not np.isfinite(counts).all()
            or np.any(counts <= 0) or np.any(counts != np.rint(counts))):
        raise ValueError("Legal counts must be matching positive finite integers")
    if not np.isfinite(temperature) or temperature <= 0 or not np.isfinite(1. / temperature):
        raise ValueError("Temperature must have a finite positive reciprocal")
    beta = 1. / temperature
    _, inverse, frequency = np.unique(scores, return_inverse=True, return_counts=True)
    # np.unique orders low scores first; allocate rank intervals high first.
    descending_frequency = frequency[::-1]
    upper = np.cumsum(descending_frequency, dtype=np.float64) / len(scores)
    lower = np.r_[0., upper[:-1]]
    width = descending_frequency / len(scores)
    # F(b)-F(a), computed in log space to retain narrow interval accuracy.
    log_group_mass = (-beta * lower + np.log(-np.expm1(-beta * width))
                      - np.log(-np.expm1(-beta)))
    log_per_object = (log_group_mass - np.log(descending_frequency))[::-1][inverse]
    rank_mass = np.exp(log_per_object - log_per_object.max())
    rank_mass /= rank_mass.sum()
    final_logits = log_per_object + np.log(counts)
    final_mass = np.exp(final_logits - final_logits.max())
    final_mass /= final_mass.sum()  # Required normalization after multiplying priors.
    q = counts / counts.max()
    q /= q.sum()
    if np.any(q == 0):
        raise ValueError("Source prior underflows")
    group_mass = np.exp(log_group_mass)
    # Expected half-object mass with fractional treatment of a boundary tie.
    fractions = np.clip((.5 - lower) / width, 0., 1.)
    actual_half = float(group_mass @ fractions)
    crosses_half = bool(np.any((lower < .5) & (upper > .5)))
    target_half = float(1. / (1. + np.exp(-.5 * beta)))
    audit = dict(rank_probability_revision=REVISION, temperature=float(temperature),
        inverse_temperature=float(beta), ranking_measure="equal_object_count",
        curve="F(x)=(1-exp(-x/T))/(1-exp(-1/T)); best_rank_at_x=0",
        rank_allocation="interval_integral", tie_policy="whole_interval_then_equal_per_object",
        source_prior="legal_window_counts_after_rank_mapping",
        source_length_enters_ranking=False, source_length_applied_once=True,
        normalization_after_source_prior="required", additional_bp_allowed=False,
        gating=False, units=len(scores), unique_scores=len(frequency),
        target_rank_top_half_mass=target_half,
        rank_top_half_mass_fractional_boundary=actual_half,
        half_boundary_inside_rank_group=crosses_half,
        half_boundary_inside_tie_group=crosses_half and bool(np.any(
            (lower < .5) & (upper > .5) & (descending_frequency > 1))),
        rank_ess=float(1. / np.sum(rank_mass ** 2)),
        window_relative_ess=float(1. / np.sum(final_mass * (final_mass / q))))
    return rank_mass, final_mass, audit


def rank_resampled_windows_v5(selected, *, temperature=DEFAULT_TEMPERATURE):
    """Construct a separate trainable selection from existing global scores.

    Never queries gradients. Missing scores raise an error, not a BP request.
    Caller must use a new run identity; frozen V4 does not call this function.
    """
    from .resampling import ResampledWindows
    if selected.metadata.get('competition', 'global') != 'global':
        raise ValueError("Local-mixture diagnostics are not final global scores")
    if selected.metadata.get('rank_deployment') or selected.metadata.get('rank_v5'):
        raise ValueError("Selection is already rank deployed")
    ref = selected.metadata.get('reference_audit') or {}
    scores = ref.get('source_scores_for_softmax', ref.get('source_scores'))
    if scores is None:
        scores = selected.metadata.get('native_scores')
    if scores is None:
        raise ValueError("Prepared final scores required; additional BP is forbidden")
    _, probabilities, audit = rank_mass_v5(scores, np.diff(selected.unit_offsets),
        temperature=temperature)
    metadata = dict(selected.metadata, rank_v5=audit, probability_mapping=REVISION,
        pre_rank_probability_mapping=selected.metadata.get('probability_mapping'),
        pre_rank_temperature=selected.metadata.get('temperature'),
        temperature=float(temperature), source_weighting='legal_window_counts')
    for key in ('less_probability_revision', 'rage_probability_revision'):
        if key in metadata:
            metadata[key] += ':' + REVISION
    result = ResampledWindows(selected.values, selected.window_locations, selected.window_ids,
        selected.unit_offsets, selected.unit_positions, probabilities,
        window_length=selected.window_length, pool_n=selected.pool_n, metadata=metadata)
    result.native_scores = np.asarray(scores, dtype=np.float64)
    return result


def apply_frozen_rank_v5(selected, protocol):
    """Formal opt-in entry: validate and persist the complete frozen identity."""
    validate_frozen_rank_protocol_v5(protocol)
    result = rank_resampled_windows_v5(selected, temperature=protocol['temperature'])
    result.metadata['frozen_v5_rank_protocol'] = frozen_rank_protocol_v5()
    return result
