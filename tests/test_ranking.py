import numpy as np
import pytest

from rags.experiments.ranking import (DEFAULT_TEMPERATURE, rank_mass,
    rank_resampled_windows, frozen_rank_protocol, validate_frozen_rank_protocol,
    apply_frozen_rank)


@pytest.mark.parametrize('size', [2, 100, 1082, 360774])
def test_top_half_eighty_percent_independent_of_object_count(size):
    rank, final, audit = rank_mass(np.arange(size), np.ones(size))
    assert rank[size // 2:].sum() == pytest.approx(.8, abs=2e-14)
    assert rank.sum() == pytest.approx(1.)
    np.testing.assert_allclose(rank, final, atol=1e-15)
    assert audit['additional_bp_allowed'] is False


def test_rank_stage_ignores_lengths_and_monotone_score_transforms():
    scores = np.array([-2., -.4, .1, .1, 1.])
    counts = np.array([2, 70, 3, 4, 50])
    rank, final, _ = rank_mass(scores, counts)
    equal_rank, _, _ = rank_mass(scores, np.ones(5))
    np.testing.assert_allclose(rank, equal_rank, atol=1e-15)
    for transformed in [scores / 492, scores * 100 + 20, np.exp(scores)]:
        other_rank, other_final, _ = rank_mass(transformed, counts)
        np.testing.assert_allclose(other_rank, rank, atol=1e-15)
        np.testing.assert_allclose(other_final, final, atol=1e-15)


def test_flat_scores_give_volume_mass_and_equal_per_window_exposure():
    rank, final, audit = rank_mass([0., 0., 0., 0.], [1, 2, 3, 4])
    np.testing.assert_allclose(rank, [.25] * 4)
    np.testing.assert_allclose(final, [.1, .2, .3, .4])
    assert audit['rank_top_half_mass_fractional_boundary'] == pytest.approx(.5)
    assert audit['half_boundary_inside_tie_group']


def test_ties_share_rank_mass_and_preserve_per_window_preference():
    rank, final, _ = rank_mass([3., 2., 2., 1.], [1, 2, 8, 3])
    assert rank[1] == rank[2]
    assert final[2] / final[1] == pytest.approx(4.)
    assert final[1] / 2 == pytest.approx(final[2] / 8)
    order = [2, 0, 3, 1]
    permuted_rank, permuted_final, _ = rank_mass(
        np.array([3., 2., 2., 1.])[order], np.array([1, 2, 8, 3])[order])
    np.testing.assert_allclose(permuted_rank, rank[order], atol=1e-15)
    np.testing.assert_allclose(permuted_final, final[order], atol=1e-15)


def test_volume_product_requires_normalization():
    rank, final, _ = rank_mass([1., 0.], [1, 1])
    assert (rank * np.array([.5, .5])).sum() == pytest.approx(.5)
    assert final.sum() == pytest.approx(1.)


def test_lower_temperature_increases_top_half_mass():
    rank, _, _ = rank_mass(np.arange(100), np.ones(100), temperature=.05)
    assert rank[50:].sum() == pytest.approx(1 / (1 + np.exp(-10)))
    assert DEFAULT_TEMPERATURE == pytest.approx(.36067376022224085)


def test_singleton():
    rank, final, _ = rank_mass([2.], [8])
    np.testing.assert_array_equal(rank, [1.])
    np.testing.assert_array_equal(final, [1.])


def test_trainable_view_roundtrip_keeps_parent_and_forbids_missing_scores():
    from rags.experiments.resampling import ResampledWindows
    parent = ResampledWindows(np.arange(8, dtype=np.float32),
        np.array([[0, 0], [0, 1], [0, 2], [0, 3]]), np.arange(4),
        [0, 1, 4], np.arange(4), [.7, .3], window_length=2, pool_n=4,
        metadata=dict(competition='global', source_weighting='equal_trajectory',
            temperature=.05, less_probability_revision='old', native_scores=[1., 1.]))
    deployed = rank_resampled_windows(parent)
    np.testing.assert_allclose(parent.probabilities, [.7, .3])
    np.testing.assert_allclose(deployed.probabilities, [.25, .75])
    np.testing.assert_allclose(deployed.window_probabilities(), [.25] * 4)
    assert deployed.metadata['less_probability_revision'] != 'old'
    roundtrip = ResampledWindows.from_payload(deployed.payload())
    np.testing.assert_allclose(roundtrip.probabilities, deployed.probabilities)
    with pytest.raises(ValueError, match='already rank'):
        rank_resampled_windows(deployed)
    parent.metadata.pop('native_scores')
    with pytest.raises(ValueError, match='additional BP is forbidden'):
        rank_resampled_windows(parent)


@pytest.mark.parametrize('scores,counts,temperature', [
    ([], [], .1), ([np.nan], [1], .1), ([1], [0], .1), ([1], [1.5], .1),
    ([1], [np.inf], .1), ([1], [1, 2], .1), ([1], [1], 0), ([1], [1], np.inf),
])
def test_invalid_inputs(scores, counts, temperature):
    with pytest.raises(ValueError):
        rank_mass(scores, counts, temperature=temperature)


def test_frozen_config_rejects_silent_temperature_and_prior_changes():
    import json
    from pathlib import Path
    protocol = json.loads(Path('configs/rags/rank_protocol.json').read_text())
    validate_frozen_rank_protocol(protocol)
    for key, value in [('temperature', .05), ('source_length_enters_ranking', True),
                       ('additional_bp_allowed', True), ('tie_policy', 'break_by_id')]:
        changed = dict(protocol, **{key: value})
        with pytest.raises(ValueError, match=key):
            validate_frozen_rank_protocol(changed)


def test_formal_view_persists_frozen_identity_across_serialization():
    from rags.experiments.resampling import ResampledWindows
    parent = ResampledWindows(np.arange(8, dtype=np.float32),
        np.array([[0, 0], [0, 1], [0, 2], [0, 3]]), np.arange(4),
        [0, 1, 4], np.arange(4), [.7, .3], window_length=2, pool_n=4,
        metadata=dict(competition='global', native_scores=[1., 1.]))
    deployed = apply_frozen_rank(parent, frozen_rank_protocol())
    restored = ResampledWindows.from_payload(deployed.payload())
    validate_frozen_rank_protocol(restored.metadata['frozen_rank_protocol'])
    np.testing.assert_allclose(restored.probabilities, [.25, .75])
