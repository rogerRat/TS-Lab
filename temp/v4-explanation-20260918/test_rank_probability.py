import numpy as np
import pytest

from rage.experiments.rank_probability import rank_probability, rank_resampled_windows


def test_monotone_score_changes_leave_rank_mass_unchanged():
    scores = np.array([-3., -.2, .5, .5, 2.])
    counts = [2, 7, 1, 5, 4]
    a, _ = rank_probability(scores, counts)
    for transformed in (scores / 492, scores * 100 + 20, np.exp(scores)):
        b, _ = rank_probability(transformed, counts)
        np.testing.assert_allclose(a, b, atol=1e-15)


def test_flat_scores_respect_data_volume_not_equal_trajectory():
    p, audit = rank_probability([0., 0., 0.], [2, 3, 5])
    np.testing.assert_allclose(p, [.2, .3, .5])
    assert audit['relative_ess'] == pytest.approx(1)


def test_splitting_same_score_unit_preserves_total_mass_and_other_units():
    before, _ = rank_probability([-.3, .7, 1.], [4, 6, 10])
    after, _ = rank_probability([-.3, .7, .7, 1.], [4, 2, 4, 10])
    np.testing.assert_allclose(before, [after[0], after[1:3].sum(), after[3]], atol=1e-15)
    assert after[1] / after[2] == pytest.approx(.5)


def test_unit_control_has_same_sorted_distribution_for_distinct_scores():
    a, _ = rank_probability([-9., .01, .02, .03])
    b, _ = rank_probability([100., -100., -300., 40.])
    np.testing.assert_allclose(np.sort(a), np.sort(b), atol=1e-15)


def test_order_is_preserved_for_per_window_density_and_permutations():
    scores = np.array([4., -1., 2., 2.])
    counts = np.array([1, 50, 3, 7])
    p, _ = rank_probability(scores, counts)
    assert p[0] / counts[0] > p[2] / counts[2] > p[1] / counts[1]
    assert p[2] / counts[2] == pytest.approx(p[3] / counts[3])
    order = [3, 1, 0, 2]
    permuted, _ = rank_probability(scores[order], counts[order])
    np.testing.assert_allclose(permuted, p[order], atol=1e-15)


@pytest.mark.parametrize('scores,counts,temperature', [
    ([], None, .05), ([np.nan], None, .05), ([[1]], None, .05),
    ([1], [0], .05), ([1], [np.inf], .05), ([1], [1, 2], .05),
    ([1], None, 0), ([1], None, np.inf),
])
def test_invalid_inputs_fail(scores, counts, temperature):
    with pytest.raises(ValueError):
        rank_probability(scores, counts, temperature=temperature)


def test_trainable_view_preserves_windows_parent_mass_and_serializes_protocol():
    from rage.experiments.resampling import ResampledWindows
    parent = ResampledWindows(np.arange(8, dtype=np.float32),
        np.array([[0, 0], [0, 1], [0, 2], [0, 3]]), np.arange(4),
        [0, 1, 4], np.arange(4), [.7, .3], window_length=2, pool_n=4,
        metadata=dict(competition='global', source_weighting='equal_trajectory',
            temperature=.05, less_probability_revision='old', native_scores=[1., 1.]))
    deployed = rank_resampled_windows(parent)
    np.testing.assert_allclose(parent.probabilities, [.7, .3])
    np.testing.assert_allclose(deployed.probabilities, [.25, .75])
    np.testing.assert_allclose(deployed.window_probabilities(), np.full(4, .25))
    np.testing.assert_array_equal(parent.unit_positions, deployed.unit_positions)
    assert deployed.metadata['less_probability_revision'] != 'old'
    roundtrip = ResampledWindows.from_payload(deployed.payload())
    np.testing.assert_allclose(roundtrip.probabilities, deployed.probabilities)
    assert roundtrip.metadata['rank_deployment']['source_prior'] == 'legal_window_counts'
    with pytest.raises(ValueError, match='already rank'):
        rank_resampled_windows(deployed)
