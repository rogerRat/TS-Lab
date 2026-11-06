import numpy as np
import pytest
from rage.core import TimeSeriesFragment
from rage.experiments.arms import ArmOutput, SelectionCost, WindowIndex
from rage.experiments.resampling import ResampledWindows
from rage.experiments import v7_io


def selection(probabilities=(0., .4, .6, 0.)):
    pool = [TimeSeriesFragment(np.arange(7, dtype=np.float32), fragment_id=100+i,
                              entity_id=str(i)) for i in range(4)]
    index = WindowIndex(pool, context_length=4, prediction_length=2)
    manifest = dict(total_equivalents=4, fragment_ids=[100, 101, 102, 103])
    units = [np.arange(*index.bounds(f.fragment_id)) for f in pool]
    selected = ResampledWindows.from_index(index, units, probabilities,
        pool_manifest=manifest, metadata={'probability_mapping': 'native_OT_mass'})
    return index, manifest, selected


@pytest.mark.parametrize('arm', ['tsds', 'lp-mirror'])
def test_native_zeros_roundtrip_and_sampling(tmp_path, arm):
    index, manifest, selected = selection()
    before = selected.probabilities.copy()
    coverage = v7_io.validate_support(selected, arm)
    assert coverage['zero_mass_legal_windows'] == 4
    assert coverage['filtered_legal_windows'] == 0
    assert coverage['support_policy'] == 'native_zero_allowed_no_smoothing'
    path = tmp_path/'selection.npz'
    identity = {'seed': 0}
    v7_io.save(path, ArmOutput(arm, selected, [], SelectionCost(), {}), identity)
    restored = v7_io.load(path, index, manifest, identity).selected
    np.testing.assert_array_equal(restored.probabilities, before)
    np.testing.assert_array_equal(selected.probabilities, before)
    sampled = restored.sample_indices(np.random.default_rng(0), 10000)
    assert set(restored.window_ids[sampled, 0]) == {101, 102}
    assert restored.pool_n == 4


@pytest.mark.parametrize('arm', ['rand', 'less-main', 'rage-block-linearSum',
                               'grad-dot', 'grad-cos', 'grand-static'])
def test_other_arms_reject_zero_probability(tmp_path, arm):
    _, _, selected = selection()
    with pytest.raises(ValueError, match=arm+': selection requires positive support'):
        v7_io.save(tmp_path/'bad.npz', ArmOutput(arm, selected, [], SelectionCost(), {}), {})
    assert not (tmp_path/'bad.npz').exists()


@pytest.mark.parametrize('p', [[0., -.1, 1.1, 0.], [0., np.nan, 1., 0.],
                             [0., np.inf, 1., 0.], [0., .2, .2, 0.], [0., 0., 0., 0.]])
def test_invalid_mass_still_rejected(p):
    _, _, selected = selection()
    selected.probabilities = np.array(p)
    with pytest.raises(ValueError, match='invalid selection probability mass'):
        v7_io.validate_support(selected, 'tsds')


def test_missing_window_ownership_still_rejected():
    _, _, selected = selection()
    selected.unit_offsets[-1] -= 1
    with pytest.raises(ValueError, match='whole actual pool'):
        v7_io.validate_support(selected, 'tsds')


def test_native_tsds_kernel_zeros_are_accepted():
    from baselines.tsds.algorithms import tsds_probability_assignment
    p = tsds_probability_assignment(np.array([[1., 2.]]), np.array([[1, 2]]),
        np.ones((1, 2)), candidate_count=4, alpha=.1, c_param=5.)
    assert p[0] == p[3] == 0
    _, _, selected = selection(p)
    v7_io.validate_support(selected, 'tsds')
    np.testing.assert_array_equal(selected.probabilities, p)
