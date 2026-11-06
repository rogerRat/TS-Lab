import numpy as np
import pytest

from rage.core import TimeSeriesFragment
from rage.crb import CRBConfig, CRBEngine
from rage.experiments.anchor_budget_simulate import ArrayGradientComputer


def test_array_replay_preserves_exact_anchor_rows_and_crb():
    f = TimeSeriesFragment(np.arange(600, dtype=np.float32), fragment_id=3)
    starts = f.valid_starts(512, 64)
    values = np.random.default_rng(12).normal(size=(len(starts), 8)).astype(np.float32)
    replay = ArrayGradientComputer(starts, values, f.fragment_id)
    np.testing.assert_array_equal(replay(f, starts[[4, 1, 4]]), values[[4, 1, 4]])
    config = CRBConfig(cover_density=.16, error_density=.08, split_density=.04)
    a = CRBEngine(config, replay).process_fragment(f)
    b = CRBEngine(config, lambda fragment, positions: values[positions-1]).process_fragment(f)
    assert a.anchors.keys() == b.anchors.keys()
    assert a.computed_gradients == b.computed_gradients
    for x,y in zip(a.blocks,b.blocks):
        np.testing.assert_array_equal(x.start_positions,y.start_positions)
        np.testing.assert_array_equal(x.mean_gradient,y.mean_gradient)
    with pytest.raises(KeyError):
        replay(f, [len(starts)+1])
    with pytest.raises(KeyError):
        replay(TimeSeriesFragment(f.data, fragment_id=4), [1])
