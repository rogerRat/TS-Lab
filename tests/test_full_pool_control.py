import numpy as np
from rage.core import TimeSeriesFragment
from rage.experiments.arms import WindowIndex
from rage.experiments.native_selection import full_pool_control


def test_full_pool_retains_every_legal_start_without_crossing_trajectories():
    pool = [TimeSeriesFragment(np.arange(n,dtype=np.float32)+i*100, entity_id=str(i), fragment_id=i)
            for i,n in enumerate([10,12])]
    index = WindowIndex(pool, context_length=4,prediction_length=2,stride=1)
    result = full_pool_control(pool,index)
    assert len(result.selected)==12
    assert result.diagnostics['actual_pool_points']==22
    expected = np.vstack([np.lib.stride_tricks.sliding_window_view(f.data,6) for f in pool])
    np.testing.assert_array_equal(result.selected[:],expected)
    assert result.diagnostics['selection_budget_matched'] is False
