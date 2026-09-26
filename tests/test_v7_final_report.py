import pytest

from scripts.report_v7_final import rank_shape, reported_mase, SHAPE_ARMS


def test_endpoint_is_frozen_and_orig_is_untrained():
    row = dict(arm="rags-shape-expSum", test_curve=[
        dict(epoch=2.5, metrics=dict(mase=0.7)),
        dict(epoch=3.0, metrics=dict(mase=0.8)),
        dict(epoch=3.5, metrics=dict(mase=0.1)),
    ], mase=0.1)
    assert reported_mase("chronos-base", row) == 0.8
    with pytest.raises(ValueError, match="3.0-epoch"):
        reported_mase("chronos-base", dict(row, test_curve=[]))
    assert reported_mase("timerxl-84m", dict(row, summary_epoch=4.0)) == 0.1
    assert reported_mase("chronos-base", dict(arm="orig", trained=False, mase=1.2)) == 1.2


def test_shape_top_two_requires_complete_identical_support():
    paired = {arm: [("task", "model", seed, 1.0 - i)
                    for seed in range(2)] for i, arm in enumerate(SHAPE_ARMS)}
    assert rank_shape(paired, target=2) == SHAPE_ARMS[:2]
    paired[SHAPE_ARMS[0]].pop()
    with pytest.raises(ValueError, match="complete paired cells"):
        rank_shape(paired, target=2)
