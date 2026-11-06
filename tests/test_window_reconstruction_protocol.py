import json
from types import SimpleNamespace
import numpy as np
import pytest
from rage.window_reconstruction import interpolate_window_gradients
from rage.epsilon_protocol import load_p10_calibration


def test_linear_raw_vectors_not_block_mean_or_unit_vectors():
    result = SimpleNamespace(valid_starts=np.array([1,3,5,7]),
        anchors={0:np.array([2.,0.]),3:np.array([0.,8.])}, blocks=[])
    expected = np.array([[2,0],[4/3,8/3],[2/3,16/3],[0,8]])
    np.testing.assert_allclose(interpolate_window_gradients(result),expected)
    # Block rejection cannot remove a legal window's estimate; no tail extrapolation.
    with pytest.raises(ValueError):
        interpolate_window_gradients(result,[4])


def test_exact_interior_anchor_and_single_legal_window():
    result = SimpleNamespace(valid_starts=np.array([1,2,3]),
        anchors={0:np.array([2.,1.]),1:np.array([-3.,4.]),2:np.array([0.,1.])})
    np.testing.assert_array_equal(interpolate_window_gradients(result,[1]),[[-3,4]])
    result = SimpleNamespace(valid_starts=np.array([1]),anchors={0:np.array([3.,4.])})
    np.testing.assert_array_equal(interpolate_window_gradients(result),[[3,4]])


def test_p10_is_frozen_and_precision_matched(tmp_path):
    identity=dict(model_id='m',revision='r',projection_dimension=4,projection_seed=42,
                  precision_policy='bf16',ckpt='model')
    report=dict(model=identity.copy(),count=100,percentile=10,epsilon=.25,distribution={'10':.25})
    path=tmp_path/'epsilon.json'
    path.write_text(json.dumps(report))
    assert load_p10_calibration(path,identity)['epsilon']==.25
    for change in ({'percentile':20},{'epsilon':1e-8},{'model':dict(identity,precision_policy='fp32')}):
        path.write_text(json.dumps(dict(report,**change)))
        with pytest.raises(ValueError):
            load_p10_calibration(path,identity)
