"""Approved rightmost tie priority and integer probe / geometric representative."""
from dataclasses import replace
import numpy as np
import pytest
from rags.core import TimeSeriesFragment
from rags.crb import CRBConfig, CRBEngine, _Interval


def run(config, gradients):
    observed=[]
    fragment=TimeSeriesFragment(np.arange(len(gradients)+2, dtype=np.float32), fragment_id=0)
    def field(_fragment, starts):
        ids=np.asarray(starts)-1
        observed.extend(ids.tolist())
        return gradients[ids]
    return CRBEngine(config, field).process_fragment(fragment), observed


@pytest.mark.parametrize("stage", ["cover", "error", "split"])
def test_rightmost_exact_ties_do_not_construct_rng(stage, monkeypatch):
    config=CRBConfig(context_length=2, prediction_length=1,
        cover_density=(4 if stage == "cover" else 3)/17,
        error_density=(1 if stage == "error" else 0)/17,
        split_density=(1 if stage == "split" else 0)/17,
        split_policy="closest_zero_v5", priority_tie_policy="right")
    gradients=(np.arange(17, dtype=np.float32)[:,None]/16)**2
    if stage == "split":
        gradients=np.full((17,1),4.,dtype=np.float32)
        gradients[[0,8,16],0]=[2.,-1.,2.]
    def forbidden(*args, **kwargs):
        raise AssertionError("Rightmost scheduling must not construct RNG")
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    for seed in (0, 1, 123):
        result, observed=run(replace(config, placement_seed=seed), gradients)
        assert observed == [0,16,8,12]
        assert result.epsilon_audit["tie_rng_enabled"] is False
        assert result.epsilon_audit["tie_seed_components"] is None
        assert result.epsilon_audit["tie_rng_streams"] == {}


def test_partial_coverage_ties_keep_rightmost_subset():
    config=CRBConfig(context_length=2,prediction_length=1,cover_density=7/33,
        error_density=0,split_density=0,split_policy="closest_zero_v5",
        priority_tie_policy="right")
    _, observed=run(config,np.zeros((33,1),dtype=np.float32))
    assert observed == [0,32,16,8,24,20,28]


def test_integer_floor_probe_is_not_geometric_representative():
    anchors={0:np.array([0.,0.]),3:np.array([3.,6.])}
    interval=_Interval(0,3)
    assert interval.midpoint == 1
    assert np.array_equal(CRBEngine._interval_mean(interval,anchors),[1.5,3.])
    # Error probes compare against the interpolation at the actual floor point.
    anchors[1]=np.array([1.,2.])
    assert CRBEngine._observed_curvature(0,1,3,anchors) == 0
    assert not _Interval(0,1).splittable
