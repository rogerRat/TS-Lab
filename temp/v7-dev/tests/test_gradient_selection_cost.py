from dataclasses import replace
import json
import numpy as np
import pytest
from rage.experiments.gradient_cost import gradient_cost_record
from rage.experiments.selection_metrics import SelectionMetrics
from test_gradient_assets import Probe, cache_at
from assets.gradient import WindowRequest
from test_selection_metrics import output


def test_required_estimate_not_shared_union_charge_and_no_overlap():
    kwargs = dict(required_source=80, required_reference=20,
        benchmark=dict(mean_seconds_per_window=.25), preparation_new_windows=240,
        preparation_seconds=65., preparation_gradient_seconds=60.,
        method_preparation_nongrad_seconds=3., selection_seconds=2.)
    record = gradient_cost_record(**kwargs)
    assert record['estimated_cold_selection_seconds'] == 30.  # 100*.25 + 3 + 2
    assert record['preparation']['new_windows'] == 240
    assert record['preparation']['nongrad_seconds'] == 5.
    warm = gradient_cost_record(**dict(kwargs, preparation_new_windows=0,
        preparation_seconds=3., preparation_gradient_seconds=0.))
    assert warm['estimated_cold_selection_seconds'] == 30.
    assert warm['preparation']['new_windows'] == 0
    missing = gradient_cost_record(**dict(kwargs, benchmark=None))
    assert missing['estimated_cold_selection_seconds'] is None
    hit = gradient_cost_record(**kwargs, selection_cached=True)
    assert hit['estimated_cold_selection_seconds'] is None
    with pytest.raises(ValueError, match='enclosing'):
        gradient_cost_record(**dict(kwargs, preparation_gradient_seconds=66.))


def test_observed_batches_cache_hits_restart_and_execution_isolation(tmp_path):
    probe = Probe()
    windows = np.arange(18, dtype=np.float32).reshape(3, 6)
    req = WindowRequest('source', np.column_stack((np.ones(3), np.arange(1, 4))),
                        lambda rows: windows[rows])
    cache = cache_at(tmp_path, probe)
    assert cache.gradient_benchmark()['mean_seconds_per_window'] is None
    cache.prepare([req])
    first = cache.gradient_benchmark()
    assert first['gradient_windows'] == 3
    assert first['gradient_seconds'] == cache.computed_gradient_seconds > 0
    cache.prepare([req])
    assert cache.gradient_benchmark() == first
    cache.freeze()
    resumed = cache_at(tmp_path, probe)
    assert resumed.gradient_benchmark() == first
    resumed.prepare([req])
    assert resumed.computed_windows == 0
    assert resumed.computed_gradient_seconds == 0
    resumed.freeze()
    probe.execution = replace(probe.execution, gradient_batch=2)
    other = cache_at(tmp_path, probe)
    assert other.path == resumed.path  # gradient values still reusable
    assert other.gradient_benchmark()['mean_seconds_per_window'] is None
    other.prepare([req])
    assert probe.count == 3
    other.close()
    assert len(json.loads((other.path / 'gradient_timing.json').read_text())['records']) == 2


def test_metric_cost_record_and_cached_original_preserved(tmp_path):
    meter = SelectionMetrics(tmp_path, device='cpu', protocol={})
    kwargs = dict(required_source=4, required_reference=2,
        benchmark=dict(mean_seconds_per_window=.5), preparation_new_windows=2,
        preparation_seconds=2., preparation_gradient_seconds=1.,
        method_preparation_nongrad_seconds=1.)
    result = meter.measure('task', 'rage', output, gradient_cost=kwargs)
    original = result.cost.notes['gradient_cost'].copy()
    assert original['estimated_cold_selection_seconds'] == 4 + original['measured_selection_body_seconds']
    meter.measure('task', 'rage', lambda: result, selection_cached=True, gradient_cost=kwargs)
    assert result.cost.notes['gradient_cost'] == original
    rows = [json.loads(r) for r in meter.path.read_text().splitlines()]
    assert rows[-1]['gradient_cost']['estimated_cold_selection_seconds'] is None
