import json

import numpy as np
import pytest

from baselines.ltsv.runtime import accounting, comparison_seconds, load_restore_timing
from scripts.exp.report_selection_runtime import report


def test_adjust_only_restore_component_and_remove_setup():
    calibration = dict(cpu_seconds=.05, gpu_seconds=.002)
    adjustment = accounting(calibration, 100, .1)
    assert adjustment['add_back_seconds'] == pytest.approx(4.7)
    assert comparison_seconds(100, adjustment) == pytest.approx(104.7)
    assert comparison_seconds(100, None) == 100
    assert accounting(None, 100, 0) is None


def test_calibration_requires_matching_explicit_model_fields(tmp_path):
    assert load_restore_timing(tmp_path, {}) is None
    payload = dict(schema='ltsv-restore-timing-v1', model={'student': 'one'},
                   cpu_seconds=.05, gpu_seconds=.002)
    (tmp_path / 'ltsv_restore_timing.json').write_text(json.dumps(payload))
    assert load_restore_timing(tmp_path, payload['model']) == payload
    with pytest.raises(ValueError, match='fields differ'):
        load_restore_timing(tmp_path, {'student': 'two'})


def test_report_uses_unoptimized_estimate_not_restore_multiplier(tmp_path):
    (tmp_path / 'protocol.json').write_text(json.dumps(dict(
        students=['model'], datasets=['task'], arms=['ltsv', 'random'])))
    folder = tmp_path / 'students/model/selections'
    folder.mkdir(parents=True)
    adjustment = accounting(dict(cpu_seconds=.05, gpu_seconds=.002), 100, .1)
    np.savez(folder / 'task__ltsv.npz', cost=json.dumps(dict(wall_seconds=100,
        notes={'runtime_adjustment': adjustment, 'scores_cached': False})))
    data = report(tmp_path)
    row = data['rows'][0]
    assert row['reported_cold_selection_seconds'] == pytest.approx(104.7)
    assert row['actual_execution_seconds'] == 100
    assert row['estimated_native_to_actual_ratio'] == pytest.approx(1.047)
    assert data['complete'] is False and len(data['missing']) == 1


def test_warm_score_read_is_not_reported_as_cold_method_time(tmp_path):
    (tmp_path / 'protocol.json').write_text(json.dumps(dict(
        students=['model'], datasets=['task'], arms=['ltsv'])))
    folder = tmp_path / 'students/model/selections'
    folder.mkdir(parents=True)
    np.savez(folder / 'task__ltsv.npz', cost=json.dumps(dict(wall_seconds=.1,
        notes={'scores_cached': True, 'runtime_adjustment': None})))
    row = report(tmp_path)['rows'][0]
    assert row['reported_cold_selection_seconds'] is None
    assert row['cache_state'] == 'warm'
