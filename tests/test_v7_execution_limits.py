import json
import sys

import pytest

from rags.experiments import v7_run


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_training_seed_gate_before_freeze(tmp_path, monkeypatch, seed):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'run_root': str(tmp_path), 'datasets': []}))
    (tmp_path / 'execution_limits.json').write_text(
        json.dumps({'allowed_training_seeds': [0, 1]}))
    sentinel = tmp_path / 'existing_result.json'
    sentinel.write_text('preserve')
    monkeypatch.setattr(v7_run, 'validate', lambda config: None)
    monkeypatch.setattr(v7_run, 'asset_protocol', lambda config: config)

    def reached_freeze(*args):
        raise RuntimeError('allowed seed reached freeze')

    monkeypatch.setattr(v7_run, 'freeze', reached_freeze)
    monkeypatch.setattr(sys, 'argv', [
        'run_v7', '--config', str(config), '--phase', 'train', '--seed', str(seed)])
    if seed == 2:
        with pytest.raises(SystemExit) as error:
            v7_run.main()
        assert error.value.code == 2
    else:
        with pytest.raises(RuntimeError, match='allowed seed reached freeze'):
            v7_run.main()
    assert sentinel.read_text() == 'preserve'
