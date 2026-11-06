import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import run_v7_suite as suite
from rage.experiments.v7_run import gate, marker, cell_identity
from assets.gradient.cache import atomic_json


def test_model_gate_does_not_wait_for_other_models(tmp_path):
    config = dict(seeds=[0,1,2], datasets=['task'], arms=['orig'])
    model = suite.MODELS[0]
    (tmp_path/'inputs').mkdir()
    for seed in config['seeds']:
        manifest = {'seed': seed}
        atomic_json(tmp_path/'inputs'/f'seed{seed}_manifest.json', manifest)
        path = marker(tmp_path,model,seed,'task','select')
        path.parent.mkdir(parents=True)
        atomic_json(path, dict(
            identity=cell_identity(config,model,seed,'task',manifest), complete=True,
            selection_new_bp=0, selections=[{'arm':'orig'}]))
    gate(config,tmp_path,'select',student=model)
    with pytest.raises(RuntimeError, match='incomplete'):
        gate(config,tmp_path,'select')
    marker(tmp_path,model,2,'task','select').unlink()
    with pytest.raises(RuntimeError, match='incomplete'):
        gate(config,tmp_path,'select',student=model)


def test_curve_training_dispatch_without_isolated_validation(tmp_path, monkeypatch):
    config = json.loads(Path('configs/rage/main_v7_curve4.json').read_text())
    assert 'timer_validation_manifest' not in config
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    calls = []
    gates = []
    monkeypatch.setattr(sys, 'argv', ['suite', '--config', str(path), '--run', str(tmp_path/'run'),
                                     '--phase', 'train', '--parallel-models', '1'])
    monkeypatch.setattr(suite, 'gate', lambda c, r, phase, **kw: gates.append((phase,kw.get('student'))))
    monkeypatch.setattr(suite.subprocess, 'run', lambda command, **kw:
                        (calls.append(command) or SimpleNamespace(returncode=0)))
    assert suite.main() == 0
    assert gates == [('assets',None)]+[('select',s) for s in suite.MODELS]
    assert len(calls) == 9
    assert [c[c.index('--seed')+1] for c in calls] == ['0','1','2']*3
    assert all(c[c.index('--phase')+1] == 'train' for c in calls)


def test_training_dispatch_still_blocked_by_selection_barrier(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['suite', '--config', 'configs/rage/main_v7_curve4.json',
                                     '--run', str(tmp_path), '--phase', 'train'])
    def incomplete(*args, **kwargs):
        raise RuntimeError('selection incomplete')
    monkeypatch.setattr(suite, 'gate', incomplete)
    def forbidden(*args, **kwargs):
        raise AssertionError('must not launch training before selection gate')
    monkeypatch.setattr(suite.subprocess, 'run', forbidden)
    with pytest.raises(RuntimeError, match='selection incomplete'):
        suite.main()
