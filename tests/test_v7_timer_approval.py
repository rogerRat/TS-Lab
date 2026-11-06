"""User-authorized Timer donor exception remains deliberately narrow."""
from copy import deepcopy
import numpy as np
import pytest
from rage.experiments.v7_cache import V7Cache, approved_timer_model_pair, TIMER_REUSE_APPROVAL
from test_v7 import Probe, request


def models():
    old = dict(model_id='thuml/timer-base-84m', student='timerxl-84m',
        revision='70077a71acce1b4c00d98332fcaabc694255d8e5',
        ckpt_revision='70077a71acce1b4c00d98332fcaabc694255d8e5',
        loaded_parameter_dtypes=['torch.bfloat16'],
        precision_policy='bf16-model-gradient-adamw-v2',
        config=dict(torch_dtype='bfloat16', transformers_version='4.40.1', hidden_size=1024))
    new = deepcopy(old)
    new['config'] = dict(dtype='bfloat16', transformers_version='4.57.1', hidden_size=1024)
    return old, new


def test_exact_pair_only_and_no_mutation():
    old, new = models()
    saved = deepcopy((old, new))
    assert approved_timer_model_pair(old, new)
    assert (old, new) == saved
    assert not approved_timer_model_pair(new, old)
    assert not approved_timer_model_pair(None, new)


@pytest.mark.parametrize('key,value', [
    ('revision', 'other'), ('ckpt_revision', 'other'), ('model_id', 'another-model'),
    ('student', 'chronos'), ('precision_policy', 'fp32'),
    ('loaded_parameter_dtypes', ['torch.float32']), ('extra', True)])
def test_other_model_changes_rejected(key, value):
    old, new = models()
    new[key] = value
    assert not approved_timer_model_pair(old, new)


@pytest.mark.parametrize('key,value', [
    ('dtype', 'float32'), ('torch_dtype', 'bfloat16'),
    ('transformers_version', '4.58.0'), ('hidden_size', 2048), ('extra', True)])
def test_other_config_changes_rejected(key, value):
    old, new = models()
    new['config'][key] = value
    assert not approved_timer_model_pair(old, new)


def test_approved_import_no_bp_preserves_existing_and_audits(tmp_path, monkeypatch):
    monkeypatch.delenv('RAGE_GRADIENT_CACHE_ROOT', raising=False)
    old_model, new_model = models()
    probe = Probe()
    common = dict(context_length=4, prediction_length=2)
    old = V7Cache(tmp_path/'old', probe, model_identity=old_model,
                  data_fields={'pool': 'old'}, **common)
    expected = old.get(request(3, 1))
    old.freeze()
    marker = old.path/'cache_manifest.json'
    original = marker.read_bytes()
    before = probe.count
    new = V7Cache(tmp_path/'new', probe, model_identity=new_model,
        data_fields={'pool': 'new'}, donors=[(old.path, {3: 100}, [])], **common)
    np.testing.assert_array_equal(new.get(request(100, 1)), expected)
    assert new.imported_windows == 1 and probe.count == before
    audit = new.reuse_audit[0]
    assert audit['compatibility_approval'] == TIMER_REUSE_APPROVAL
    assert audit['original_differing_fields'] == ['model']
    assert audit['matched'] and not audit['numerical_equivalence_claimed']
    new.freeze()
    again = V7Cache(tmp_path/'new', probe, model_identity=new_model,
        data_fields={'pool': 'new'}, donors=[(old.path, {3: 100}, [])], **common)
    np.testing.assert_array_equal(again.get(request(100, 1)), expected)
    assert again.imported_windows == 0 and probe.count == before
    again.close()
    assert marker.read_bytes() == original
    bad = V7Cache(tmp_path/'bad', probe, model_identity=new_model,
        data_fields={'pool': 'new'}, context_length=5, prediction_length=2,
        donors=[(old.path, {3: 100}, [])])
    assert not bad.donor_rows and not bad.reuse_audit[0]['matched']
    assert 'compatibility_approval' not in bad.reuse_audit[0]
    bad.close()
