import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from rags.experiments.v7_extension import make_config, validate_extension, publish_source_cache, CANDIDATES
from rags.experiments.v7_cache import V7Cache


def test_extension_changes_only_approved_fields():
    parent = json.loads(Path('configs/rags/main_v7_curve4.json').read_text())
    new = make_config(parent,'/parent','/new')
    validate_extension(new)
    assert new['seeds'] == [0,1]
    assert len(new['datasets']) == 1
    assert new['arms'] == parent['arms']
    assert new['settings'] == parent['settings']
    broken = deepcopy(new)
    broken['settings']['chronos-base']['epochs'] = 4
    with pytest.raises(ValueError):
        validate_extension(broken)
    for task in CANDIDATES:
        validate_extension(make_config(parent,'/parent','/new',task))


def test_initialize_original_only_reuses_evaluation_no_training(tmp_path,monkeypatch):
    import torch
    import rags.experiments.v7_train as train
    parent=json.loads(Path('configs/rags/main_v7_curve4.json').read_text())
    config=make_config(parent,'/parent',str(tmp_path))
    task=config['datasets'][0]; student='chronos-base'
    directory=tmp_path/'inputs'/student;directory.mkdir(parents=True)
    np.savez(directory/f'{task}__exam.npz',windows=np.zeros((1,576),dtype=np.float32),entities=['a'],scales=[1.])
    (directory/f'{task}__manifest.json').write_text('{}')
    calls=[]
    monkeypatch.setattr(train,'evaluate',lambda *a,**k: calls.append(1) or {'mase':1.})
    def forbidden(*a,**k):raise AssertionError('Initialization trained a model')
    monkeypatch.setattr(train,'run_arm_training',forbidden)
    native=SimpleNamespace(model=torch.nn.Linear(2,2))
    for seed in (0,1):
        train.train_task(config,tmp_path,student,seed,task,native,
            SimpleNamespace(context_length=512),{}, {'model_id':'toy'},initialize_only=True)
        files=list((tmp_path/f'seeds/seed{seed}'/student/'cells').glob('*.json'))
        assert [p.name for p in files]==[f'{task}__orig.json']
    assert calls==[1]


def test_source_missing_fails_before_any_bp():
    cache = V7Cache.__new__(V7Cache)
    cache.manifest = {'data':{'extension_source_reuse':True}}
    cache.rows = {}
    with pytest.raises(RuntimeError,match='no BP allowed'):
        cache.prepare([SimpleNamespace(dataset='source',ids=np.array([[3,1]]))])


def test_source_links_preserve_inodes_exclude_reference_and_freeze_fields(tmp_path):
    old = tmp_path/'old'; old.mkdir()
    data = dict(feasible_pool={'rows':[9]},actual_training_pools={'0':{'fragment_ids':[0]}},
                reference_identities={'old':{}})
    fields = dict(data=data,model={'revision':'pinned'},format='test')
    (old/'cache_manifest.json').write_text(json.dumps(fields))
    for name,ds,ids in [('windows-000000.npz','source',[[0,1],[0,4]]),
                        ('windows-000001.npz','reference:old',[[0,1]])]:
        np.savez(old/name,dataset=ds,window_ids=ids,gradients_bf16=np.zeros((len(ids),3),dtype=np.uint16))
    np.savez(old/'window_index.npz',datasets=['source','reference:old'],dataset_ids=[0,0,1],
        window_ids=[[0,1],[0,4],[0,1]],shards=['windows-000000.npz','windows-000001.npz'],
        shard_ids=[0,0,1],shard_rows=[0,1,0])
    crb = old/'source-crb/fields-0'; crb.mkdir(parents=True)
    (crb/'cache_manifest.json').write_text(json.dumps({'gradient':fields,'crb':{'cover':.005}}))
    np.savez(crb/'trajectory-0.npz',example=[1])
    shared = deepcopy(data); shared['reference_identities']={'new':{}}
    shared['extension_source_reuse']={'version':'test'}
    root = tmp_path/'new'
    receipt = publish_source_cache(old,shared,root)
    target = root/'old'
    assert receipt['source_rows'] == 2 and receipt['source_payload_new_bytes'] == 0
    assert (old/'windows-000000.npz').samefile(target/'windows-000000.npz')
    assert not (target/'windows-000001.npz').exists()
    assert (crb/'trajectory-0.npz').samefile(target/'source-crb/fields-0/trajectory-0.npz')
    assert json.loads((old/'cache_manifest.json').read_text()) == fields
    assert publish_source_cache(old,shared,root) == receipt
    changed = deepcopy(shared); changed['feasible_pool']['rows']=[8]
    with pytest.raises(ValueError,match='physical identities'):
        publish_source_cache(old,changed,tmp_path/'bad')


def test_completed_task_reentry_skips_inputs_assets_and_selection(tmp_path,monkeypatch):
    import sys
    import scripts.run_v7_extension_queue as queue
    observed=[]
    monkeypatch.setattr(queue,'CANDIDATES',[CANDIDATES[0]])
    monkeypatch.setattr(queue,'read',lambda path:{})
    monkeypatch.setattr(queue,'make_config',lambda base,parent,run,task:{'run_root':str(run),'students':['chronos-base']})
    monkeypatch.setattr(queue,'freeze',lambda *args:None)
    monkeypatch.setattr(queue,'complete',lambda *args:True)
    monkeypatch.setattr(queue,'finalize_task',lambda *args:observed.append('finalized'))
    monkeypatch.setattr(queue,'prepare',lambda *args:pytest.fail('Repeated completed inputs'))
    monkeypatch.setattr(queue,'run_task',lambda *args,**kwargs:pytest.fail('Repeated completed work'))
    monkeypatch.setattr(sys,'argv',['queue','--parent',str(tmp_path)])
    queue.main()
    assert observed==['finalized']


def test_select_admission_checks_only_its_own_model_assets(tmp_path,monkeypatch):
    from rags.experiments import v7_run
    calls=[]
    def inspect(config,run,phase,*,student=None):
        calls.append((phase,student))
        raise RuntimeError('admission_observed')
    monkeypatch.setattr(v7_run,'gate',inspect)
    with pytest.raises(RuntimeError,match='admission_observed'):
        v7_run.model_phase({},tmp_path,'timesfm-2.5','select')
    assert calls==[('assets','timesfm-2.5')]
