from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from rage.experiments.v7_protocol import (make_curve_config, make_reduced_config,
    validate, exposure_audit, asset_protocol, EXCLUDED_TASKS)
from rage.experiments.v7_amendment import (report_view, compatible_identity,
    freeze_training, approved_prefix_contract, write_report_views)
from rage.experiments.v7_train import checkpoint_curve
from rage.experiments.train_eval import fine_tune
from test_training_resume import engine
from test_production_recipe import base_recipe


def configs():
    base=json.loads(Path('configs/rage/main_v7.json').read_text())
    return make_curve_config(base),make_reduced_config(base)


def test_reduced_exact_config_assets_and_steps():
    old,new=configs()
    validate(new)
    assert json.loads(Path('configs/rage/main_v7_curve4.json').read_text())==new
    assert asset_protocol(old)==asset_protocol(new)
    assert len(new['datasets'])==7 and not set(EXCLUDED_TASKS)&set(new['datasets'])
    assert len(new['datasets'])*len(new['arms'])*3*3==1260
    a=exposure_audit(new)
    assert a['chronos-base']['max_steps']==1527
    assert a['timesfm-2.5']['max_steps']==a['timerxl-84m']['max_steps']==2036
    assert all(r['schedule_steps']==3054 for r in a.values())
    bad=deepcopy(new);bad['settings']['chronos-base']['learning_rate']=.1
    with pytest.raises(ValueError):validate(bad)


def test_freeze_training_archives_only_approved_previous_protocol(tmp_path):
    old,new=configs()
    freeze_training(tmp_path,old)
    freeze_training(tmp_path,new)
    freeze_training(tmp_path,new)
    assert json.loads((tmp_path/'training_protocol.json').read_text())==new['training_protocol']
    history=list((tmp_path/'training_protocol_history').glob('*.json'))
    assert len(history)==1 and json.loads(history[0].read_text())==old['training_protocol']
    with pytest.raises(ValueError):freeze_training(tmp_path,old)


def raw_result(arm, identity):
    records=[dict(epoch=i/2,step=int(np.ceil(16284*i/2/32)),metrics={'mase':10-i}) for i in range(1,9)]
    return dict(arm=arm,identity=identity,mase=2,final_epoch=4,summary_epoch=4,
        checkpoints=records,test_curve=[dict(epoch=0,metrics={'mase':10})]+records,
        train={'source':{'steps':2036}},elapsed_seconds=123)


def test_raw_curve_prefix_reuse_does_not_overwrite_results(tmp_path):
    old,new=configs();new=deepcopy(new);new['arms']=['orig','rand','grad-dot']
    expected=dict(model='fixture',training_protocol=new['training_protocol'])
    legacy=dict(expected,training_protocol=old['training_protocol'])
    (tmp_path/'cells').mkdir()
    for arm in new['arms']:
        row=raw_result(arm,legacy)
        (tmp_path/'cells'/f'task__{arm}.json').write_text(json.dumps(row))
    before={p.name:p.read_bytes() for p in (tmp_path/'cells').glob('*')}
    write_report_views(tmp_path,'task',new,'chronos-base',expected)
    assert before=={p.name:p.read_bytes() for p in (tmp_path/'cells').glob('*')}
    report=json.loads((tmp_path/'reported_cells/task__grad-dot.json').read_text())
    assert report['mase']==4 and report['final_epoch']==3 and report['reported_step']==1527
    assert len(report['checkpoints'])==6 and report['executed_final_epoch']==4
    assert report['train']['source']['steps']==2036 and report['elapsed_seconds']==123
    assert report['improvement_vs_rand']==0
    assert report_view(raw_result('rand',legacy),expected,new,'timerxl-84m')['final_epoch']==4
    assert not compatible_identity(dict(legacy,model='wrong'),expected,new)
    short=raw_result('rand',legacy);short['checkpoints']=short['checkpoints'][:5]
    with pytest.raises(ValueError,match='incomplete'):report_view(short,expected,new,'chronos-base')


def test_three_epoch_checkpoint_grid():
    raw=raw_result('rand',{})
    raw['checkpoints']=raw['checkpoints'][:6]
    assert len(checkpoint_curve(raw,{'mase':1},3))==7
    with pytest.raises(ValueError):checkpoint_curve(raw,{'mase':1},4)


def test_prefix_resume_matches_uninterrupted_three_epoch(tmp_path):
    native=engine()
    base={k:v.detach().clone() for k,v in native.model.state_dict().items()}
    windows=np.arange(120,dtype=np.float32).reshape(20,6)/100
    recipe=base_recipe(epochs=4,batch_size=3,micro_batch_size=3,learning_rate=1e-5)
    path=tmp_path/'resume.pt'
    def stop(step):
        if step==3:raise RuntimeError('interrupt')
    with pytest.raises(RuntimeError,match='interrupt'):
        fine_tune(native,windows,steps=2036,batch_size=3,learning_rate=1e-5,seed=7,
            recipe=recipe,schedule_total_steps=3054,resume_path=path,checkpoint_steps=[2],on_step=stop)
    state=torch.load(path,weights_only=False)
    new_contract=deepcopy(state['contract']);new_contract['steps']=1527;new_contract['recipe']['epochs']=3
    assert approved_prefix_contract(state['contract'],new_contract)
    changed=deepcopy(new_contract);changed['schedule_total_steps']=1527
    assert not approved_prefix_contract(state['contract'],changed)
    native.allow_v7_epoch_prefix_resume=True
    result=fine_tune(native,windows,steps=1527,batch_size=3,learning_rate=1e-5,seed=7,
        recipe=replace(recipe,epochs=3),schedule_total_steps=3054,resume_path=path)
    assert result['resumed_from_step']==2 and result['steps']==1527
    final={k:v.detach().clone() for k,v in native.model.state_dict().items()}
    native.model.load_state_dict(base)
    fine_tune(native,windows,steps=1527,batch_size=3,learning_rate=1e-5,seed=7,
        recipe=replace(recipe,epochs=3),schedule_total_steps=3054)
    for k in final:torch.testing.assert_close(native.model.state_dict()[k],final[k],rtol=0,atol=0)
