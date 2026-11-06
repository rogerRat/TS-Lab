import json
import numpy as np
import pytest
import torch
from test_training_resume import engine
from test_production_recipe import base_recipe
from rage.experiments.train_eval import _online_curve_recipe
from rage.experiments.checkpoint_retention import cleanup_completed_weights
from rage.experiments.v7_train import checkpoint_curve

def test_four_epoch_online_resume_and_immediate_retention(tmp_path):
    native=engine()
    base={k:v.detach().clone() for k,v in native.model.state_dict().items()}
    windows=np.arange(120,dtype=np.float32).reshape(20,6)/100
    recipe=base_recipe(epochs=4,batch_size=3,micro_batch_size=3,learning_rate=1e-5)
    captured=[]
    def assess():
        torch.rand(20)  # Evaluation must not perturb training RNG/dropout.
        captured.append({k:v.detach().clone() for k,v in native.model.state_dict().items()})
        return {'mase':float(native.model[0].weight.detach().square().sum())}
    kwargs=dict(base_state=base,recipe=recipe,seed=7,
        checkpoints=[(e,int(e*4)) for e in [.5,1.,1.5,2.,2.5,3.,3.5,4.]],
        schedule_total_steps=24,evaluate_fn=assess)
    expected=_online_curve_recipe(native,windows,**kwargs)
    final=captured[-1]
    path=tmp_path/'resume.pt'
    curve_path=tmp_path/'curve.json'
    def interrupt(row):
        if row['epoch']==1:
            raise RuntimeError('evaluation persisted, resume not yet promoted')
    with pytest.raises(RuntimeError,match='not yet promoted'):
        _online_curve_recipe(native,windows,**kwargs,on_record=interrupt,
            resume_path=path,observations_path=curve_path)
    assert torch.load(path,weights_only=False)['step']==2
    assert path.with_suffix('.pt.tmp').exists()
    actual=_online_curve_recipe(native,windows,**kwargs,
        resume_path=path,observations_path=curve_path)
    assert [r['metrics'] for r in actual['curve']]==[r['metrics'] for r in expected['curve']]
    for k in final:
        torch.testing.assert_close(captured[-1][k],final[k],rtol=0,atol=0)
    assert torch.load(path,weights_only=False)['step']==16
    assert not path.with_suffix('.pt.tmp').exists()
    assert list(tmp_path.glob('epoch-*.pt'))==[]
    records=checkpoint_curve({'checkpoints':actual['curve']},{'mase':3})
    assert len(records)==9
    captured.clear()
    _online_curve_recipe(native,windows,**kwargs,resume_path=path,observations_path=curve_path)
    assert not captured  # Completed work is reused, not evaluated/trained again.
    result=tmp_path/'cell.json'
    result.write_text(json.dumps({'mase':records[-1]['metrics']['mase'],'test_curve':records}))
    cleanup_completed_weights(result,tmp_path)
    assert not path.exists() and curve_path.exists() and result.exists()
    cleanup_completed_weights(result,tmp_path)  # crash-after-results cleanup is idempotent
