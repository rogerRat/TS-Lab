"""2026-09-11 epoch protocol and forecast-exam regressions."""
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from rage.experiments import train_eval
from rage.experiments.schedule import epoch_checkpoints
from rage.experiments.metrics import quantile_forecasts
from rage.experiments.checkpoint_report import write_checkpoint_report


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)

    def forward(self, input_ids, **kwargs):
        logits = self.linear(input_ids.float())[:, None, :].expand(-1, 2, -1)
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids, **kwargs):
        assert 'labels' not in kwargs
        assert input_ids.shape[1] == 2
        logits = self(input_ids).logits
        return SimpleNamespace(scores=tuple(logits[:, i] for i in range(2)),
                               sequences=torch.cat([torch.zeros(len(logits), 1, dtype=torch.long),
                                                    logits.argmax(-1)], dim=1))


def engine():
    torch.manual_seed(42)
    return SimpleNamespace(model=Model(), device='cpu',
        config=SimpleNamespace(context_length=2, prediction_length=2),
        batch_tensors=lambda w: {'input_ids': torch.tensor(w[:, :2]),
                                'labels': torch.tensor(w[:, 2:]).long() % 3},
        tokenizer=SimpleNamespace(
            context_input_transform=lambda x: (x, torch.ones_like(x), torch.ones(len(x))),
            output_transform=lambda tokens, scale: (tokens.float() + 10) * scale[:, None, None]))


@pytest.mark.parametrize('n,steps', [(1024,128),(2048,256),(4096,512),(8192,1024)])
def test_document_epoch_steps(n, steps):
    assert epoch_checkpoints(n,32,4)[-1] == (4,steps)
    assert len(epoch_checkpoints(n,32,4)) == 4


def test_native_baseline_coverage_counts_overlap_and_repeats_once():
    from rage.experiments.deploy import selection_coverage
    stats=selection_coverage([(1,1),(1,1),(1,3),(2,1)],4)
    assert stats == {'selected_windows':4,'unique_windows':3,'unique_points':10,
                     'disjoint_window_equivalents':2.5, 'budget_windows':4, 'budget_points':16,
                     'delivered_units':4, 'delivered_points':16, 'repeated_points':6,
                     'overflow_points':0, 'overflow_ratio':0., 'legal_window_occurrences':4}


def test_epoch_boundaries_use_cumulative_ceiling_not_rounded_epoch_lengths():
    assert epoch_checkpoints(8,3,4) == [(1,3),(2,6),(3,8),(4,11)]


def test_continuous_optimizer_all_checkpoints_final_not_best_and_restore(tmp_path, monkeypatch):
    model_engine = engine()
    base = {k: v.clone() for k,v in model_engine.model.state_dict().items()}
    windows = np.arange(32,dtype=np.float32).reshape(8,4) / 10
    scores = iter([4.,1.,2.,3.])
    def evaluate(*args, **kwargs):
        value = next(scores)
        return {'mase':value, 'sql':value, 'mae':value}
    monkeypatch.setattr(train_eval,'evaluate',evaluate)
    result = train_eval.run_arm_training(model_engine, source_windows=windows,
        source_epochs=4,batch_size=3,learning_rate=.001,seed=5,
        checkpoint_dir=tmp_path, test_windows=windows[:1],
        test_entities=['ETT_1H'],context_length=2,base_state=base)
    assert result['mase'] == 3.  # Not the best checkpoint's 1.0.
    assert [r['step'] for r in result['checkpoints']] == [3,6,8,11]
    assert result['train']['source']['sample_exposures'] == 33
    assert len(list(tmp_path.glob('epoch-*.pt'))) == 4
    for k,v in model_engine.model.state_dict().items():
        torch.testing.assert_close(v,base[k],rtol=0,atol=0)

    uninterrupted = engine()
    train_eval.fine_tune(uninterrupted,windows,steps=11,batch_size=3,learning_rate=.001,seed=5)
    final = torch.load(tmp_path/'epoch-04.pt',weights_only=True)
    for k,v in uninterrupted.model.state_dict().items():
        torch.testing.assert_close(v,final[k],rtol=0,atol=0)


def test_restore_on_checkpoint_evaluation_error(tmp_path, monkeypatch):
    e=engine()
    before={k:v.clone() for k,v in e.model.state_dict().items()}
    def fail(*args,**kwargs):
        raise RuntimeError('exam failed')
    monkeypatch.setattr(train_eval,'evaluate',fail)
    with pytest.raises(RuntimeError,match='exam failed'):
        train_eval.run_arm_training(e,source_windows=np.ones((4,4),dtype=np.float32),
            source_epochs=1,batch_size=2,learning_rate=.001,seed=0,
            checkpoint_dir=tmp_path,test_windows=np.ones((1,4)),
            test_entities=['ETT_1H'],context_length=2)
    for k,v in e.model.state_dict().items():
        torch.testing.assert_close(v,before[k],rtol=0,atol=0)


def test_future_labels_cannot_change_forecasts_and_tokenizer_decodes_bins():
    e=engine()
    a=np.array([[1.,2.,3.,4.]],dtype=np.float32)
    b=np.array([[1.,2.,300.,-400.]],dtype=np.float32)
    qa,_=quantile_forecasts(e,a)
    qb,_=quantile_forecasts(e,b)
    np.testing.assert_array_equal(qa,qb)
    assert qa.shape == (1,2,9)
    assert qa.min() >= 10  # Uses official tokenizer transform, not guessed bin offsets.


def test_checkpoint_report_retains_all_epochs(tmp_path):
    (tmp_path/'cells').mkdir()
    cell={'dataset':'ETT_1H','arm':'rage','seed':0,'n_windows':4096,'checkpoints':[
        {'epoch':e,'step':e*128,'checkpoint':f'epoch-{e}.pt',
         'metrics':{'mase':1/e,'sql':2/e,'mae':3/e}} for e in range(1,5)]}
    (tmp_path/'cells'/'cell.json').write_text(json.dumps(cell))
    rows=write_checkpoint_report(tmp_path)
    assert len(rows)==4
    assert (tmp_path/'checkpoint_curve.csv').is_file()
    assert (tmp_path/'checkpoint_curve.html').is_file()
    assert (tmp_path/'cost_curve.html').is_file()
    assert (tmp_path/'data_curve.html').is_file()
    assert [r['epoch'] for r in json.loads((tmp_path/'checkpoint_curve.json').read_text())]==[1,2,3,4]


def test_formal_config_is_epoch_based():
    from configs.paths import PROJECT_ROOT
    import yaml
    config=yaml.safe_load((PROJECT_ROOT/'configs/rage/run.yaml').read_text())
    assert config['source_epochs']==4
    assert config['learning_rate']==1e-5
    assert 'source_steps' not in config and 'schedule_variants' not in config


def test_zero_time_budget_does_not_train_or_fabricate_epoch_checkpoints(monkeypatch):
    e=engine()
    monkeypatch.setattr(train_eval,'evaluate',lambda *a,**kw: {'mase':1.,'sql':1.,'mae':1.})
    result=train_eval.run_arm_training(e,source_windows=np.ones((4,4),dtype=np.float32),
        source_epochs=4,training_seconds=0,batch_size=2,learning_rate=.001,seed=0,
        test_windows=np.ones((1,4)),test_entities=['ETT_1H'],context_length=2)
    assert result['checkpoints']==[]
    assert result['final_epoch']==0
    assert result['train']['source']['steps']==0


def test_deployment_reports_missing_resources_without_launching(tmp_path,monkeypatch):
    from rage.experiments import deployment
    monkeypatch.setattr(deployment.shutil,'disk_usage',lambda path: SimpleNamespace(free=0))
    result=deployment.check_deployment({
        'source_load':str(tmp_path/'source'), 'reference_root':str(tmp_path/'ref'),
        'model_path':str(tmp_path/'model'), 'source_epochs':4,
        'n_windows':4096,'train_batch':32,'datasets':['ETT_1H'],'seeds':[0],
    },out_root=tmp_path)
    assert not result['ready']
    assert result['checkpoint_steps']==[(1,128),(2,256),(3,384),(4,512)]
    assert any('source data' in b for b in result['blockers'])
    assert any('reference data' in b for b in result['blockers'])
    assert any('Chronos-small' in b for b in result['blockers'])
    assert any('GiB free' in b for b in result['blockers'])
