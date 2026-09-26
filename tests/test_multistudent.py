"""Offline contracts for the three-student main table (no real-data jobs)."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from adapters.foundation import build_adapter, get_model_spec
from adapters.foundation.training import FineTuneConfig
from rags.experiments.metrics import point_metrics
from rags.experiments.students import (STUDENTS, adapter_engine, atomic_json,
                                      run_student_task, worker_command)
from rags.experiments.student_report import summarize_students


class TinyAdapter:
    def __init__(self, key):
        self.spec = get_model_spec(key)
        torch.manual_seed(42)
        self._model = torch.nn.Linear(3, 1)

    def load(self):
        return self._model

    def training_loss(self, windows, **kwargs):
        x = torch.tensor(windows)
        return (self._model(x[:, :3]) - x[:, 3:]).square().mean()

    def forecast(self, contexts, **kwargs):
        with torch.no_grad():
            return self._model(torch.tensor(contexts)).numpy()


def make_manifest(tmp_path):
    windows = np.arange(32, dtype=np.float32).reshape(8, 4) / 10
    selection, exam = tmp_path / 'selection.npz', tmp_path / 'exam.npz'
    np.savez(selection, windows=windows, ids=np.array([(0, i*4+1) for i in range(8)]))
    np.savez(exam, windows=windows[:2], entities=np.array(['ETT_1H:a']*2), scales=np.ones(2))
    manifest = {'dataset':'ETT_1H', 'context_length':3, 'prediction_length':1,
                'training': {'epochs':4, 'batch_size':3, 'learning_rate':1e-5,
                             'weight_decay':.01,'gradient_clipping':None,'dtype':'float32','seeds':[0]},
                'exam': {'path':str(exam)},
                'selections': {'random':{'path':str(selection),
                                         'cost':{'wall_seconds':12.}}}}
    path = tmp_path / 'task.json'
    atomic_json(path,manifest)
    return path,manifest


def test_shared_artifact_workers_all_checkpoints_resume_and_restore(tmp_path):
    path,manifest=make_manifest(tmp_path)
    for student in ('timesfm-2.5','timemoe-200m'):
        engine=adapter_engine(TinyAdapter(student),3,1)
        original={k:v.clone() for k,v in engine.model.state_dict().items()}
        out=tmp_path/'students'/student
        run_student_task(student,path,out,engine=engine)
        cell=out/'cells'/'ETT_1H__random__seed0.json'
        result=json.loads(cell.read_text())
        assert [r['step'] for r in result['checkpoints']]==[3,6,8,11]
        assert result['selection_artifact']==manifest['selections']['random']['path']
        assert result['mase']==result['checkpoints'][-1]['metrics']['mase']
        assert 'sql' not in result and 'mae' not in result
        assert 'selection_cost' not in result  # Shared once, not billed to every student.
        for k,v in engine.model.state_dict().items():
            torch.testing.assert_close(v,original[k],rtol=0,atol=0)
        before=cell.read_bytes()
        run_student_task(student,path,out,engine=engine)
        assert cell.read_bytes()==before
        assert (out/'RUN.md').is_file()


def test_metric_series_equal_weight_zero_scale_separate():
    windows=np.zeros((4,4))
    result=point_metrics(windows,np.array([[1.],[1.],[5.],[10.]]),['a','a','b','constant'],
                         context_length=3,scales=np.array([1.,1.,1.,0.]))
    assert result['mase']==3.
    assert result['constant_series_mae']=={'constant':10.}
    assert 'sql' not in result


@pytest.mark.parametrize('student', STUDENTS)
def test_duplicate_selection_rows_reach_training_unchanged(tmp_path, monkeypatch, student):
    path, manifest = make_manifest(tmp_path)
    windows = np.repeat(np.arange(8, dtype=np.float32).reshape(2, 4), 4, axis=0)
    ids = np.repeat(np.array([(0, 1), (0, 5)]), 4, axis=0)
    np.savez(tmp_path / 'selection.npz', windows=windows, ids=ids)
    manifest['selections'] = {'tsds': manifest['selections']['random']}
    atomic_json(path, manifest)
    from rags.experiments import train_eval
    original_train = train_eval.run_arm_training
    received = []

    def train(*args, **kwargs):
        received.append(kwargs['source_windows'].copy())
        return original_train(*args, **kwargs)

    monkeypatch.setattr(train_eval, 'run_arm_training', train)
    # Inject the same tiny differentiable engine into all three worker routes.
    engine = adapter_engine(TinyAdapter('timesfm-2.5'), 3, 1)
    out = tmp_path / student
    run_student_task(student, path, out, engine=engine)
    np.testing.assert_array_equal(received[0], windows)
    result = json.loads((out / 'cells' / 'ETT_1H__tsds__seed0.json').read_text())
    assert [row['step'] for row in result['checkpoints']] == [3, 6, 8, 11]
    assert result['selection_coverage']['selected_windows'] == 8
    assert result['selection_coverage']['unique_windows'] == 2


def test_excluded_models_cannot_become_students_and_env_is_explicit(tmp_path):
    for key in ('sundial-base-128m','timesfm-3.0'):
        assert build_adapter(key)  # Historical adapters retained, not deployed.
        with pytest.raises(ValueError,match='excluded'):
            worker_command(key,tmp_path/'task.json',tmp_path/'out')
    command=worker_command('timemoe-200m',tmp_path/'task.json',tmp_path/'out')
    assert command[:6]==['conda','run','--no-capture-output','-n','ts3-timemoe','python']
    assert worker_command('timemoe-200m',tmp_path/'t',tmp_path/'o',
                           environments={'timemoe-200m':'/env/bin/python'})[0]=='/env/bin/python'
    assert FineTuneConfig(512,64).weight_decay==.01
    assert FineTuneConfig(512,64).max_grad_norm is None


def test_student_summary_does_not_mix_models_or_repeat_selection_cost(tmp_path):
    path,manifest=make_manifest(tmp_path)
    (tmp_path/'tasks').mkdir()
    atomic_json(tmp_path/'tasks'/'ETT_1H.json',manifest)
    for student in STUDENTS:
        for arm,value in [('original',3),('random',2),('rags',1)]:
            atomic_json(tmp_path/'students'/student/'cells'/f'{arm}.json',
                        {'student':student,'dataset':'ETT_1H','arm':arm,'seed':0,'mase':value})
    result=summarize_students(tmp_path,[SimpleNamespace(dataset='ETT_1H')],[0],
                              ['original','random','rags'],STUDENTS)
    assert len(result['selection_costs_shared_once'])==1
    assert result['students']['timemoe-200m']['rags']['vs_random']['ETT_1H']==.5
    assert result['students']['chronos-base']['rags']['average_rank_complete_tasks']==1
    assert 'sql' not in json.dumps(result)
    assert (tmp_path/'student_comparison.html').is_file()


def test_official_timemoe_batch_is_shifted_all_selected_points_not_exam():
    from adapters.foundation.timemoe import native_training_batch
    x=np.array([[1,2,3,4],[5,5,5,5]],dtype=np.float32)
    batch=native_training_batch(x,device='cpu')
    expected=(x[0]-x[0].mean())/x[0].std()
    np.testing.assert_allclose(batch['input_ids'][0],expected[:-1])
    np.testing.assert_allclose(batch['labels'][0],expected[1:])
    assert torch.equal(batch['labels'][1],torch.zeros(3))
    assert batch['loss_masks'].sum()==6


def test_fev_scale_only_uses_history_before_each_forecast(monkeypatch,tmp_path):
    from rags.experiments import tasks
    values=np.arange(2000,dtype=np.float64)
    monkeypatch.setattr(tasks,'load_bundle_index',lambda path: None)
    monkeypatch.setattr(tasks,'_usable_full_series',lambda *a,**k: [('ETT_1H:a',values)])
    spec=tasks.TaskSpec(dataset='ETT_1H',load_dir=tmp_path)
    before=tasks.build_task_data(spec,context_length=512,prediction_length=64,pool_fragments=[])
    values[-64:]+=100000  # final exam future: cannot affect any forecast's y_past scale.
    after=tasks.build_task_data(spec,context_length=512,prediction_length=64,pool_fragments=[])
    np.testing.assert_array_equal(before.test_scales,after.test_scales)
    np.testing.assert_array_equal(before.reference_fragments[0].data,after.reference_fragments[0].data)


def test_main_table_selects_once_per_arm_then_deploys_all_students(monkeypatch,tmp_path):
    from rags.experiments import main_table as main
    from rags.experiments.arms import ArmOutput,SelectionCost
    windows=np.arange(32,dtype=np.float32).reshape(8,4)/10
    task=SimpleNamespace(reference_fragments=[],test_windows=windows[:2],
                         test_entities=['ETT_1H:a']*2,test_scales=np.ones(2),constant_series=[])
    monkeypatch.setattr(main,'build_task_specs',lambda **kw:[SimpleNamespace(dataset='ETT_1H')])
    monkeypatch.setattr(main,'build_task_data',lambda *a,**k:task)
    monkeypatch.setattr(main,'build_ratio_pool',lambda **k:([],{'total_points':32,'total_equivalents':8}))
    monkeypatch.setattr(main,'WindowIndex',lambda *a,**k:list(range(8)))
    class Engine:
        def __init__(self,*a,**k):
            self.model=torch.nn.Linear(3,1)
            self.parameter_count=4
            self.device='cpu'
    monkeypatch.setattr(main,'ChronosT5Engine',Engine)
    calls=[]
    def selected(name):
        calls.append(name)
        return ArmOutput(name=name,selected=windows,window_ids=[(0,1+4*i) for i in range(8)],
                         cost=SelectionCost(),diagnostics={})
    monkeypatch.setattr(main,'rags_crb_select',lambda *a,**k:selected('rags'))
    monkeypatch.setattr(main,'streaming_random_select',lambda *a,**k:selected('random'))
    monkeypatch.setattr(main,'trajectory_level_select',lambda *a,**k:selected('trajectory_level'))
    monkeypatch.setattr(main,'run_baseline',lambda name,**k:selected(name))
    deployed=[]
    def deploy(students,path,out,**kw):
        payload=json.loads(path.read_text())
        assert len(payload['selections'])==7
        for student in students:
            deployed.append(student)
            run_student_task(student,path,out/'students'/student,
                             engine=adapter_engine(TinyAdapter('timesfm-2.5'),3,1))
    monkeypatch.setattr(main,'deploy_students',deploy)
    argv=['--datasets','ETT_1H','--context-length','3','--prediction-length','1',
          '--n-windows','8','--train-batch','3','--seeds','0','--out',str(tmp_path/'main')]
    assert main.main(argv)==0
    assert len(calls)==7 and len(set(calls))==7
    assert deployed==list(STUDENTS)
    assert len(list((tmp_path/'main'/'students').glob('*/cells/*.json')))==24
    # Resume reuses the seven selections and completed cells, no new selection calls.
    assert main.main(argv)==0
    assert len(calls)==7
    rows=[json.loads(line) for line in
          (tmp_path/'main'/'selection_metrics.jsonl').read_text().splitlines()]
    assert len(rows)==16  # eight arms per invocation, shared across all students
    assert [r['cache_state'] for r in rows[:8]]==['not_applicable']+['cold']*7
    assert [r['cache_state'] for r in rows[8:]]==['not_applicable']+['selection_cache_hit']*7
    assert {r['arm'] for r in rows}==set(calls)|{'original'}
