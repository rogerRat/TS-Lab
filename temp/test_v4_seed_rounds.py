from copy import deepcopy
import json
from pathlib import Path
import pytest

from rage.experiments.students import atomic_json
from rage.experiments.v4_protocol import validate_v4_config
from rage.experiments.v4_seed_rounds import run_rounds,round_completion,round_config


def template(tmp_path):
    c=json.loads(Path('configs/rage/main_v4.json').read_text())
    cache=tmp_path/'existing-gradients';cache.mkdir()
    c['gradient_cache_source']=str(cache)
    return c


def enabled(parent):
    parent.mkdir(exist_ok=True)
    atomic_json(parent/'execution_control.json',dict(mode='run',selection_enabled=True,
        training_enabled=True,evaluation_enabled=True))


def publish(config,folder,*,omit_student=None):
    for student in config['students']:
        (folder/'students'/student/'selections').mkdir(parents=True,exist_ok=True)
        if student==omit_student:continue
        for dataset in config['datasets']:
            for arm in config['arms']:
                seed=0 if arm=='orig' else config['seeds'][0]
                p=folder/'students'/student/'cells'/f'{dataset}__{arm}__seed{seed}.json'
                if p.exists():continue
                atomic_json(p,dict(student=student,dataset=dataset,arm=arm,seed=seed,
                    experiment_version='v4',mase=1.))


def test_v4_models_and_single_natural_seed_are_enforced(tmp_path):
    c=template(tmp_path);validate_v4_config(c);validate_v4_config(round_config(c,2))
    for change in [{'students':['chronos-base','timesfm-2.5','timemoe-200m']},
                   {'seeds':[0,1]},{'seeds':[3]},{'seeds':[-1]},{'seeds':[True]},
                   {'student_parallelism':1}]:
        with pytest.raises(ValueError):validate_v4_config(dict(c,**change))


def test_complete_round_barrier_and_shared_static_assets(tmp_path):
    c=template(tmp_path);parent=tmp_path/'suite';enabled(parent);observed=[]
    def runner(config,folder,execution):
        seed=config['seeds'][0];observed.append(seed)
        validate_v4_config(config)
        assert execution['student_parallelism']==2
        assert not (parent/'rounds'/f'seed{seed+1}').exists()
        for earlier in range(seed):
            assert round_completion(round_config(c,earlier),parent/'rounds'/f'seed{earlier}')['complete']
        publish(config,folder)
        return 0
    run_rounds(c,parent,max_seeds=3,runner=runner)
    assert observed==[0,1,2]
    assert json.loads((parent/'state.json').read_text())['completed_seeds']==[0,1,2]
    later=parent/'rounds'/'seed2'
    assert (later/'grad_cache').resolve()==Path(c['gradient_cache_source'])
    assert (later/'students'/'chronos-base'/'selections').is_symlink()
    assert (later/'students'/'chronos-base'/'cells'/f"{c['datasets'][0]}__orig__seed0.json").is_symlink()


def test_missing_model_blocks_next_seed_even_with_zero_exit_then_resume(tmp_path):
    c=template(tmp_path);parent=tmp_path/'suite';enabled(parent)
    def incomplete(config,folder,execution):
        publish(config,folder,omit_student='timesfm-2.5');return 0
    with pytest.raises(RuntimeError,match='next seed blocked'):
        run_rounds(c,parent,max_seeds=2,runner=incomplete)
    assert not (parent/'rounds'/'seed1').exists()
    observed=[]
    def finish(config,folder,execution):
        observed.append(config['seeds'][0]);publish(config,folder);return 0
    run_rounds(c,parent,max_seeds=2,runner=finish)
    assert observed==[0,1]


def test_worker_failure_blocks_next_round(tmp_path):
    c=template(tmp_path);parent=tmp_path/'suite';enabled(parent)
    def failed(config,folder,execution):raise RuntimeError('model failed')
    with pytest.raises(RuntimeError,match='model failed'):
        run_rounds(c,parent,max_seeds=2,runner=failed)
    assert not (parent/'rounds'/'seed1').exists()
    assert json.loads((parent/'state.json').read_text())['status']=='failed'


def test_invalid_cell_identity_cannot_release_barrier(tmp_path):
    c=template(tmp_path);folder=tmp_path/'seed0';publish(c,folder)
    assert round_completion(c,folder)['complete']
    p=next((folder/'students'/'chronos-base'/'cells').glob('*__rand__seed0.json'))
    record=json.loads(p.read_text());record['seed']=9;atomic_json(p,record)
    assert round_completion(c,folder)['missing']==[str(p)]


def test_prepared_execution_controls_cannot_launch(tmp_path):
    c=template(tmp_path);parent=tmp_path/'suite';parent.mkdir()
    atomic_json(parent/'execution_control.json',dict(mode='prepared',training_enabled=False))
    def forbidden(*args,**kwargs):raise AssertionError('Must not launch')
    with pytest.raises(RuntimeError,match='paused'):
        run_rounds(c,parent,max_seeds=1,runner=forbidden)
    assert not (parent/'rounds').exists()



def test_v4_summary_recognizes_orig_rand_and_actual_round_seed(tmp_path):
    from rage.experiments.full_pool_run import summarize
    c=round_config(template(tmp_path),2);folder=tmp_path/'seed2';publish(c,folder)
    summary=summarize(folder,c)
    assert summary['complete']
    assert summary['completed_cells']==234
    assert len([r for r in summary['cells'] if r['arm']=='orig'])==18
    assert all(p['baseline']=='rand' for p in summary['paired_differences'])
    assert all(p['pairs'][0]['seed']==2 for p in summary['paired_differences'])


def test_default_schedule_stops_after_exactly_three_seeds(tmp_path):
    c=template(tmp_path);parent=tmp_path/'suite';enabled(parent);seen=[]
    def runner(config,folder,execution):
        seen.append(config['seeds'][0]);publish(config,folder);return 0
    run_rounds(c,parent,runner=runner)
    assert seen==[0,1,2]
    assert not (parent/'rounds'/'seed3').exists()
    assert json.loads((parent/'state.json').read_text())['status']=='complete'
