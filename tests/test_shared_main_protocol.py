import copy
import json
from pathlib import Path
import pytest
from rage.experiments.full_pool_run import validate_config, summarize
from rage.experiments.formal_main_entry import preflight


def configuration():
    config=json.loads(Path('configs/rage/main.json').read_text())
    config['shared_task_decisions']=str(Path('configs/rage/shared_tasks.json').resolve())
    return config


def test_exact_shared_ten_recipe_budget_and_fractional_epoch():
    config=configuration()
    validate_config(config)
    assert len(config['datasets'])==9
    assert config['settings']['timesfm-2.5']['epochs']==2.625
    for key,value in [('epochs',2.5),('weight_decay',0),('schedule_epochs',2.625)]:
        bad=copy.deepcopy(config);bad['settings']['timesfm-2.5'][key]=value
        with pytest.raises(ValueError,match='recipe differs'):validate_config(bad)
    bad=copy.deepcopy(config);bad['crb']['split_density']=.005
    with pytest.raises(ValueError,match='CRB differs'):validate_config(bad)
    with pytest.raises(ValueError,match='reviewed'):
        preflight(config,dict(ready=True,config={}))


def test_paired_summary_uses_same_seed_only(tmp_path):
    folder=tmp_path/'students'/'m'/'cells';folder.mkdir(parents=True)
    for arm,seed,mase in [('orig',0,4),('rand',0,2),('rand',1,4),('rage-block-maxSum',0,1),('rage-block-maxSum',1,2)]:
        (folder/f't__{arm}__seed{seed}.json').write_text(json.dumps(dict(student='m',dataset='t',arm=arm,seed=seed,mase=mase)))
    result=summarize(tmp_path,dict(students=['m'],datasets=['t'],arms=['rand','rage-block-maxSum'],seeds=[0,1,2]))
    assert not result['complete']
    row=result['paired_differences'][0]
    assert row['mean_delta_mase']==-1.5
    assert [p['seed'] for p in row['pairs']]==[0,1]
    assert row['std_delta_mase']==pytest.approx(2**-.5)
