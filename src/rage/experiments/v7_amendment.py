"""Explicit amendment: preserve raw cells, derive current reporting views."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from assets.gradient.cache import atomic_json
from .v7_protocol import CURVE_PROTOCOL, REDUCED_PROTOCOL, make_curve_config, asset_protocol


def compatible_identity(actual, expected, config):
    if actual == expected:
        return True
    if config.get('training_protocol',{}).get('version') != REDUCED_PROTOCOL:
        return False
    legacy = dict(expected, training_protocol=make_curve_config(asset_protocol(config))['training_protocol'])
    return actual == legacy


def freeze_training(run, config):
    from .v7_run import freeze
    path = Path(run)/'training_protocol.json'
    new = config['training_protocol']
    if path.exists() and json.loads(path.read_text()) != new:
        old = json.loads(path.read_text())
        if (new.get('version') != REDUCED_PROTOCOL or
                old != make_curve_config(asset_protocol(config))['training_protocol']):
            raise ValueError('Unapproved training protocol migration')
        freeze(Path(run)/'training_protocol_history'/f'{CURVE_PROTOCOL}.json',old)
        atomic_json(path,new)
    else:
        freeze(path,new)


def report_view(raw, expected, config, student):
    if not compatible_identity(raw.get('identity'), expected, config):
        raise ValueError('Completed V7 cell identity differs')
    view = deepcopy(raw)
    view['identity'] = expected
    if raw['arm'] != 'orig':
        maximum = config['settings'][student]['epochs']
        records = [r for r in raw['checkpoints'] if r['epoch'] <= maximum]
        if [r['epoch'] for r in records] != [i/2 for i in range(1,2*maximum+1)]:
            raise ValueError('Required reporting prefix is incomplete')
        if any(not np.isfinite(r['metrics']['mase']) for r in records):
            raise ValueError('Nonfinite reporting prefix')
        view.update(records[-1]['metrics'])
        view['checkpoints'] = records
        view['test_curve'] = [r for r in raw['test_curve'] if r['epoch'] <= maximum]
        view['final_epoch'] = view['summary_epoch'] = maximum
        view['reported_step'] = records[-1]['step']
        view['executed_final_epoch'] = raw['final_epoch']
        view['raw_training_recipe'] = raw.get('training_recipe')
        view['reporting_policy'] = 'prefix_without_retraining_raw_costs_preserved'
    return view


def write_report_views(root, task, config, student, identity):
    target = Path(root)/'reported_cells'
    target.mkdir(parents=True,exist_ok=True)
    rows = {a: report_view(json.loads((Path(root)/'cells'/f'{task}__{a}.json').read_text()),
                          identity, config, student) for a in config['arms']}
    for arm, row in rows.items():
        row['improvement_vs_rand'] = 100*(1-row['mase']/rows['rand']['mase'])
        row['improvement_vs_orig'] = 100*(1-row['mase']/rows['orig']['mase'])
        row['raw_result_path'] = str(Path(root)/'cells'/f'{task}__{arm}.json')
        atomic_json(target/f'{task}__{arm}.json',row)


def approved_prefix_contract(old, new):
    """Only stopping epoch changes; optimizer/schedule/sampling remain exact."""
    if old.get('steps') != 2036 or new.get('steps') != 1527:
        return False
    changed = deepcopy(old)
    if changed.get('recipe',{}).get('epochs') != 4 or new.get('recipe',{}).get('epochs') != 3:
        return False
    changed['steps'] = 1527
    changed['recipe']['epochs'] = 3
    return changed == new
