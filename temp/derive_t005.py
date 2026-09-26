"""Derive new probabilities from frozen V3 scores, retaining explicit provenance."""
import json
import os
from pathlib import Path
import re
import time

import numpy as np
from rags.experiments.regions import load_selection, save_selection
from rags.experiments.resampling import value_mass
from rags.experiments.students import atomic_json
from rags.experiments.v3_protocol import V3_MATRIX, V3_TEMPERATURE_ARMS, validate_v3_config

old = Path('/root/autodl-tmp/TS3/outputs/main-shared-nine-v3-20260917')
new = Path('/root/autodl-tmp/TS3/outputs/main-shared-nine-v3-t005-20260917')
old_config = json.loads((old/'protocol.json').read_text())
config = json.loads((new/'protocol.json').read_text())
validate_v3_config(config)
for key in ['students','datasets','arms','seeds','model_paths','source_load','reference_root',
            'inputs','context_length','prediction_length','crb','method_protocols','pool_equivalents','pool_seed']:
    assert config[key] == old_config[key], key
for student in config['students']:
    for key in ['learning_rate','epochs','batch_size','optimizer','weight_decay','schedule_epochs']:
        assert config['settings'][student][key] == old_config['settings'][student][key]

provenance = dict(source_run=str(old), target_run=str(new), temperature=.05,
    unchanged_arms=['orig','rand','tsds'], changed_arms=list(V3_TEMPERATURE_ARMS),
    reused_cells=[], reused_selections=[], retempered_selections=[], reused_method_caches=[])
for student in config['students']:
    source, dest = old/'students'/student, new/'students'/student
    dest.mkdir(parents=True, exist_ok=True)
    identity = json.loads((source/'model.json').read_text())
    atomic_json(dest/'model.json', identity)
    for p in (source/'method_cache').glob('*'):
        if p.is_file():
            target = dest/'method_cache'/p.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.link(p,target)
            provenance['reused_method_caches'].append(str(p))
    for task in config['datasets']:
        for arm in V3_MATRIX:
            source_selection = source/'selections'/f'{task}__{arm}.npz'
            target_selection = dest/'selections'/source_selection.name
            if source_selection.exists():
                output = load_selection(source_selection)
                meta = output.selected.metadata
                assert meta['pool_manifest']['total_equivalents'] == config['pool_equivalents']
                assert meta.get('model', identity) == identity
                target_selection.parent.mkdir(parents=True, exist_ok=True)
                if arm in V3_TEMPERATURE_ARMS:
                    started = time.perf_counter()
                    assert meta['temperature'] == .25
                    if arm == 'lp-mirror':
                        scores = output.selected.native_scores
                        assert meta['probability_mapping'] == 'value'
                    elif arm == 'less-main':
                        assert meta['reference_aggregation'] == 'mean_reference_cosines'
                        assert meta['source_weighting'] == 'equal_trajectory'
                        scores = meta['native_scores']
                    else:
                        assert meta['reference_audit']['aggregation'] == 'mean_reference_trajectory_cosines'
                        assert meta['source_weighting'] == ('equal_block' if arm == 'rags-main' else 'equal_trajectory')
                        scores = meta['reference_audit']['source_scores']
                    scores = np.asarray(scores)
                    np.testing.assert_allclose(value_mass(scores, temperature=.25, gating=False),
                                               output.selected.probabilities, rtol=1e-12, atol=1e-15)
                    output.selected.probabilities = value_mass(scores, temperature=.05, gating=False)
                    meta.update(temperature=.05, temperature_reuse=dict(source_artifact=str(source_selection),
                        source_temperature=.25, reuse='scores_and_units_only; softmax recomputed'))
                    if 'reference_audit' in meta:
                        meta['reference_audit']['temperature'] = .05
                    if 'reference_audit' in output.diagnostics:
                        output.diagnostics['reference_audit']['temperature'] = .05
                    output.selected._prepare_sampling()
                    output.diagnostics['resampling'] = output.selected.coverage()
                    output.diagnostics['temperature_reuse'] = meta['temperature_reuse']
                    output.cost.gradient_computations = output.cost.feature_computations = output.cost.proxy_training_steps = 0
                    output.cost.wall_seconds = time.perf_counter()-started
                    output.cost.notes['reused_scoring_from'] = str(source_selection)
                    save_selection(target_selection,output)
                    provenance['retempered_selections'].append(str(target_selection))
                else:
                    if not target_selection.exists():
                        os.link(source_selection,target_selection)
                    provenance['reused_selections'].append(str(target_selection))
                del output
            cell = source/'cells'/f'{task}__{arm}__seed0.json'
            if cell.exists() and arm not in V3_TEMPERATURE_ARMS:
                data = json.loads(cell.read_text())
                assert (data['experiment_version'],data['student'],data['dataset'],data['arm'],data['seed']) == ('v3',student,task,arm,0)
                data['reuse_provenance'] = dict(source_cell=str(cell), source_run=str(old),
                    reason='Temperature-independent arm; identical model/data/seed/training/evaluation protocol')
                if 'selection_artifact' in data:
                    data['selection_artifact'] = str(target_selection)
                atomic_json(dest/'cells'/cell.name,data)
                provenance['reused_cells'].append(str(cell))
    assert not any((dest/'cells').glob('*__less-main__*'))
    assert not any((dest/'cells').glob('*__rags-main__*'))
    assert not any((dest/'cells').glob('*__rags-traj__*'))
    assert not any((dest/'cells').glob('*__lp-mirror__*'))

atomic_json(new/'temperature_change.json',provenance)
atomic_json(new/'execution_control.json',dict(mode='run',training_enabled=True,evaluation_enabled=True,
    selection_enabled=True,assets_enabled=True,awaiting_user_authorization=False,
    authorization='User requested all temperature methods at 0.05, including lp-mirror'))
ready = json.loads((new/'readiness.json').read_text()); ready.update(authorized=True,status='ready_authorized')
atomic_json(new/'readiness.json',ready)
current_config = Path('configs/rags/main_v3.json')
archive_config = Path('configs/rags/main_v3_T025.json')
if not archive_config.exists():
    atomic_json(archive_config,json.loads(current_config.read_text()))
atomic_json(current_config,config)
snapshot = Path('docs/archive/MAIN-v3-t025-snapshot-20260917.md')
if not snapshot.exists():
    snapshot.write_text(Path('docs/MAIN-v3.md').read_text())
text = snapshot.read_text()
for arm in V3_MATRIX:
    text = re.sub(r'(?<![\w-])'+re.escape(arm)+r'(?![\w-])',arm+'-T025',text)
text = text.replace('(archive/','(').replace('(../temp/','(../../temp/')
snapshot.write_text(text)
print('T005 ready:',{k:len(provenance[k]) for k in ['reused_cells','retempered_selections','reused_selections']})
