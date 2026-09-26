"""Train only validated per-seed selections; Timer stop point never uses test MASE."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from assets.gradient.cache import atomic_json
from .train_eval import TrainingRecipe, run_arm_training, evaluate
from .v7_protocol import VERSION
from . import v7_io

def choose_checkpoint(records):
    if not records or any(not np.isfinite(r['mase']) for r in records):
        raise ValueError('Every validation checkpoint needs finite MASE')
    return min(records, key=lambda r:(r['mase'],r['epoch']))

def validation_inputs(config, run, task, index, manifest):
    path = Path(config['timer_validation_manifest'])
    if not path.exists():
        raise FileNotFoundError('TimerXL requires the approved, isolated validation manifest: '+str(path))
    description = json.loads(path.read_text())
    if (description.get('protocol') != 'v7-timer-isolated-validation-v1' or
            description.get('context_length') != 480 or description.get('prediction_length') != 96 or
            description.get('selection_rule') != 'min_validation_mase_earliest_epoch_tie'):
        raise ValueError('Timer validation protocol differs')
    spec = description['tasks'][task]
    if not spec.get('native_data_provenance'):
        raise ValueError('Validation native data provenance is required')
    native = json.loads((Path(run)/'inputs'/'timerxl-84m'/f'{task}__manifest.json').read_text())
    excluded = {s['entity'] for s in native['series']}
    excluded.update(str(f.entity_id) for f in index.by_fragment.values())
    data_path = Path(spec['path'])
    with np.load(data_path,allow_pickle=False) as data:
        windows, entities, scales = data['windows'], data['entities'].tolist(), data['scales']
    if (windows.ndim != 2 or windows.shape[1] != 576 or len(windows) == 0 or
            len(entities) != len(windows) or len(scales) != len(windows) or
            not np.isfinite(windows).all() or not np.isfinite(scales).all() or np.any(scales <= 0)):
        raise ValueError('Invalid native validation arrays')
    if set(entities) != set(spec['entities']) or set(entities) & excluded:
        raise ValueError('Validation must be trajectory-disjoint from reference, test and source')
    receipt = dict(manifest=description, actual_pool=manifest, task=task)
    from .v7_run import freeze
    freeze(Path(run)/'validation_receipts'/f'seed{manifest["seed"]}__{task}.json',receipt)
    return dict(test_windows=windows,test_entities=entities,test_scales=scales,context_length=480)

def train_task(config,run,student,seed,task,engine,index,manifest,model_identity):
    from .v7_run import folder,artifact_identity,freeze
    from .checkpoint_retention import cleanup_completed_weights
    root = folder(run,student,seed)
    root.joinpath('cells').mkdir(parents=True,exist_ok=True)
    identity = artifact_identity(config,student,seed,task,manifest,model_identity)
    with np.load(Path(run)/'inputs'/student/f'{task}__exam.npz',allow_pickle=False) as data:
        exam = dict(test_windows=data['windows'],test_entities=data['entities'].tolist(),
            test_scales=data['scales'],context_length=index.context_length)
    validation = validation_inputs(config,run,task,index,manifest) if student=='timerxl-84m' else None
    base = {k:v.detach().cpu().clone() for k,v in engine.model.state_dict().items()}
    settings = config['settings'][student]
    recipe = TrainingRecipe.from_settings(dict(settings,
        learning_rates=[settings['learning_rate']],epochs=[settings['epochs']]))
    original_path = root/'cells'/f'{task}__orig.json'
    if original_path.exists():
        original = json.loads(original_path.read_text())
        if original.get('identity') != identity:
            raise ValueError('Orig identity changed')
    else:
        original = dict(evaluate(engine,batch_size=32,**exam),identity=identity,version=VERSION,
            student=student,seed=seed,task=task,arm='orig',trained=False)
        atomic_json(original_path,original)
    validation_zero = evaluate(engine,batch_size=32,**validation) if validation else None
    for arm in config['arms']:
        if arm=='orig':
            continue
        cell = root/'cells'/f'{task}__{arm}.json'
        checkpoint = root/'checkpoints'/task/arm
        if cell.exists():
            if json.loads(cell.read_text()).get('identity') != identity:
                raise ValueError('Completed V7 cell identity differs')
            continue
        selected = v7_io.load(root/'selections'/f'{task}__{arm}.npz',index,manifest,identity).selected
        if selected.pool_n != 16284:
            raise ValueError('Feasible pool must never become the epoch exposure budget')
        started = time.perf_counter()
        result = run_arm_training(engine,source_windows=selected,source_epochs=settings['epochs'],
            batch_size=32,learning_rate=settings['learning_rate'],recipe=recipe,seed=seed,
            base_state=base,checkpoint_dir=checkpoint,online_evaluation=False,
            **(validation if validation else exam))
        if validation:
            candidates = [dict(epoch=0,step=0,mase=validation_zero['mase'],checkpoint=None)]
            candidates += [dict(epoch=r['epoch'],step=r['step'],mase=r['metrics']['mase'],
                                checkpoint=r['checkpoint']) for r in result['checkpoints']]
            chosen = choose_checkpoint(candidates)
            result['validation_curve'] = candidates
            result['selected_epoch'] = chosen['epoch']
            result['selected_step'] = chosen['step']
            result['checkpoint_policy'] = 'isolated_validation_min_mase_earliest_tie'
            # The stopping decision above is fixed before reading any checkpoint's test score.
            test_curve = [dict(epoch=0,step=0,metrics={k:v for k,v in original.items()
                if k in ('mase','mae','mse')})]
            chosen_metrics = None
            try:
                for row in candidates:
                    state = base if row['checkpoint'] is None else torch.load(
                        row['checkpoint'],map_location=engine.device,weights_only=True)
                    engine.model.load_state_dict(state,strict=True)
                    metrics = evaluate(engine,batch_size=32,**exam)
                    if row['epoch'] == chosen['epoch']:
                        chosen_metrics = metrics
                    if row['epoch']:
                        test_curve.append(dict(epoch=row['epoch'],step=row['step'],metrics=metrics))
                result.update(chosen_metrics)
                result['test_curve'] = test_curve
            finally:
                engine.model.load_state_dict(base,strict=True)
        else:
            result['selected_epoch'] = result['final_epoch']
        result.update(identity=identity,version=VERSION,student=student,seed=seed,task=task,arm=arm,
            actual_pool=manifest,elapsed_seconds=time.perf_counter()-started,
            improvement_vs_orig=100*(1-result['mase']/original['mase']))
        rand_path = root/'cells'/f'{task}__rand.json'
        if rand_path.exists():
            rand = json.loads(rand_path.read_text())
            result['improvement_vs_rand'] = 100*(1-result['mase']/rand['mase'])
        atomic_json(cell,result)
        if config['checkpoint_retention']=='completed_cell_cleanup':
            cleanup_completed_weights(cell,checkpoint)
        print('V7_CELL_COMPLETE',student,seed,task,arm,flush=True)
    # Uniform comparison fields for arms executed before rand; original included.
    rand = json.loads((root/'cells'/f'{task}__rand.json').read_text())
    for arm in config['arms']:
        path = root/'cells'/f'{task}__{arm}.json'
        row = json.loads(path.read_text())
        row['improvement_vs_rand'] = 100*(1-row['mase']/rand['mase'])
        atomic_json(path,row)
