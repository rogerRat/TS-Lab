"""All models: half-epoch evaluations through four epochs, no checkpoint selection."""
from pathlib import Path
import json
import time
import numpy as np
from assets.gradient.cache import atomic_json
from .train_eval import TrainingRecipe, run_arm_training, evaluate
from .v7_protocol import VERSION, CURVE_PROTOCOL, REDUCED_PROTOCOL
from .v7_amendment import compatible_identity, write_report_views
from . import v7_io

def checkpoint_curve(result, original, max_epochs=4):
    records = result['checkpoints']
    expected = [i/2 for i in range(1,2*max_epochs+1)]
    if [r['epoch'] for r in records] != expected:
        raise ValueError('All required half-epoch checkpoints must be evaluated')
    if any(not np.isfinite(r['metrics']['mase']) for r in records):
        raise ValueError('Nonfinite checkpoint evaluation')
    return [dict(epoch=0.,step=0,metrics={k:original[k] for k in ('mase','mae','mse') if k in original})] + [
        dict(epoch=r['epoch'],step=r['step'],metrics=r['metrics']) for r in records]

def train_task(config,run,student,seed,task,engine,index,manifest,model_identity, *, initialize_only=False):
    from .v7_run import folder,artifact_identity
    from .checkpoint_retention import cleanup_completed_weights
    protocol = config.get('training_protocol', {})
    if protocol.get('version') not in (CURVE_PROTOCOL, REDUCED_PROTOCOL):
        raise ValueError('Use the approved four-epoch curve configuration for V7 training')
    root = folder(run,student,seed)
    root.joinpath('cells').mkdir(parents=True,exist_ok=True)
    selection_identity = artifact_identity(config,student,seed,task,manifest,model_identity)
    identity = dict(selection_identity,training_protocol=protocol)
    with np.load(Path(run)/'inputs'/student/f'{task}__exam.npz',allow_pickle=False) as data:
        exam = dict(test_windows=data['windows'],test_entities=data['entities'].tolist(),
            test_scales=data['scales'],context_length=index.context_length)
    base = {k:v.detach().cpu().clone() for k,v in engine.model.state_dict().items()}
    settings = config['settings'][student]
    engine.allow_v7_epoch_prefix_resume = (protocol.get('version') == REDUCED_PROTOCOL
                                          and student == 'chronos-base')
    recipe = TrainingRecipe.from_settings(dict(settings,
        learning_rates=[settings['learning_rate']],epochs=[settings['epochs']]))
    original_path = root/'cells'/f'{task}__orig.json'
    if original_path.exists():
        original = json.loads(original_path.read_text())
        if not compatible_identity(original.get('identity'), identity, config):
            raise ValueError('Orig identity changed')
    else:
        shared_original=Path(run)/'original'/student/f'{task}.json'
        shared_identity=dict(version=VERSION,model=model_identity,
            exam_manifest=json.loads((Path(run)/'inputs'/student/f'{task}__manifest.json').read_text()))
        if shared_original.exists():
            shared=json.loads(shared_original.read_text())
            if shared.get('identity') != shared_identity:
                raise ValueError('Shared V7 original identity differs')
            metrics=shared['metrics']
        else:
            metrics=evaluate(engine,batch_size=32,**exam)
            shared_original.parent.mkdir(parents=True,exist_ok=True)
            atomic_json(shared_original,dict(identity=shared_identity,metrics=metrics))
        original = dict(metrics,identity=identity,version=VERSION,
            student=student,seed=seed,task=task,arm='orig',trained=False)
        atomic_json(original_path,original)
    if initialize_only:
        return
    for arm in config['arms']:
        if arm=='orig': continue
        cell = root/'cells'/f'{task}__{arm}.json'
        checkpoint = root/'checkpoints'/task/arm
        if cell.exists():
            if not compatible_identity(json.loads(cell.read_text()).get('identity'), identity, config):
                raise ValueError('Completed V7 cell identity differs')
            # Recover a crash between durable cell results and final state cleanup.
            cleanup_completed_weights(cell,checkpoint)
            continue
        selected = v7_io.load(root/'selections'/f'{task}__{arm}.npz',index,manifest,selection_identity).selected
        if selected.pool_n != 16284:
            raise ValueError('Feasible pool must never become the epoch exposure budget')
        started = time.perf_counter()
        # Existing recipe-aware online path saves recoverable state before evaluating,
        # atomically writes metrics, then replaces the preceding resume state.
        result = run_arm_training(engine,source_windows=selected,source_epochs=settings['epochs'],
            batch_size=32,learning_rate=settings['learning_rate'],recipe=recipe,seed=seed,
            base_state=base,checkpoint_dir=checkpoint,online_evaluation=True,**exam)
        result['test_curve'] = checkpoint_curve(result,original,settings['epochs'])
        result['checkpoint_selection'] = 'none'
        result['summary_epoch'] = result['final_epoch']
        result.update(identity=identity,version=VERSION,student=student,seed=seed,task=task,arm=arm,
            actual_pool=manifest,elapsed_seconds=time.perf_counter()-started,
            improvement_vs_orig=100*(1-result['mase']/original['mase']))
        rand_path = root/'cells'/f'{task}__rand.json'
        if rand_path.exists():
            rand = json.loads(rand_path.read_text())
            result['improvement_vs_rand'] = 100*(1-result['mase']/rand['mase'])
        atomic_json(cell,result)
        cleanup_completed_weights(cell,checkpoint)
        print('V7_CELL_COMPLETE',student,seed,task,arm,flush=True)
    if protocol.get('version') == REDUCED_PROTOCOL:
        write_report_views(root,task,config,student,identity)
        return
    rand = json.loads((root/'cells'/f'{task}__rand.json').read_text())
    for arm in config['arms']:
        path = root/'cells'/f'{task}__{arm}.json'
        row = json.loads(path.read_text())
        row['improvement_vs_rand'] = 100*(1-row['mase']/rand['mase'])
        atomic_json(path,row)
