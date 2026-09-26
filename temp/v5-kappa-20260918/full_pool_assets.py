"""Prepare frozen reference method-union assets, without evaluation or training."""
import argparse
from dataclasses import asdict
import fcntl
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from assets.gradient.cache import GradientCache, FragmentGradients, index_request, readable
from assets.gradient.variants import aggregate_retained, aggregate_anchors, trajectory_counts
from baselines.trajectory import trajectory_positions
from rags.crb import CRBConfig, CRBEngineMulti
from rags.execution import SelectionExecution, selection_execution
from .arms import WindowIndex
from .development_inputs import load_fragments
from .full_pool_models import build_model
from .resampling import REVISION
from .students import STUDENTS, atomic_json


def _save_result(path, result):
    """Persist one trajectory at a time, avoiding whole-task vector copies."""
    blocks = result.blocks
    keys = sorted(result.anchors)
    aggregate = aggregate_retained(result)
    anchor_aggregate = aggregate_anchors(result)
    width = len(result.anchors[keys[0]]) if keys else 0
    data = dict(fragment_id=result.fragment_id, valid_starts=result.valid_starts,
        anchor_indices=np.asarray(keys, dtype=np.int64),
        anchors=np.stack([result.anchors[k] for k in keys]) if keys else np.empty((0,width)),
        block_left=np.asarray([b.left_index for b in blocks], dtype=np.int64),
        block_right=np.asarray([b.right_index for b in blocks], dtype=np.int64),
        block_offsets=np.cumsum([0]+[len(b.start_positions) for b in blocks]),
        block_starts=np.concatenate([b.start_positions for b in blocks]) if blocks else np.empty(0,dtype=np.int64),
        block_means=np.stack([b.mean_gradient for b in blocks]) if blocks else np.empty((0,width)),
        block_directions=np.stack([b.direction for b in blocks]) if blocks else np.empty((0,width)),
        trajectory_mean=aggregate.mean_gradient if aggregate is not None else np.zeros(width),
        trajectory_anchor_mean=anchor_aggregate.mean_gradient if anchor_aggregate is not None else np.zeros(width),
        audit=json.dumps(dict(counts=trajectory_counts(result), epsilon=result.epsilon_audit,
                             **({"split":result.split_audit} if result.split_audit is not None else {}))))
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **data)
    temporary.replace(path)


def run_assets(run_dir, student, *, datasets=None, reference_batch=32, model_bundle=None,
               gradient_workers=None):
    if reference_batch < 1:
        raise ValueError('reference_batch must be positive')
    run = Path(run_dir)
    protocol = json.loads((run/'protocol.json').read_text())
    if protocol['resampling_revision'] != REVISION or student not in protocol['students']:
        raise ValueError('Wrong protocol/student')
    tasks = protocol['datasets'] if datasets is None else datasets
    if len(set(tasks)) != len(tasks) or any(t not in protocol['datasets'] for t in tasks):
        raise ValueError('Tasks must be a unique subset of the run protocol')
    out = run/'students'/student
    out.mkdir(parents=True, exist_ok=True)
    with (out/'worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        engine, probe, identity = model_bundle or build_model(student,
            model_path=protocol.get('model_paths',{}).get(student),
            device=protocol.get('device','cuda'), context_length=protocol['context_length'],
            prediction_length=protocol['prediction_length'])
        model_path = out/'model.json'
        if model_path.exists() and json.loads(model_path.read_text()) != identity:
            raise ValueError('Loaded model identity changed')
        atomic_json(model_path, identity)
        execution = dict(protocol.get('selection_execution',{}))
        execution.update(protocol.get('student_selection_execution',{}).get(student,{}))
        if (run/'execution.json').exists():
            execution.update(json.loads((run/'execution.json').read_text()).get(
                'student_selection_execution',{}).get(student,{}))
        if gradient_workers is not None:
            execution['gradient_workers'] = gradient_workers
        probe.execution = SelectionExecution(**execution)
        c,h = protocol['context_length'],protocol['prediction_length']
        thresholds = {}
        if protocol['crb'].get('epsilon_policy') not in ('trajectory_p10', 'none'):
            from rags.epsilon_protocol import load_p10_calibration
            calibration = load_p10_calibration(protocol['settings'][student]['epsilon_calibration'], identity)
            thresholds = dict(mean_epsilon=calibration['epsilon'],endpoint_epsilon=calibration['epsilon'])
        crb = CRBConfig(context_length=c,prediction_length=h,**thresholds,**protocol['crb'])
        inputs = Path(protocol['inputs'])
        manifest = json.loads((inputs/'pool_manifest.json').read_text())
        state = out/'assets_state.json'
        cache = GradientCache(run,probe,model_identity=identity,data_fields=dict(pool=manifest),
                              context_length=c,prediction_length=h)
        try:
            for task in tasks:
                started = time.perf_counter()
                root = out/'reference_assets'/task
                root.mkdir(parents=True,exist_ok=True)
                refs = load_fragments(inputs/f'{task}__reference.npz')
                fields = dict(schema='reference-method-union-v1',dataset=task,model=identity,
                    crb=asdict(crb),gradient_cache=str(cache.path),
                    fragments=[dict(fragment_id=int(f.fragment_id),entity_id=str(f.entity_id),
                                    length=int(f.length)) for f in refs])
                fields_path = root/'protocol.json'
                if fields_path.exists() and json.loads(fields_path.read_text()) != fields:
                    raise ValueError('Reference asset fields changed; use a new asset directory')
                atomic_json(fields_path,fields)
                # Match prepare_gradients' immutable reference namespace marker.
                marker = cache.path/('reference-'+readable(task)+'.json')
                marker_fields = dict(task=task,fragments=fields['fragments'])
                if marker.exists() and json.loads(marker.read_text()) != marker_fields:
                    raise ValueError('Reference cache fields changed')
                atomic_json(marker,marker_fields)
                files = [root/f'trajectory-{f.fragment_id}.npz' for f in refs]
                completed = sum(p.exists() for p in files)
                atomic_json(state,dict(mode='assets_only',stage='reference_gradients',dataset=task,
                    status='running',completed_trajectories=completed,total_trajectories=len(refs)))
                before = cache.computed_windows
                namespace = 'reference:'+task
                with selection_execution(probe):
                    for begin in range(0,len(refs),reference_batch):
                        group = [f for f in refs[begin:begin+reference_batch]
                                 if not (root/f'trajectory-{f.fragment_id}.npz').exists()]
                        if not group:
                            continue
                        ref_index = WindowIndex(group,context_length=c,prediction_length=h)
                        cache.prepare([index_request(namespace,ref_index,trajectory_positions(ref_index,crb))])
                        results = CRBEngineMulti(crb,FragmentGradients(cache,namespace)).process_fragments(group)
                        for result in results:
                            _save_result(root/f'trajectory-{result.fragment_id}.npz',result)
                        completed += len(group)
                        atomic_json(state,dict(mode='assets_only',stage='reference_gradients',dataset=task,
                            status='running',completed_trajectories=completed,total_trajectories=len(refs),
                            task_new_windows=cache.computed_windows-before,
                            computed_gradient_seconds=cache.computed_gradient_seconds,
                            elapsed_seconds=time.perf_counter()-started))
                        print(student,task,'REFERENCE_BATCH_COMPLETE',completed,'/',len(refs),
                              'new_windows=',cache.computed_windows-before,flush=True)
                        del results,ref_index,group,result
                        gc.collect()
                report = dict(fields,complete=True,trajectories=len(refs),
                    selection_execution=asdict(probe.execution),
                    new_windows=cache.computed_windows-before,elapsed_seconds=time.perf_counter()-started,
                    artifacts=[p.name for p in files])
                atomic_json(root/'complete.json',report)
                print(student,task,'REFERENCE_ASSETS_COMPLETE',flush=True)
                del refs,files
                gc.collect()
                if torch.device(engine.device).type == 'cuda':
                    torch.cuda.empty_cache()
            atomic_json(state,dict(mode='assets_only',status='complete',student=student,datasets=tasks))
        except BaseException as error:
            atomic_json(state,dict(mode='assets_only',status='failed',student=student,error=repr(error)))
            raise
        finally:
            cache.close()
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--student',choices=STUDENTS,required=True)
    parser.add_argument('--tasks',help='Comma-separated protocol tasks')
    parser.add_argument('--reference-batch',type=int,default=32)
    parser.add_argument('--gradient-workers',type=int,default=None,
                        help='Override independent native-window worker count, not the gradient definition')
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    run_assets(args.run,args.student,datasets=args.tasks.split(',') if args.tasks else None,
               reference_batch=args.reference_batch,gradient_workers=args.gradient_workers)


if __name__ == '__main__':
    main()
