"""V7 phased worker. No old seed-round shortcuts; no implicit phase transitions."""
import argparse
from dataclasses import asdict
import fcntl
import gc
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from assets.gradient.cache import atomic_json, index_request
from .arms import WindowIndex, ArmOutput, SelectionCost
from .development_inputs import load_fragments
from .v7_protocol import VERSION, MODELS, NEW_ARMS, WINDOWS, crb_for, validate, asset_protocol
from . import v7_io
from .v7_amendment import compatible_identity, freeze_training

def read(path):
    return json.loads(Path(path).read_text())

def native_cache_directory(out, task, arm, tasks):
    """Reuse TSDS source features within one model/seed, never task references.

    trajectory_features still validates the complete existing field manifest.
    Keep legacy locations to avoid moving or duplicating reusable resources.
    """
    root = Path(out)/'native_assets'
    if arm == 'tsds':
        for candidate in tasks:
            directory = root/candidate
            if (directory/'tsds-trajectory-features.npy').exists():
                return directory
    return root/task

def freeze(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise ValueError('Immutable V7 identity differs: '+str(path))
    else:
        atomic_json(path, value)

def cell_identity(config, student, seed, task, manifest):
    return dict(version=VERSION, student=student, seed=seed, task=task,
        actual_pool=manifest, scientific_protocol=asset_protocol(config))

def folder(run, student, seed):
    return Path(run)/'seeds'/f'seed{seed}'/student

def marker(run, student, seed, task, phase):
    return folder(run, student, seed)/phase/f'{task}.json'

def gate(config, run, phase, *, student=None):
    if student is not None and student not in MODELS:
        raise ValueError('Unknown model for phase audit')
    students = [student] if student is not None else MODELS
    missing = []
    for seed in config['seeds']:
        manifest = read(Path(run)/'inputs'/f'seed{seed}_manifest.json')
        for student in students:
            for task in config['datasets']:
                path = marker(run, student, seed, task, phase)
                if not path.exists():
                    missing.append(str(path))
                    continue
                row = read(path)
                if row.get('identity') != cell_identity(config, student, seed, task, manifest) or not row.get('complete'):
                    raise ValueError('V7 phase marker identity/integrity differs: '+str(path))
                if phase=='assets':
                    audit=row['audit']
                    root=folder(run,student,seed)
                    required=[root/'assets'/f'{task}__requirements.npz',
                        root/'native_assets'/f'{task}__lp-mirror.pt',
                        Path(audit['path'])/'window_index.npz']
                    if any(not p.is_file() for p in required):
                        raise RuntimeError('Asset receipt exists but a required artifact is missing')
                    if read(Path(audit['path'])/'cache_manifest.json') != audit['gradient_manifest']:
                        raise ValueError('Completed gradient manifest changed')
                elif phase=='select':
                    if len(row.get('selections',[])) != len(config['arms']) or row.get('selection_new_bp') != 0:
                        raise RuntimeError('Incomplete or nonzero-BP selection receipt')
                    for selection in row['selections']:
                        if selection['arm']!='orig' and not Path(selection['path']).is_file():
                            raise RuntimeError('Missing completed selection artifact')
    if missing:
        raise RuntimeError(f'V7 {phase} barrier: {len(missing)} incomplete seed/model/tasks; first={missing[0]}')

def seed_pool(run, seed, feasible):
    manifest = read(Path(run)/'inputs'/f'seed{seed}_manifest.json')
    by_id = {f.fragment_id:f for f in feasible}
    pool = [by_id[fid] for fid in manifest['fragment_ids']]
    if sum(f.length//576 for f in pool) != 16284:
        raise ValueError('Training pool exposure must remain 16284')
    if [f.length for f in pool] != manifest['lengths']:
        raise ValueError('Training pool physical identity differs')
    return pool, manifest

def artifact_identity(config, student, seed, task, manifest, model):
    identity = dict(cell_identity(config,student,seed,task,manifest), model=model)
    if 'shape_amendment' in config:
        identity['shape_amendment'] = config['shape_amendment']
    return identity

def requirements(path, assets, pool, task, *, check=False):
    cache = assets.cache
    if check:
        with np.load(path, allow_pickle=False) as data:
            for side, fid, start in data['keys']:
                key = ('source' if side == 0 else 'reference:'+task, int(fid), int(start))
                if key not in cache.rows:
                    raise RuntimeError('Prepared V7 union lost a required gradient row: '+str(key))
        return
    fids = {f.fragment_id for f in pool}
    keys = [(0 if ds=='source' else 1, fid, start) for ds,fid,start in cache.rows
        if (ds=='source' and fid in fids) or ds=='reference:'+task]
    v7_io.atomic_npz(path, keys=np.asarray(keys,dtype=np.int64).reshape(-1,3))

def new_scores(assets, index, ref_index, audit, wanted=NEW_ARMS):
    from .v7_scores import reference_mean, score_trajectory
    plans = audit['plans']
    def trajectories(side, current, dataset):
        for row in plans[side]:
            pos = np.asarray(row['positions'], dtype=np.int64)
            gradients = assets.cache.get(index_request(dataset, current, pos+row['global_offset']))
            yield row, gradients
    target = (reference_mean((r['legal_windows'],r['positions'],g)
        for r,g in trajectories('reference', ref_index, assets.reference_dataset))
        if any(a in wanted for a in ('grad-dot','grad-cos')) else None)
    scores = {a:np.empty(len(index), dtype=np.float64) for a in wanted}
    for row, gradients in trajectories('source', index, 'source'):
        out = score_trajectory(row['legal_windows'],row['positions'],gradients,target)
        lo = row['global_offset']; hi = lo+row['legal_windows']
        for arm in wanted:
            scores[arm][lo:hi] = out[arm]
    return scores

def model_phase(config, run, student, phase, *, seed_filter=None, task_filter=None):
    from .full_pool_models import build_model
    from .full_pool import select_full_pool
    from .v7_assets import prepare
    from .v7_cache import verified_donors
    from .v7_scores import deploy
    from rags.execution import SelectionExecution, selection_execution
    import torch
    run = Path(run)
    if task_filter is not None and (phase != 'train' or task_filter not in config['datasets']):
        raise ValueError('Task filter must name an active training task')
    if phase=='assets':
        admission=read(run/'input_preflight.json')
        if not admission.get('complete') or not admission.get('disk_ready'):
            raise RuntimeError('V7 input/capacity admission incomplete; no gradient work started')
    if phase == 'select':
        # Each model owns an independent gradient namespace and may score as
        # soon as its own sparse assets are complete.  Other models can still
        # prepare assets without forcing this GPU to wait at a global barrier.
        gate(config, run, 'assets', student=student)
    if phase == 'train':
        gate(config, run, 'assets')
    if phase == 'train':
        gate(config, run, 'select', student=student)
        if seed_filter is None:
            raise ValueError('Training needs explicit seed; seeds advance within each model')
        for prior in range(seed_filter):
            for other in [student]:
                for task in config['datasets']:
                    for arm in config['arms']:
                        path = folder(run,other,prior)/'cells'/f'{task}__{arm}.json'
                        prior_manifest = read(run/'inputs'/f'seed{prior}_manifest.json')
                        expected = dict(artifact_identity(config,student,prior,task,prior_manifest,
                            read(run/'models'/f'{student}.json')),training_protocol=config['training_protocol'])
                        if (not path.exists() or read(path).get('version') != VERSION or
                                not compatible_identity(read(path).get('identity'),expected,config)):
                            raise RuntimeError('Prior seed barrier is incomplete: '+str(path))
    c,h = WINDOWS[student]
    engine, probe, identity = build_model(student, model_path=config['model_paths'].get(student),
                                         context_length=c, prediction_length=h)
    if identity['projection_dimension'] != 4096 or identity['projection_seed'] != 42:
        raise ValueError('Frozen V7 projection identity differs')
    execution = dict(config['selection_execution'])
    override = run/'execution.json'
    if override.exists():
        execution.update(read(override).get(student, {}))
    probe.execution = SelectionExecution(**execution)
    freeze(run/'models'/f'{student}.json', identity)
    feasible = load_fragments(run/'inputs'/'feasible.npz')
    feasible_manifest = read(run/'inputs'/'feasible_manifest.json')
    refs = ({t:load_fragments(run/'inputs'/student/f'{t}__reference.npz') for t in config['datasets']}
            if phase != 'train' else {})
    shared = read(run/'inputs'/student/'gradient_data_identity.json') if phase != 'train' else None
    donors = verified_donors(config,student,feasible,feasible_manifest,refs) if phase=='assets' else []
    for seed in (config['seeds'] if seed_filter is None else [seed_filter]):
        pool, manifest = seed_pool(run,seed,feasible)
        index = WindowIndex(pool,context_length=c,prediction_length=h)
        out = folder(run,student,seed)
        out.mkdir(parents=True,exist_ok=True)
        for task in config['datasets']:
            if task_filter is not None and task != task_filter:
                continue
            key = cell_identity(config,student,seed,task,manifest)
            if phase == 'train':
                key = dict(key,training_protocol=config.get('training_protocol'))
            artifact = artifact_identity(config,student,seed,task,manifest,identity)
            complete = marker(run,student,seed,task,phase)
            complete.parent.mkdir(parents=True,exist_ok=True)
            if complete.exists():
                row = read(complete)
                compatible = (phase=='train' and compatible_identity(row.get('identity'),key,config))
                if (row.get('identity') != key and not compatible) or not row.get('complete'):
                    raise ValueError('Existing phase completion differs')
                if phase != 'assets' and not (phase=='train' and row.get('identity') != key):
                    continue
                # Verify a completed task without allowing missing rows to cause BP.
                readonly = True
            else:
                readonly = phase != 'assets'
            if phase == 'train':
                from .v7_train import train_task
                from .v7_train_leases import train_once
                def execute():
                    state = out/(f'state__{task}.json' if task_filter else 'state.json')
                    atomic_json(state,dict(phase=phase,task=task,seed=seed,status='running',pid=os.getpid()))
                    train_task(config,run,student,seed,task,engine,index,manifest,identity)
                train_once(run,student,seed,task,complete,key,
                    lambda actual, expected: compatible_identity(actual,expected,config),
                    execute,atomic_json)
                continue
            atomic_json(out/'state.json', dict(phase=phase,task=task,seed=seed,status='running'))
            with selection_execution(probe):
                assets, ref_index, audit = prepare(probe=probe,run_dir=out,pool=pool,index=index,
                    refs=refs[task],manifest=manifest,identity=identity,crb=crb_for(student),task=task,
                    donors=donors if not readonly else (),readonly=readonly,shared_data=shared)
            requirement_path = out/'assets'/f'{task}__requirements.npz'
            if readonly:
                requirements(requirement_path,assets,pool,task,check=True)
            lp_path = out/'native_assets'/f'{task}__lp-mirror.pt'
            if phase == 'assets':
                if not lp_path.exists():
                    if readonly:
                        raise RuntimeError('Completed asset marker lost LP-Mirror resource; explicit recovery required')
                    # LP's own proxy backward is charged here, not hidden in zero-BP selection.
                    from baselines.lp_mirror.runner import lp_mirror_select
                    with selection_execution(probe):
                        lp = lp_mirror_select(probe,reference_fragments=refs[task],
                            index=index,pool_manifest=manifest,context_length=c,prediction_length=h,
                            seed=seed,prepare_only=True,cache_dir=out/'native_assets'/'lp-proxy')
                    v7_io.atomic_torch(lp_path,dict(identity=artifact,prepared_proxy=lp))
                else:
                    saved=torch.load(lp_path,map_location='cpu',weights_only=True)
                    if saved['identity']!=artifact:
                        raise ValueError('LP proxy asset identity differs')
                    lp=saved['prepared_proxy']
                audit['native_proxy_cost']=dict(source_proxy_steps=lp['proxy_steps'],
                    reference_adaptation_steps=lp['adapt_steps'],selection_backward_steps=0)
                del lp
                if not complete.exists():
                    requirements(requirement_path,assets,pool,task)
                    atomic_json(complete,dict(identity=key,complete=True,audit=audit))
            else:
                before = assets.cache.computed_windows
                score_arrays = None
                selections = []
                for arm in config['arms']:
                    if arm == 'orig':
                        selections.append(dict(arm=arm,kind='pretrained_no_selection',selection_new_bp=0))
                        continue
                    path = out/'selections'/f'{task}__{arm}.npz'
                    started = time.perf_counter()
                    if path.exists():
                        output = v7_io.load(path,index,manifest,artifact)
                    elif arm in NEW_ARMS:
                        if score_arrays is None:
                            wanted=[a for a in NEW_ARMS if not (out/'selections'/f'{task}__{a}.npz').exists()]
                            score_arrays = new_scores(assets,index,ref_index,audit,wanted=wanted)
                        selected = deploy(index,score_arrays[arm],arm=arm,manifest=manifest,
                            identity=dict(model=identity,gradient_asset_identity=assets.cache.manifest,
                                seed=seed,task=task,actual_bp=audit['plans']['source_audit'],
                                reference_bp=(audit['plans']['reference_audit'] if arm!='grand-static' else None)))
                        output = ArmOutput(arm,selected,[],SelectionCost(gradient_computations=0,
                            notes=dict(required_source=audit['plans']['source_audit']['required_bp'],
                                score_execution_scope='shared_missing_uniform_arms',
                                shared_score_arms=list(score_arrays),
                                required_reference=(audit['plans']['reference_audit']['required_bp']
                                                    if arm!='grand-static' else 0))),{})
                    elif arm == 'lp-mirror':
                        from baselines.lp_mirror.runner import lp_mirror_select
                        saved=torch.load(lp_path,map_location='cpu',weights_only=True)
                        if saved['identity']!=artifact:
                            raise ValueError('LP proxy asset identity differs')
                        output = lp_mirror_select(probe,reference_fragments=refs[task],index=index,
                            pool_manifest=manifest,context_length=c,prediction_length=h,seed=seed,
                            prepared_proxy=saved['prepared_proxy'],mapping='value',
                            resampling=dict(model=identity,task=task,temperature=.05,
                                crb=asdict(crb_for(student)),additional_bp_allowed=False))
                        output.name=arm
                        output.cost.notes.update(selection_proxy_backward_steps=0,
                            preparation_proxy_training_steps=output.cost.proxy_training_steps)
                    else:
                        with selection_execution(probe):
                            output = select_full_pool(arm,probe,reference_fragments=refs[task],pool=pool,
                                index=index,pool_manifest=manifest,model_identity=identity,
                                crb_config=crb_for(student),gradient_assets=assets,seed=seed,
                                temperature=config['settings'][student]['temperature'].get(arm,1.),
                                cache_dir=native_cache_directory(out,task,arm,config['datasets']))
                    if assets.cache.computed_windows != before:
                        raise RuntimeError('Selection performed additional pretrained BP')
                    coverage = v7_io.validate_support(output.selected, arm)
                    if output.selected.pool_n != 16284 or not set(np.unique(output.selected.window_ids[:,0])) <= set(manifest['fragment_ids']):
                        raise ValueError('Pool-external window exposure or wrong pool_n')
                    if not path.exists():
                        output.cost.wall_seconds = time.perf_counter()-started
                        v7_io.save(path,output,artifact)
                    selections.append(dict(arm=arm,path=str(path),selection_new_bp=0,
                        elapsed_seconds=time.perf_counter()-started,cost=vars(output.cost),coverage=coverage))
                    del output
                atomic_json(complete,dict(identity=key,complete=True,selections=selections,
                    reconstruction_new_bp=audit['new_bp'],selection_new_bp=0))
            del assets, ref_index
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print('V7_PHASE_COMPLETE',phase,student,seed,task,flush=True)
    return 0

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/rags/main_v7_curve4.json')
    parser.add_argument('--run')
    parser.add_argument('--phase',choices=['preflight','inputs','assets','select','train','audit-assets','audit-select'],required=True)
    parser.add_argument('--student',choices=MODELS)
    parser.add_argument('--seed',type=int,choices=[0,1,2])
    parser.add_argument('--parallel-train',action='store_true',
                        help='Allow independent tasks; keeps per-task exclusive leases')
    parser.add_argument('--train-task',help='Run one active task; requires --parallel-train')
    args = parser.parse_args()
    config = read(args.config); validate(config)
    if args.parallel_train and args.phase != 'train':
        parser.error('--parallel-train is training-only')
    if args.train_task and (not args.parallel_train or args.train_task not in config['datasets']):
        parser.error('--train-task requires --parallel-train and an active dataset')
    run = Path(args.run or config['run_root'])
    limits_path = run/'execution_limits.json'
    if args.phase == 'train' and limits_path.exists():
        allowed = read(limits_path)['allowed_training_seeds']
        if args.seed not in allowed:
            parser.error(f'Training seed {args.seed} disabled by user execution limits: {allowed}')
    run.mkdir(parents=True,exist_ok=True)
    freeze(run/'protocol.json',asset_protocol(config))
    if 'training_protocol' in config:
        freeze_training(run,config)
    if args.phase == 'preflight':
        from .v7_inputs import preflight
        audit = preflight(config)
        atomic_json(run/'input_preflight.json',audit)
        print(json.dumps(audit,indent=2))
        return int(bool(audit['missing_shards']))
    if args.phase == 'inputs':
        from .v7_inputs import prepare
        prepare(config,run); return 0
    if args.phase.startswith('audit-'):
        gate(config,run,'assets' if args.phase=='audit-assets' else 'select',student=args.student); return 0
    if args.student is None:
        parser.error('A worker requires --student; runtime supervisor owns GPU admission')
    from .v7_train_leases import model_lease
    with model_lease(run,args.student,args.phase,parallel_train=args.parallel_train):
        try:
            return model_phase(config,run,args.student,args.phase,seed_filter=args.seed,
                               task_filter=args.train_task)
        except BaseException as exc:
            atomic_json(run/f'{args.student}__failure.json',dict(phase=args.phase,seed=args.seed,
                error=repr(exc),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
            raise

if __name__ == '__main__':
    raise SystemExit(main())
