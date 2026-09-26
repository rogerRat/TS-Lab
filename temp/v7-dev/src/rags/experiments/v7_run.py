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
from .v7_protocol import VERSION, MODELS, NEW_ARMS, WINDOWS, crb_for, validate
from . import v7_io

def read(path):
    return json.loads(Path(path).read_text())

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
        actual_pool=manifest, scientific_protocol=config)

def folder(run, student, seed):
    return Path(run)/'seeds'/f'seed{seed}'/student

def marker(run, student, seed, task, phase):
    return folder(run, student, seed)/phase/f'{task}.json'

def gate(config, run, phase):
    missing = []
    for seed in config['seeds']:
        manifest = read(Path(run)/'inputs'/f'seed{seed}_manifest.json')
        for student in MODELS:
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
                        root/'native_assets'/f'{task}__lp-mirror.npz',
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
    return dict(cell_identity(config,student,seed,task,manifest), model=model)

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

def new_scores(assets, index, ref_index, audit):
    from .v7_scores import reference_mean, score_trajectory
    plans = audit['plans']
    def trajectories(side, current, dataset):
        for row in plans[side]:
            pos = np.asarray(row['positions'], dtype=np.int64)
            gradients = assets.cache.get(index_request(dataset, current, pos+row['global_offset']))
            yield row, gradients
    target = reference_mean((r['legal_windows'],r['positions'],g)
        for r,g in trajectories('reference', ref_index, assets.reference_dataset))
    scores = {a:np.empty(len(index), dtype=np.float64) for a in NEW_ARMS}
    for row, gradients in trajectories('source', index, 'source'):
        out = score_trajectory(row['legal_windows'],row['positions'],gradients,target)
        lo = row['global_offset']; hi = lo+row['legal_windows']
        for arm in NEW_ARMS:
            scores[arm][lo:hi] = out[arm]
    return scores

def model_phase(config, run, student, phase, *, seed_filter=None):
    from .full_pool_models import build_model
    from .full_pool import select_full_pool
    from .v7_assets import prepare
    from .v7_cache import verified_donors
    from .v7_scores import deploy
    from rags.execution import SelectionExecution, selection_execution
    import torch
    run = Path(run)
    if phase in ('select','train'):
        gate(config, run, 'assets')
    if phase == 'train':
        gate(config, run, 'select')
        if seed_filter is None:
            raise ValueError('Training needs explicit seed; supervisor enforces model-wide seed barrier')
        for prior in range(seed_filter):
            for other in MODELS:
                for task in config['datasets']:
                    for arm in config['arms']:
                        path = folder(run,other,prior)/'cells'/f'{task}__{arm}.json'
                        if not path.exists() or read(path).get('version') != VERSION:
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
    refs = {t:load_fragments(run/'inputs'/student/f'{t}__reference.npz') for t in config['datasets']}
    shared = read(run/'inputs'/student/'gradient_data_identity.json')
    donors = verified_donors(config,student,feasible,feasible_manifest,refs) if phase=='assets' else []
    for seed in (config['seeds'] if seed_filter is None else [seed_filter]):
        pool, manifest = seed_pool(run,seed,feasible)
        index = WindowIndex(pool,context_length=c,prediction_length=h)
        out = folder(run,student,seed)
        out.mkdir(parents=True,exist_ok=True)
        for task in config['datasets']:
            key = cell_identity(config,student,seed,task,manifest)
            artifact = artifact_identity(config,student,seed,task,manifest,identity)
            complete = marker(run,student,seed,task,phase)
            complete.parent.mkdir(parents=True,exist_ok=True)
            if complete.exists():
                row = read(complete)
                if row.get('identity') != key or not row.get('complete'):
                    raise ValueError('Existing phase completion differs')
                if phase != 'assets':
                    continue
                # Verify a completed task without allowing missing rows to cause BP.
                readonly = True
            else:
                readonly = phase != 'assets'
            atomic_json(out/'state.json', dict(phase=phase,task=task,seed=seed,status='running'))
            if phase == 'train':
                from .v7_train import train_task
                train_task(config,run,student,seed,task,engine,index,manifest,identity)
                atomic_json(complete,dict(identity=key,complete=True))
                continue
            with selection_execution(probe):
                assets, ref_index, audit = prepare(probe=probe,run_dir=out,pool=pool,index=index,
                    refs=refs[task],manifest=manifest,identity=identity,crb=crb_for(student),task=task,
                    donors=donors if not readonly else (),readonly=readonly,shared_data=shared)
            requirement_path = out/'assets'/f'{task}__requirements.npz'
            if readonly:
                requirements(requirement_path,assets,pool,task,check=True)
            lp_path = out/'native_assets'/f'{task}__lp-mirror.npz'
            if phase == 'assets':
                if not lp_path.exists():
                    if readonly:
                        raise RuntimeError('Completed asset marker lost LP-Mirror resource; explicit recovery required')
                    # LP's own proxy backward is charged here, not hidden in zero-BP selection.
                    with selection_execution(probe):
                        lp = select_full_pool('lp-mirror',probe,reference_fragments=refs[task],
                            pool=pool,index=index,pool_manifest=manifest,model_identity=identity,
                            crb_config=crb_for(student),temperature=.05,seed=seed,
                            cache_dir=out/'native_assets'/'lp-proxy')
                    v7_io.save(lp_path,lp,artifact)
                    audit['native_proxy_cost'] = vars(lp.cost)
                    del lp
                else:
                    lp = v7_io.load(lp_path,index,manifest,artifact)
                    audit['native_proxy_cost'] = vars(lp.cost)
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
                            score_arrays = new_scores(assets,index,ref_index,audit)
                        selected = deploy(index,score_arrays[arm],arm=arm,manifest=manifest,
                            identity=dict(model=identity,gradient_asset_identity=assets.cache.manifest,
                                seed=seed,task=task,actual_bp=audit['plans']['source_audit'],
                                reference_bp=(audit['plans']['reference_audit'] if arm!='grand-static' else None)))
                        output = ArmOutput(arm,selected,[],SelectionCost(gradient_computations=0,
                            notes=dict(required_source=audit['plans']['source_audit']['required_bp'],
                                required_reference=(audit['plans']['reference_audit']['required_bp']
                                                    if arm!='grand-static' else 0))),{})
                    elif arm == 'lp-mirror':
                        output = v7_io.load(lp_path,index,manifest,artifact)
                    else:
                        with selection_execution(probe):
                            output = select_full_pool(arm,probe,reference_fragments=refs[task],pool=pool,
                                index=index,pool_manifest=manifest,model_identity=identity,
                                crb_config=crb_for(student),gradient_assets=assets,seed=seed,
                                temperature=config['settings'][student]['temperature'].get(arm,1.),
                                cache_dir=out/'native_assets'/task)
                    if assets.cache.computed_windows != before:
                        raise RuntimeError('Selection performed additional pretrained BP')
                    coverage = output.selected.coverage()
                    if coverage['filtered_legal_windows'] or coverage['zero_mass_legal_windows']:
                        raise ValueError('Selection must cover the whole actual pool with positive support')
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
    parser.add_argument('--config',default='configs/rags/main_v7.json')
    parser.add_argument('--run')
    parser.add_argument('--phase',choices=['preflight','inputs','assets','select','train','audit-assets','audit-select'],required=True)
    parser.add_argument('--student',choices=MODELS)
    parser.add_argument('--seed',type=int,choices=[0,1,2])
    args = parser.parse_args()
    config = read(args.config); validate(config)
    run = Path(args.run or config['run_root'])
    run.mkdir(parents=True,exist_ok=True)
    freeze(run/'protocol.json',config)
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
        gate(config,run,'assets' if args.phase=='audit-assets' else 'select'); return 0
    if args.student is None:
        parser.error('A worker requires --student; runtime supervisor owns GPU admission')
    with (run/f'{args.student}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            return model_phase(config,run,args.student,args.phase,seed_filter=args.seed)
        except BaseException as exc:
            atomic_json(run/f'{args.student}__failure.json',dict(phase=args.phase,seed=args.seed,
                error=repr(exc),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
            raise

if __name__ == '__main__':
    raise SystemExit(main())
