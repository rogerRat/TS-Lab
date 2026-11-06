"""Independent V7 shape-arm supplement; reads frozen assets, writes new cells only."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np

from assets.gradient.cache import atomic_json
from .arms import ArmOutput, SelectionCost, WindowIndex
from .development_inputs import load_fragments
from .full_pool_models import build_model
from .ranking import frozen_rank_protocol, rank_mass
from .resampling import ResampledWindows
from .v7_amendment import report_view
from .v7_assets import prepare
from .v7_protocol import WINDOWS, crb_for, validate
from .v7_run import artifact_identity, folder, freeze, read, seed_pool
from .v7_shape import ARMS, VERSION, feature_cache, protocol, reference_weights, scores_many
from .v7_train import train_task
from . import v7_io

SUPPLEMENT = 'v7-shape-backfill-two-seeds-20260925'
EXTENSIONS = ('ne-china-wind-20260925', 'time_sg_pm25_h-20260925',
              'time_coastal_ts_h-20260925')


def publish_receipt(path, row):
    """Publish a small result marker even on a pristine sidecar run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, row)


def sidecar_config(parent, run):
    validate(parent)
    config = deepcopy(parent)
    config['run_root'] = str(Path(run).resolve())
    config['seeds'] = [0, 1]
    selected = tuple(parent.get('task_extension', {}).get('v7_e_shape_arms', ARMS))
    if not selected or any(arm not in ARMS for arm in selected):
        raise ValueError('Shape sidecar methods differ from approved Shape arms')
    config['arms'] = ['orig', 'rand', *selected]
    config['shape_amendment'] = dict(protocol(), supplement=SUPPLEMENT,
        parent_asset_protocol=parent['asset_protocol']['experiment_version'],
        source='existing_V7_CRB_endpoint_gradients',
        training='original_model_optimizer_epochs_half_epoch_grid',
        result_namespace='shape_backfill_20260925')
    if selected != ARMS:
        config['shape_amendment']['methods'] = list(selected)
        config['shape_amendment']['aggregations'] = [a.removeprefix('rage-shape-') for a in selected]
        config['shape_amendment']['selection_basis'] = 'frozen_complete_ten_task_mean_improvement'
    return config


def create_sidecar(parent_run, parent_config):
    parent_run = Path(parent_run).resolve()
    sidecar = parent_run/'shape_backfill_20260925'
    config = sidecar_config(parent_config, sidecar)
    sidecar.mkdir(parents=True, exist_ok=True)
    for name in ('inputs', 'models'):
        path = sidecar/name
        if path.is_symlink():
            if path.resolve() != (parent_run/name).resolve():
                raise ValueError('Sidecar resource link identity differs: '+str(path))
        elif path.exists():
            raise ValueError('Sidecar immutable resource target is not a link: '+str(path))
        else:
            path.symlink_to(parent_run/name, target_is_directory=True)
    freeze(sidecar/'shape_config.json', config)
    freeze(sidecar/'shape_protocol.json', config['shape_amendment'])
    freeze(sidecar/'training_protocol.json', config['training_protocol'])
    freeze(sidecar/'execution_limits.json', dict(version=SUPPLEMENT,allowed_training_seeds=[0,1]))
    return sidecar, config


def copy_baselines(parent, sidecar, parent_config, config, model, seed, task, manifest, identity):
    """Relabel verified immutable metrics; never retrain rand or re-evaluate orig."""
    parent_root, target = folder(parent,model,seed), folder(sidecar,model,seed)
    target.joinpath('cells').mkdir(parents=True, exist_ok=True)
    old_expected = dict(artifact_identity(parent_config,model,seed,task,manifest,identity),
                        training_protocol=parent_config['training_protocol'])
    new_expected = dict(artifact_identity(config,model,seed,task,manifest,identity),
                        training_protocol=config['training_protocol'])
    for arm in ('orig','rand'):
        src = parent_root/'cells'/f'{task}__{arm}.json'
        dst = target/'cells'/src.name
        source = report_view(read(src),old_expected,parent_config,model)
        expected = dict(source, identity=new_expected, reused_from=str(src),
                        reuse_scope='metrics_and_curve_only_no_eval_no_training')
        freeze(dst, expected)


def select_shape(parent, sidecar, parent_config, config, model, seed, task,
                 probe, identity, index, pool, manifest, main_run):
    root = folder(sidecar,model,seed)
    selected_arms = tuple(config['arms'][2:])
    modes = tuple(arm.removeprefix('rage-shape-') for arm in selected_arms)
    paths = {arm:root/'selections'/f'{task}__{arm}.npz' for arm in selected_arms}
    receipt = root/'shape_select'/f'{task}.json'
    artifact = artifact_identity(config,model,seed,task,manifest,identity)
    score_cache = root/'shape_scores'/f'{task}.npz'
    if receipt.exists():
        row = read(receipt)
        if (row.get('identity') != artifact or not row.get('complete') or
                row.get('selection_new_bp') != 0 or any(not p.is_file() for p in paths.values())):
            raise ValueError('Shape selection receipt or artifacts differ')
        return row
    if any(p.exists() for p in paths.values()) and not score_cache.is_file():
        raise RuntimeError('Shape selection exists without reusable score cache; audit before resume')
    refs = load_fragments(Path(parent)/'inputs'/model/f'{task}__reference.npz')
    shared = read(Path(parent)/'inputs'/model/'gradient_data_identity.json')
    started = time.perf_counter()
    original_asset = read(folder(parent,model,seed)/'assets'/f'{task}.json')['audit']
    original_cache = Path(original_asset['path'])
    if read(original_cache/'cache_manifest.json') != original_asset['gradient_manifest']:
        raise ValueError('Original gradient manifest differs from its asset receipt')
    prior_cache_root = os.environ.get('RAGE_GRADIENT_CACHE_ROOT')
    os.environ['RAGE_GRADIENT_CACHE_ROOT'] = str(original_cache.parent)
    try:
        assets, ref_index, audit = prepare(probe=probe, run_dir=folder(parent,model,seed),
            pool=pool, index=index, refs=refs, manifest=manifest, identity=identity,
            crb=crb_for(model), task=task, readonly=True, shared_data=shared)
    finally:
        if prior_cache_root is None:
            os.environ.pop('RAGE_GRADIENT_CACHE_ROOT',None)
        else:
            os.environ['RAGE_GRADIENT_CACHE_ROOT'] = prior_cache_root
    try:
        if assets.cache.path != original_cache:
            raise ValueError('Readonly shape reconstruction selected a different cache namespace')
        if audit['new_bp'] != 0 or assets.cache.computed_windows != 0:
            raise RuntimeError('Shape supplement must not compute gradients')
        source_fields = dict(model=identity,pool=manifest,crb=asdict(crb_for(model)),
            side='source',source_physical_identity='V7_verified_source_rows')
        source_path = Path(main_run)/'shape_backfill_cache'/'source'/f'seed{seed}'/model
        if Path(parent).resolve() != Path(main_run).resolve():
            # Extension source shards are already field/key-verified hardlinks
            # to the original run; the feature cache still checks endpoints.
            source_reuse = Path(parent)/'gradient_cache'
            if not any(source_reuse.glob('*/extension_source_reuse.json')):
                raise RuntimeError('Extension has no verified source reuse receipt')
        source, source_cost = feature_cache(source_path,cache=assets.cache,dataset='source',
            index=index,blocks=assets.source_blocks,fields=source_fields)
        reference_fields = dict(model=identity,task=task,crb=asdict(crb_for(model)),
            side='reference',native_input=read(Path(parent)/'inputs'/model/f'{task}__manifest.json'))
        reference_path = Path(parent)/'shape_backfill_cache'/'reference'/model/task
        reference, reference_cost = feature_cache(reference_path,cache=assets.cache,
            dataset=assets.reference_dataset,index=ref_index,
            blocks=assets.reference_blocks,fields=reference_fields)
        weights = reference_weights(assets.reference_blocks,assets.reference_results)
        units = [b.start_positions for b in assets.source_blocks]
        counts = np.asarray([len(unit) for unit in units], dtype=np.int64)
        if score_cache.is_file():
            with np.load(score_cache, allow_pickle=False) as saved:
                if (json.loads(str(saved['identity'])) != artifact or
                        not np.array_equal(saved['source_block_counts'],counts)):
                    raise ValueError('Shape score cache identity or block counts differ')
                scored = {mode:(saved[mode],dict(score_cache_hit=True,aggregation=mode))
                          for mode in modes}
            scoring_seconds = 0.0
            score_cache_hit = True
        else:
            before = assets.cache.computed_windows
            score_started = time.perf_counter()
            scored = scores_many(reference,source,weights,aggregations=modes)
            scoring_seconds = time.perf_counter()-score_started
            if assets.cache.computed_windows != before:
                raise RuntimeError('Shape scoring performed unexpected BP')
            v7_io.atomic_npz(score_cache,identity=json.dumps(artifact),source_block_counts=counts,
                **{mode:scored[mode][0] for mode in modes})
            score_cache_hit = False
        records = []
        for arm in selected_arms:
            scores, score_audit = scored[arm.removeprefix('rage-shape-')]
            began = time.perf_counter()
            if paths[arm].is_file():
                existing = v7_io.load(paths[arm],index,manifest,artifact)
                records.append(dict(arm=arm,path=str(paths[arm]),
                    selection_file_bytes=paths[arm].stat().st_size,
                    score_and_save_seconds=0.0,reused=True,
                    coverage=existing.selected.coverage()))
                del existing
                continue
            _, mass, rank_audit = rank_mass(scores, counts)
            selected = ResampledWindows.from_index(index,units,mass,pool_manifest=manifest,
                metadata=dict(model=identity,task=task,seed=seed,scoring_unit='CRB_block_shape',
                    shape_protocol=config['shape_amendment'],method=arm,
                    gradient_asset_identity=assets.cache.manifest,
                    reference_weight_sum=float(weights.sum()),
                    reference_audit={k:v for k,v in score_audit.items() if k!='source_scores'},
                    rank_probability=rank_audit,frozen_rank_protocol=frozen_rank_protocol(),
                    source_length_applied_once=True,selection_new_bp=0))
            selected.native_scores = scores
            coverage = v7_io.validate_support(selected,arm)
            output = ArmOutput(arm,selected,[],SelectionCost(gradient_computations=0,
                index_bytes=scores.nbytes,wall_seconds=time.perf_counter()-began,
                notes=dict(shared_scoring_seconds=scoring_seconds,source_feature_cache=source_cost,
                    reference_feature_cache=reference_cost,required_source_bp=audit['plans']['source_audit']['required_bp'],
                    required_reference_bp=audit['plans']['reference_audit']['required_bp'])),{})
            v7_io.save(paths[arm],output,artifact)
            records.append(dict(arm=arm,path=str(paths[arm]),selection_file_bytes=paths[arm].stat().st_size,
                                score_and_save_seconds=time.perf_counter()-began,coverage=coverage))
            del output, selected
        row=dict(identity=artifact,complete=True,selection_new_bp=0,
            elapsed_seconds=time.perf_counter()-started,score_seconds=scoring_seconds,
            score_cache_hit=score_cache_hit,score_cache_file_bytes=score_cache.stat().st_size,
            source_features=source_cost,reference_features=reference_cost,
            source_feature_file_bytes=(source_path/'features.npy').stat().st_size,
            reference_feature_file_bytes=(reference_path/'features.npy').stat().st_size,
            source_blocks=len(units),reference_blocks=len(assets.reference_blocks),
            required_source_bp=audit['plans']['source_audit']['required_bp'],
            required_reference_bp=audit['plans']['reference_audit']['required_bp'],
            methods=records)
        publish_receipt(receipt,row)
        return row
    finally:
        assets.cache.close()


def complete_task(sidecar, config, model, seed, task, manifest, identity):
    root = folder(sidecar,model,seed)
    expected = dict(artifact_identity(config,model,seed,task,manifest,identity),
                    training_protocol=config['training_protocol'])
    selected_arms = tuple(config['arms'][2:])
    for arm in selected_arms:
        row = report_view(read(root/'cells'/f'{task}__{arm}.json'),expected,config,model)
        if row.get('arm') != arm or row.get('seed') != seed or row.get('task') != task:
            raise ValueError('Shape cell identity differs')
    receipt = read(root/'shape_select'/f'{task}.json')
    publish_receipt(root/'shape_complete'/f'{task}.json',dict(identity=expected,complete=True,
        method_count=len(selected_arms),selection_elapsed_seconds=receipt['elapsed_seconds'],
        selection_storage_bytes=sum(r['selection_file_bytes'] for r in receipt['methods']),
        zero_new_bp=True,time=time.time()))


def run_model(main_run, parents, configs, model, arm, *, seed_filter=None, task_filter=None):
    import torch
    if arm not in ARMS:
        raise ValueError('Unknown shape arm')
    main_run = Path(main_run).resolve()
    c,h = WINDOWS[model]
    engine, probe, identity = build_model(model,model_path=configs[0]['model_paths'].get(model),
                                           context_length=c,prediction_length=h)
    if read(main_run/'models'/f'{model}.json') != identity:
        raise ValueError('Shape model differs from original V7 identity')
    base = {k:v.detach().cpu().clone() for k,v in engine.model.state_dict().items()}
    sidecars = [create_sidecar(parent,conf) for parent,conf in zip(parents,configs)]
    for seed in (0,1):
        if seed_filter is not None and seed != seed_filter:
            continue
        for parent, parent_config, (sidecar, config) in zip(parents,configs,sidecars):
            if task_filter is not None and not any(t in task_filter for t in config['datasets']):
                continue
            if read(Path(parent)/'models'/f'{model}.json') != identity:
                raise ValueError('Task model differs from original V7 identity')
            feasible = load_fragments(Path(parent)/'inputs/feasible.npz')
            pool, manifest = seed_pool(parent,seed,feasible)
            index = WindowIndex(pool,context_length=c,prediction_length=h)
            for task in config['datasets']:
                if task_filter is not None and task not in task_filter:
                    continue
                root = folder(sidecar,model,seed)
                if arm not in config['arms']:
                    raise ValueError('Shape arm is not in this sidecar protocol')
                done = root/'shape_complete'/f'{task}.json'
                if done.exists():
                    expected = dict(artifact_identity(config,model,seed,task,manifest,identity),
                                    training_protocol=config['training_protocol'])
                    if read(done).get('identity') != expected:
                        raise ValueError('Completed shape task differs')
                    continue
                copy_baselines(parent,sidecar,parent_config,config,model,seed,task,manifest,identity)
                cell = root/'cells'/f'{task}__{arm}.json'
                if cell.is_file():
                    expected = dict(artifact_identity(config,model,seed,task,manifest,identity),
                                    training_protocol=config['training_protocol'])
                    report_view(read(cell),expected,config,model)
                    if arm == config['arms'][-1]:
                        complete_task(sidecar,config,model,seed,task,manifest,identity)
                    print('V7_SHAPE_ARM_TASK_REUSED',arm,model,seed,task,flush=True)
                    continue
                select_shape(parent,sidecar,parent_config,config,model,seed,task,probe,identity,
                             index,pool,manifest,main_run)
                engine.model.load_state_dict(base)
                phase_config = dict(config, arms=['orig', 'rand', arm])
                train_task(phase_config,sidecar,model,seed,task,engine,index,manifest,identity)
                if arm == config['arms'][-1]:
                    complete_task(sidecar,config,model,seed,task,manifest,identity)
                print('V7_SHAPE_ARM_TASK_COMPLETE',arm,model,seed,task,flush=True)
            del feasible,pool,index
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def checked_model_lease(run, model):
    """One worker per model, independent of the finished original run."""
    path = Path(run)/'shape_backfill_20260925'/f'{model}.lock'
    path.parent.mkdir(parents=True,exist_ok=True)
    handle = path.open('a+')
    fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle
