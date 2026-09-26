"""One exploratory TIME task, two seeds; immutable source resources hardlinked."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import fcntl
import numpy as np
from assets.gradient.cache import atomic_json

TASK = dict(id='time_ne_china_wind_h',benchmark='time',dataset='NE_China_Wind/H',
            evaluation_config='configs/eval/time/official_all.yaml',seasonality=24)
REVISION = '83e3d0b3be28d11c7182bffcc1892d19b36c4da1'
EXTENSION = 'v7-exploratory-ne-china-wind-two-seeds-20260925'
CANDIDATES = [TASK,
    dict(id='time_sg_pm25_h',benchmark='time',dataset='SG_PM25/H',
         evaluation_config='configs/eval/time/official_all.yaml',seasonality=24),
    dict(id='time_coastal_ts_h',benchmark='time',dataset='Coastal_T_S/H',
         evaluation_config='configs/eval/time/official_all.yaml',seasonality=24)]

# Pre-exam geometry only: finite references, native exam windows, no model
# performance screening.  The queue retains this order and stops explicitly
# if these finite pinned TIME candidates are exhausted or disk admission fails.
V7_E_CANDIDATES = [
    dict(id='time_supply_chain_customer_d', benchmark='time', dataset='Supply_Chain_Customer/D',
         evaluation_config='configs/eval/time/official_all.yaml', seasonality=7),
    dict(id='time_supply_chain_location_d', benchmark='time', dataset='Supply_Chain_Location/D',
         evaluation_config='configs/eval/time/official_all.yaml', seasonality=7),
    dict(id='time_azure_u_5t', benchmark='time', dataset='azure2019_U/5T',
         evaluation_config='configs/eval/time/official_all.yaml', seasonality=288),
    dict(id='time_current_velocity_5t', benchmark='time', dataset='current_velocity/5T',
         evaluation_config='configs/eval/time/official_all.yaml', seasonality=288),
    dict(id='time_current_velocity_15t', benchmark='time', dataset='current_velocity/15T',
         evaluation_config='configs/eval/time/official_all.yaml', seasonality=96),
    dict(id='time_epf_electricity_price_h', benchmark='time', dataset='epf_electricity_price/H',
         evaluation_config='configs/eval/time/official_all.yaml', seasonality=24),
    dict(id='time_ecdc_covid_d', benchmark='time', dataset='ECDC_COVID/D',
         evaluation_config='configs/eval/time/official_all.yaml', seasonality=7),
    dict(id='time_crypto_d', benchmark='time', dataset='Crypto/D',
         evaluation_config='configs/eval/time/official_all.yaml', seasonality=7),
    dict(id='time_coastal_ts_20t', benchmark='time', dataset='Coastal_T_S/20T',
         evaluation_config='configs/eval/time/official_all.yaml', seasonality=72),
    dict(id='time_current_velocity_h', benchmark='time', dataset='current_velocity/H',
         evaluation_config='configs/eval/time/official_all.yaml', seasonality=24),
]


def v7_e_native_arms(shape_arms):
    from .v7_shape import ARMS
    if (not isinstance(shape_arms, (list, tuple)) or len(shape_arms) != 2 or
            len(set(shape_arms)) != 2 or any(arm not in ARMS for arm in shape_arms)):
        raise ValueError('V7-e requires exactly two distinct frozen Shape arms')
    suffixes = [arm.removeprefix('rags-shape-') for arm in shape_arms]
    return ['orig', 'rand', *(f'rags-block-{s}' for s in suffixes),
            *(f'rags-traj-{s}' for s in suffixes), 'lp-mirror', 'tsds', 'less-main']


def make_config(parent, parent_run, run, task=None, *, shape_arms=None):
    task = TASK if task is None else task
    v7_e = task in V7_E_CANDIDATES
    if task not in CANDIDATES and not v7_e:
        raise ValueError('Task not in the user-approved exploratory queue')
    if v7_e:
        native_arms = v7_e_native_arms(shape_arms)
        version = 'v7-e-two-shape-two-seeds-20260926-'+task['id']
    else:
        if shape_arms is not None:
            raise ValueError('Original V7 extension cannot carry V7-e Shape choices')
        version = EXTENSION if task == TASK else 'v7-exploratory-'+task['id']+'-two-seeds-20260925'
    config = deepcopy(parent)
    config['datasets'] = [task['id']]
    if v7_e:
        config['arms'] = native_arms
    config['seeds'] = [0,1]
    config['run_root'] = str(run)
    config['legacy_inputs'] = {s:str(Path(run)/'inputs'/s) for s in config['students']}
    config['training_protocol']['datasets'] = [task['id']]
    config['training_protocol']['task_selection_basis'] = 'exploratory_reference_size_and_domain_prior_no_test_screening'
    config['asset_protocol'] = deepcopy(parent['asset_protocol'])
    config['asset_protocol'].update(datasets=[task['id']],seeds=[0,1],run_root=str(run),
                                  task_extension=version,task=task)
    config['task_extension'] = dict(version=version,task=task,parent_run=str(parent_run),
        parent_config=deepcopy(parent), source_reuse='verified_same_inode_immutable_shards',
        order='all_sparse_assets_then_zero_BP_selection_then_two_seed_training')
    if v7_e:
        config['task_extension'].update(v7_e_shape_arms=list(shape_arms),
            v7_e_native_arms=native_arms,phase='V7-e exploratory task loop')
    return config


def validate_extension(config):
    from .v7_protocol import validate
    ext = config['task_extension']
    parent = ext['parent_config']
    if 'task_extension' in parent:
        raise ValueError('Extension parent must be the completed original V7 matrix')
    validate(parent)
    if config != make_config(parent,ext['parent_run'],config['run_root'],ext['task'],
                             shape_arms=ext.get('v7_e_shape_arms')):
        raise ValueError('Unapproved task extension fields')


def link(src, dst):
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        if not os.path.samefile(src,dst):
            raise ValueError('Existing resource is not the validated inode: '+str(dst))
    else:
        os.link(src,dst)


def publish_source_cache(old, shared, root):
    """Only immutable source shard/CRB payloads share inodes, never indexes/manifests."""
    from .v7_run import read, freeze
    old, root = Path(old), Path(root)
    previous = read(old/'cache_manifest.json')
    old_data = previous['data']
    if any(shared[k] != old_data[k] for k in ('feasible_pool','actual_training_pools')):
        raise ValueError('Extension changed source pool/physical identities')
    fields = deepcopy(previous)
    fields['data'] = shared
    target = root/old.name.split('--fields-')[0]
    target.mkdir(parents=True,exist_ok=True)
    freeze(target/'cache_manifest.json',fields)
    receipt = target/'extension_source_reuse.json'
    if receipt.exists():
        audit = read(receipt)
        if audit['old_manifest'] != previous or audit['new_manifest'] != fields:
            raise ValueError('Published extension reuse identity differs')
        return audit
    with np.load(old/'window_index.npz',allow_pickle=False) as stored:
        data = {name:stored[name] for name in stored.files}
        source = data['dataset_ids'] == list(data['datasets']).index('source')
        old_shards = data['shards']
        indices = np.unique(data['shard_ids'][source])
        remap = {int(v):i for i,v in enumerate(indices)}
        names = old_shards[indices]
        arrays = dict(datasets=np.asarray(['source']),dataset_ids=np.zeros(int(source.sum()),dtype=np.int32),
            window_ids=data['window_ids'][source], shards=names,
            shard_ids=np.asarray([remap[int(v)] for v in data['shard_ids'][source]],dtype=np.int32),
            shard_rows=data['shard_rows'][source])
        # Verify every indexed physical key while linking each shard only once.
        for old_index,name in zip(indices,names):
            chosen = source & (data['shard_ids'] == old_index)
            with np.load(old/str(name),allow_pickle=False) as shard:
                if str(shard['dataset']) != 'source' or not np.array_equal(
                        shard['window_ids'][data['shard_rows'][chosen]],data['window_ids'][chosen]):
                    raise ValueError('Source shard physical keys differ')
            link(old/str(name),target/str(name))
    from .v7_io import atomic_npz
    atomic_npz(target/'window_index.npz',**arrays)
    crb_count = 0
    for marker in (old/'source-crb').glob('fields-*/cache_manifest.json'):
        crb = read(marker)
        if crb['gradient'] != previous:
            raise ValueError('Source CRB gradient fields differ')
        new_crb = deepcopy(crb)
        new_crb['gradient'] = fields
        directory = target/'source-crb'/marker.parent.name
        freeze(directory/'cache_manifest.json',new_crb)
        for payload in marker.parent.glob('trajectory-*.npz'):
            link(payload,directory/payload.name)
            crb_count += 1
    audit = dict(old_path=str(old),path=str(target),old_manifest=previous,new_manifest=fields,
        source_rows=len(arrays['window_ids']),source_shards=len(names),source_crb_payloads=crb_count,
        source_payload_new_bytes=0,source_bp_allowed=False,reference_rows_reused=0)
    atomic_json(receipt,audit)
    return audit


def prepare(config):
    from .v7_run import read, freeze
    from .v7_protocol import WINDOWS
    from .shared_formal_inputs import prepare_shared_formal_inputs
    validate_extension(config)
    task = config['task_extension']['task']
    version = config['task_extension']['version']
    run = Path(config['run_root']); parent = Path(config['task_extension']['parent_run'])
    out = run/'inputs'; out.mkdir(parents=True,exist_ok=True)
    # Parent physical manifest + hardlinked packed data are the source of truth.
    for name in ('feasible.npz','feasible_manifest.json','seed0_manifest.json','seed1_manifest.json'):
        link(parent/'inputs'/name,out/name)
    freeze(run/'protocol.json',config['asset_protocol'])
    freeze(run/'training_protocol.json',config['training_protocol'])
    freeze(run/'execution_limits.json',dict(version=version,allowed_training_seeds=[0,1]))
    if (parent/'execution.json').exists():
        freeze(run/'execution.json',read(parent/'execution.json'))
    declarations = out/'declarations'
    freeze(declarations/f"{task['id']}__manifest.json",dict(task=task,revision=REVISION))
    decisions = dict(datasets=[task['id']],input_roots={task['id']:str(declarations)})
    cache_root = run/'gradient_cache'
    cache_root.mkdir(exist_ok=True)
    audits = {}
    for student in config['students']:
        c,h = WINDOWS[student]
        native = prepare_shared_formal_inputs(decisions,out/student,context_length=c,prediction_length=h)
        data = read(parent/'inputs'/student/'gradient_data_identity.json')
        if data['feasible_pool'] != read(out/'feasible_manifest.json') or any(
                data['actual_training_pools'][str(s)] != read(out/f'seed{s}_manifest.json') for s in (0,1)):
            raise ValueError('Packed source/input manifests differ from gradient provenance')
        shared = deepcopy(data)
        shared['reference_identities'] = {task['id']:native[0]}
        shared['extension_source_reuse'] = dict(parent_run=str(parent),version=version)
        freeze(out/student/'gradient_data_identity.json',shared)
        link(parent/'models'/f'{student}.json',run/'models'/f'{student}.json')
        asset = read(parent/'seeds/seed0'/student/'assets/time_nem_demand_5t.json')['audit']
        if read(Path(asset['path'])/'cache_manifest.json') != asset['gradient_manifest']:
            raise ValueError('Parent gradient manifest differs from receipt')
        # No parent writer may run during index snapshot/immutable payload linking.
        from assets.gradient.cache import readable
        m = asset['gradient_manifest']['model']
        name = readable(m['model_id'])+'--'+readable(m['ckpt_revision'])
        with (Path(asset['path']).parent/(name+'.lock')).open('a+') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB)
            audits[student] = publish_source_cache(asset['path'],shared,cache_root)
        # Source TSDS features are model/seed identities, independent of target.
        for seed in (0,1):
            old_native = parent/f'seeds/seed{seed}'/student/'native_assets'
            for directory in old_native.iterdir():
                if (directory/'tsds-trajectory-features.npy').is_file():
                    for payload in directory.glob('tsds-trajectory-features*'):
                        if payload.is_file():
                            link(payload,run/f'seeds/seed{seed}'/student/'native_assets'/task['id']/payload.name)
                    break
    # One task/two seeds: conservative uncompressed selection arrays + 24GiB
    # model/optimizer headroom. Source payload hardlinks cost no new data blocks.
    legal = sum(read(out/f'seed{s}_manifest.json')['total_legal_windows'] for s in (0,1))
    estimate = legal*96*3 + 24*2**30
    free = shutil.disk_usage(run).free
    atomic_json(run/'input_preflight.json',dict(complete=True,disk_ready=free>estimate,
        estimated_new_bytes_upper_bound=estimate,free_bytes=free,source_reuse=audits,
        task=task,selection_basis='reference_size_domain_prior_no_test_performance',cache_root=str(cache_root)))
    if free <= estimate:
        raise RuntimeError('Extension disk headroom insufficient')
    return audits
