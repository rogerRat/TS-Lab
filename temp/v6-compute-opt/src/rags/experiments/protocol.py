"""Frozen V5 scientific identity and inherited model/task/training matrix."""
import json
from pathlib import Path
from dataclasses import asdict
from rags.crb import CRBConfig
from .full_pool import ARMS
from .ranking import frozen_rank_protocol,validate_frozen_rank_protocol,DEFAULT_TEMPERATURE
from .reference_opinions import selection_options,weight_placement_for,UNI_ARMS,UNI_MODES,is_block_uni_arm,is_win_uni_arm

STUDENTS=('chronos-base','timesfm-2.5')
DATASETS=('time_nem_demand_5t','time_azure_i_cpu_5t','time_darwin_water_15t',
    'time_finland_15t','time_eweld_15t','time_cphl_h','time_nem_price_5t','time_solar_h','time_port_activity_d')
SEED_SCHEDULE=dict(mode='serial_rounds',start=0,step=1,stop=3,
    barrier='all_models_tasks_methods_complete',parallelism='within_round')
EXPERIMENT_VERSIONS=('v5','v5-uni','v5-timerxl','v6-timerxl')

# The approved-frozen V5 formal matrix; never extended in place.
V5_FROZEN_ARMS=('rand','orig','less-main','rags-block-linearSum','rags-block-zeroClipSum',
    'rags-block-maxSum','rags-block-expSum','rags-traj-linearSum','rags-traj-zeroClipSum',
    'rags-traj-maxSum','rags-traj-expSum','tsds','lp-mirror')
# Experiment 3 (v5-uni): identical protocol/training matrix; the eight RAGS
# block/traj arms are replaced by the two intra-trajectory uniformized variants
# of the strongest block arms; all non-RAGS baselines are kept in-run.
V5_UNI_ARMS=('rand','orig','less-main',f'rags-block-{UNI_MODES[0]}-uni',
    f'rags-block-{UNI_MODES[1]}-uni','tsds','lp-mirror')
# Timer-XL supplementary run (approved PLAN temp/timerxl-plan-20260920): the third
# student over the union of the frozen V5 and V5-uni matrices (13 + 2 uni = 15
# unique arms), same nine tasks, pool, protocol, CRB and seed barrier in one run.
TIMERXL_STUDENTS=('timerxl-84m',)
V5_TIMERXL_ARMS=V5_FROZEN_ARMS+tuple(f'rags-block-{mode}-uni' for mode in UNI_MODES)
V6_TIMERXL_ARMS=('orig',)+tuple(f'rags-block-{mode}-win-uni' for mode in UNI_MODES)+tuple(f'rags-block-{mode}-block-uni' for mode in UNI_MODES)+tuple(arm for arm in V5_FROZEN_ARMS if arm != 'orig')
FROZEN_ARMS_BY_VERSION={'v5': V5_FROZEN_ARMS, 'v5-uni': V5_UNI_ARMS,
    'v5-timerxl': V5_TIMERXL_ARMS, 'v6-timerxl': V6_TIMERXL_ARMS}
STUDENTS_BY_VERSION={'v5': STUDENTS, 'v5-uni': STUDENTS, 'v5-timerxl': TIMERXL_STUDENTS}
STUDENTS_BY_VERSION['v6-timerxl'] = TIMERXL_STUDENTS
STUDENT_PARALLELISM_BY_VERSION={'v5': 2, 'v5-uni': 2, 'v5-timerxl': 1, 'v6-timerxl': 1}


def method_protocols(arms=ARMS):
    result={}
    for arm in arms:
        if arm in ('orig','rand','tsds','lp-mirror'):
            result[arm]=dict(kind={'orig':'pretrained_evaluation','rand':'uniform_legal_windows',
                'tsds':'unchanged_native_tsds','lp-mirror':'unchanged_native_lp_mirror'}[arm])
            continue
        result[arm]=dict(**selection_options(arm),reference_weights_stage='score_aggregation',
            weight_placement=weight_placement_for(arm),source_prior='legal_window_counts_after_rank_mapping',
            rank_protocol=frozen_rank_protocol()['protocol_id'],gating=False,post_rank_softmax=False,
            raw_representatives_retain_norm=True,matching='cosine_after_representative_normalization',
            anchors='uniform_rounded_linspace' if arm=='less-main' else 'CRB_coverage_risk_min_kappa',
            representative='raw_endpoint_geometric_midpoint' if arm.startswith('rags-block') else 'raw_anchor_arithmetic_mean',
            reference_weight='1/N_ref*owned_starts/trajectory_starts' if arm.startswith('rags-block') else '1/N_ref',
            additional_bp_allowed=False)
        if is_block_uni_arm(arm):
            result[arm]['intra_trajectory_uniformization']='uniform_block_share'
        elif is_win_uni_arm(arm):
            result[arm]['intra_trajectory_uniformization']='uniform_trajectory_window'
    return result


def validate_config(config):
    from .resampling import REVISION
    version=config.get('experiment_version')
    if version not in EXPERIMENT_VERSIONS or config.get('resampling_revision')!=REVISION:
        raise ValueError('Only the frozen V5 experiment protocols are supported')
    arms=config.get('arms',[])
    if not arms or len(arms)!=len(set(arms)) or any(a not in ARMS for a in arms):
        raise ValueError('Unknown or duplicate canonical method names')
    if version == 'v6-timerxl':
        expected_rank = dict(frozen_rank_protocol(), protocol_id='v6-rank-probability-frozen-20260921')
        if config.get('rank_protocol') != expected_rank:
            raise ValueError('V6 rank protocol fields differ')
    else:
        validate_frozen_rank_protocol(config.get('rank_protocol'))
    if config.get('method_protocols')!=method_protocols(arms):raise ValueError('V5 method definitions differ')
    CRBConfig(context_length=config['context_length'],prediction_length=config['prediction_length'],**config['crb'])
    for student in config['students']:
        if student not in config.get('settings',{}):raise ValueError('Missing model settings: '+student)
        temperatures=config['settings'][student]['temperature']
        for arm in arms:
            if arm.startswith('rags-') or arm=='less-main':
                if not isinstance(temperatures,dict) or temperatures.get(arm)!=DEFAULT_TEMPERATURE:
                    raise ValueError('V5 gradient methods require the frozen rank temperature')
    if config.get('purpose')!='formal':return
    frozen_arms=FROZEN_ARMS_BY_VERSION[version]
    students=STUDENTS_BY_VERSION[version]
    timer_v6 = version == 'v6-timerxl'
    required=dict(students=list(students),datasets=list(DATASETS),arms=list(frozen_arms),
        pool_equivalents=16284,context_length=(480 if timer_v6 else 512),
        prediction_length=(96 if timer_v6 else 64),
        student_parallelism=STUDENT_PARALLELISM_BY_VERSION[version],
        seed_schedule=SEED_SCHEDULE,original_first=True)
    for key,value in required.items():
        if config.get(key)!=value:raise ValueError(f'V5 frozen matrix differs: {key}')
    seeds=config.get('seeds')
    if not isinstance(seeds,list) or not (seeds==[0,1,2] or
            len(seeds)==1 and type(seeds[0]) is int and seeds[0] in (0,1,2)):
        raise ValueError('V5 seeds must be [0,1,2] or one natural seed in 0..2')
    for key,value in dict(cover_density=.02,error_density=.01,split_density=.01,
            split_policy='min_kappa',epsilon_policy='none',priority_tie_policy='right',
            initial_placement='longest',error_placement='risk').items():
        if config['crb'].get(key)!=value:raise ValueError(f'Frozen V5 CRB differs: {key}')
    profile=json.loads(Path('configs/rags/crb_protocol.json').read_text())
    if version != 'v6-timerxl' and config.get('crb_protocol')!=profile:raise ValueError('CRB protocol fields differ')
    decisions=json.loads(Path(config['shared_task_decisions']).read_text())
    if decisions['datasets']!=list(DATASETS):raise ValueError('Timer inherits the nine shared tasks')
    for student in students:
        recipe=decisions['models'][student];settings=config['settings'][student]
        expected=dict(learning_rate=recipe['learning_rates'][0],epochs=recipe['epochs'][0],
            batch_size=recipe['batch_size'],weight_decay=recipe['weight_decay'],
            optimizer=recipe['optimizer'],schedule_epochs=recipe['schedule_epochs'])
        if any(settings.get(k)!=v for k,v in expected.items()):raise ValueError('Inherited training recipe differs')
