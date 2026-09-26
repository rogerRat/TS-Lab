"""Independent V7 identity. Never changes the V5/V6 validators or configuration."""
from copy import deepcopy
from dataclasses import asdict
import numpy as np
from rags.crb import CRBConfig
from .protocol import DATASETS, V6_TIMERXL_ARMS, method_protocols
from .ranking import frozen_rank_protocol

VERSION = 'v7-seeded-quarter-pools-20260923'
NEW_ARMS = ('grad-dot', 'grad-cos', 'grand-static')
ARMS = V6_TIMERXL_ARMS + NEW_ARMS
MODELS = ('chronos-base', 'timesfm-2.5', 'timerxl-84m')
WINDOWS = {'chronos-base': (512, 64), 'timesfm-2.5': (512, 64), 'timerxl-84m': (480, 96)}
EPOCHS = dict(zip(MODELS, (3, 3, 4)))

def gradient_method_protocol(arm):
    if arm not in NEW_ARMS:
        raise ValueError(arm)
    return dict(version=VERSION, state='single_pretrained_checkpoint', dimension=4096,
        projection_seed=42, representation='raw_projected_gradient',
        budget='RAGS_actual_distinct_BP_per_side_per_trajectory',
        anchors='unique_rint_linspace_endpoints; K=1 prefers left',
        interpolation='within_trajectory_piecewise_linear',
        reference='mean_legal_windows_then_mean_trajectories' if arm != 'grand-static' else 'none',
        score={'grad-dot': 'signed_dot', 'grad-cos': 'cosine_zero_norm_is_zero',
               'grand-static': 'gradient_l2'}[arm],
        numerics='float64_piecewise_dot_Gram_norm; cancellation_recomputed_from_raw_vectors',
        objects='legal_windows_m1', rank=frozen_rank_protocol(), additional_bp_allowed=False)

def crb_for(student):
    c, h = WINDOWS[student]
    return CRBConfig(context_length=c, prediction_length=h,
        cover_density=.005, error_density=.005, split_density=.005)

def make_config(ct, timer):
    settings = {s: deepcopy((timer if s == 'timerxl-84m' else ct)['settings'][s]) for s in MODELS}
    for s, setting in settings.items():
        setting['epochs'] = EPOCHS[s]
        for arm in ARMS:
            if arm not in ('orig', 'rand', 'tsds', 'lp-mirror'):
                setting['temperature'][arm] = frozen_rank_protocol()['temperature']
    return dict(experiment_version=VERSION, purpose='formal', students=list(MODELS),
        datasets=list(DATASETS), arms=list(ARMS), seeds=[0, 1, 2],
        feasible_pool_equivalents=65136, training_pool_equivalents=16284,
        pool_seed=0, source_load=ct['source_load'], settings=settings,
        shared_task_decisions=ct['shared_task_decisions'],
        legacy_inputs={s: (timer if s == 'timerxl-84m' else ct)['inputs'] for s in MODELS},
        model_paths=dict(ct.get('model_paths', {}), **timer.get('model_paths', {})),
        student_environments=dict(ct.get('student_environments', {}),
            **{'timerxl-84m': '/root/miniconda3/envs/ts3/bin/python'}),
        windows={s: list(WINDOWS[s]) for s in MODELS},
        crb={s: asdict(crb_for(s)) for s in MODELS}, rank_protocol=frozen_rank_protocol(),
        method_protocols=dict(method_protocols(V6_TIMERXL_ARMS),
            **{a: gradient_method_protocol(a) for a in NEW_ARMS}),
        selection_execution=dict(cpu_threads=4, gradient_batch=8, gradient_workers=1,
            gradient_vjp_batch=16),
        legacy_gradient_roots=['/root/autodl-tmp/TS3/cache/grad-pool'],
        phase_order=['inputs', 'assets_all_models_tasks', 'selection_zero_bp', 'seed_task_arms_training'],
        timer_validation_manifest='/root/autodl-tmp/TS3/outputs/main-v7-20260923/inputs/timer_validation.json',
        checkpoint_retention='completed_cell_cleanup',
        checkpoint_selection='timer_isolated_validation_min_mase_earliest_tie; CT_final_epoch',
        storage='atomic_explicit_field_identity_no_hash',
        input_reuse='same_native_reference_exam_fields_only; new_exact_source_pool',
        run_root='/root/autodl-tmp/TS3/outputs/main-v7-20260923')

def validate(config):
    for key, expected in dict(experiment_version=VERSION, students=list(MODELS),
            datasets=list(DATASETS), arms=list(ARMS), seeds=[0,1,2],
            feasible_pool_equivalents=65136, training_pool_equivalents=16284,
            pool_seed=0, rank_protocol=frozen_rank_protocol(),
            windows={s: list(WINDOWS[s]) for s in MODELS},
            crb={s: asdict(crb_for(s)) for s in MODELS},
            method_protocols=dict(method_protocols(V6_TIMERXL_ARMS),
                **{a: gradient_method_protocol(a) for a in NEW_ARMS})).items():
        if config.get(key) != expected:
            raise ValueError('Frozen V7 field differs: '+key)
    for s in MODELS:
        t = config['settings'][s]
        expected = dict(epochs=EPOCHS[s], schedule_epochs=6, batch_size=32,
            learning_rate={'chronos-base': .001, 'timesfm-2.5': .00003, 'timerxl-84m': .0001}[s],
            weight_decay=0 if s == 'chronos-base' else .01,
            optimizer=dict(betas=[.9,.999], eps=1e-8, max_grad_norm=1, micro_batch_size=32,
                min_lr_ratio=0, schedule='linear' if s == 'chronos-base' else 'cosine', warmup_ratio=0))
        if any(t.get(k) != v for k, v in expected.items()):
            raise ValueError('Frozen V7 training recipe differs: '+s)
        if any(t['temperature'].get(a) != frozen_rank_protocol()['temperature']
               for a in ARMS if a.startswith(('rags-', 'grad')) or a == 'less-main'):
            raise ValueError('Frozen gradient rank temperature differs')

def exposure_audit(config):
    return {s: dict(pool_n=16284, feasible_pool_equivalents=65136,
        per_epoch_exposure_ratio_vs_v6=1,
        max_epochs=EPOCHS[s], max_steps=int(np.ceil(16284 * EPOCHS[s] / 32)),
        schedule_steps=int(np.ceil(16284 * 6 / 32)),
        sampled_windows=int(np.ceil(16284 * EPOCHS[s] / 32)) * 32)
        for s in MODELS}
