"""V4: two active models, neutral opinions, one natural seed per round."""
import json
from pathlib import Path

from .v3_protocol import NEUTRAL_MATRIX, V3_DATASETS, neutral_method_protocols

V4_STUDENTS = ('chronos-base', 'timesfm-2.5')
SEED_SCHEDULE = dict(mode='serial_rounds', start=0, step=1, stop=3,
                     barrier='all_models_tasks_methods_complete', parallelism='within_round')


def validate_v4_config(config):
    from .resampling import REVISION
    required = dict(experiment_version='v4', purpose='formal',
        resampling_revision=REVISION, students=list(V4_STUDENTS),
        datasets=list(V3_DATASETS), arms=list(NEUTRAL_MATRIX),
        pool_equivalents=16284, context_length=512, prediction_length=64,
        task_protocol='shared-ten-formal-20260915', temperature_protocol='reference-mean-t005',
        opinion_aggregation_protocol='reference-opinion-v4-mixed-weight-order',
        method_protocols=neutral_method_protocols(weighted_first=True, exp_after=True),
        rage_main_method=None, rage_main_method_selected=False, seed_schedule=SEED_SCHEDULE,
        original_first=True)
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f'V4 requires {key}={expected!r}')
    seeds = config.get('seeds')
    if not isinstance(seeds, list) or not (seeds==[0,1,2] or (len(seeds)==1 and type(seeds[0]) is int and seeds[0] in (0,1,2))):
        raise ValueError('V4 suite seeds must be [0,1,2], or a single round seed from 0,1,2')
    if config.get('student_parallelism') != 2:
        raise ValueError('V4 requires both active models parallel within each seed round')
    for field in ('settings','model_paths','student_environments','student_selection_execution'):
        if set(config.get(field, {})) - set(V4_STUDENTS):
            raise ValueError(f'V4 {field} cannot include inactive models')
    for key,value in dict(cover_density=.02,error_density=.01,split_density=.01,
                          split_policy='closest_zero_v2',epsilon_policy='trajectory_p10').items():
        if config['crb'].get(key)!=value:
            raise ValueError(f'V4 frozen CRB requires {key}={value}')
    decisions=json.loads(Path(config['shared_task_decisions']).read_text())
    if decisions['datasets']!=list(V3_DATASETS):
        raise ValueError('V4 requires the same nine shared tasks')
    for student in V4_STUDENTS:
        settings=config['settings'][student];recipe=decisions['models'][student]
        expected=dict(learning_rate=recipe['learning_rates'][0],epochs=recipe['epochs'][0],
            batch_size=recipe['batch_size'],weight_decay=recipe['weight_decay'],
            optimizer=recipe['optimizer'],schedule_epochs=recipe['schedule_epochs'])
        for key,value in expected.items():
            if settings.get(key)!=value:
                raise ValueError(f'V4 frozen recipe differs: {student}.{key}')
        if not settings.get('development_evidence'):
            raise ValueError(f'Missing V4 development evidence: {student}')
        if settings.get('temperature')!={a:.05 for a in NEUTRAL_MATRIX if a not in ('orig','rand','tsds')}:
            raise ValueError('V4 all sampling Softmax temperatures must be .05')
