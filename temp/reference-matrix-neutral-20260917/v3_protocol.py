"""V3 canonical method names and frozen scientific method definitions."""
import json
from pathlib import Path


V3_MATRIX = ('orig', 'rand', 'tsds', 'lp-mirror', 'less-main', 'rags-main', 'rags-traj')
V3_BASES = {'rand': 'random', 'lp-mirror': 'lp_mirror', 'less-main': 'less',
            'rags-main': 'rags', 'rags-traj': 'rags_traj'}
V3_DATASETS = ('time_nem_demand_5t', 'time_azure_i_cpu_5t', 'time_darwin_water_15t',
               'time_finland_15t', 'time_eweld_15t', 'time_cphl_h',
               'time_nem_price_5t', 'time_solar_h', 'time_port_activity_d')
V3_TEMPERATURE_ARMS = ('lp-mirror', 'less-main', 'rags-main', 'rags-traj')
V3_GRADIENT_ARMS = ('less-main', 'rags-main', 'rags-traj')


def v3_temperatures(config):
    profile = config.get('temperature_protocol', 'reference-mean-t025')
    if profile not in ('reference-mean-t025', 'reference-mean-t005'):
        raise ValueError('Unknown V3 temperature protocol')
    gradient_temperature = .05 if profile == 'reference-mean-t005' else .25
    return {arm: gradient_temperature for arm in V3_TEMPERATURE_ARMS}


V3_METHOD_PROTOCOLS = {
    'orig': {'kind': 'pretrained_evaluation', 'training': False},
    'rand': {'kind': 'unchanged_uniform_legal_window_sampling'},
    'tsds': {'kind': 'unchanged_native_tsds_distribution'},
    'lp-mirror': {'kind': 'unchanged_lp_mirror', 'mapping': 'raw_value_softmax',
                  'predecessor': 'lpv_t025'},
    'less-main': {
        'gating': False, 'competition': 'global', 'scoring_unit': 'trajectory',
        'anchors': 'rounded_linspace_uniform_endpoints',
        'trajectory_gradient': 'arithmetic_mean_of_raw_sampled_window_gradients',
        'reference_trajectory_weight': '1/N', 'reference_total_weight': 1,
        'reference_aggregation': 'mean_cosine', 'probability': 'softmax(score/T)',
        'source_length_weighting': False, 'post_softmax_weighting': False,
    },
    'rags-main': {
        'gating': False, 'competition': 'global', 'scoring_unit': 'block',
        'anchors': 'CRB_coverage_error_closest_zero_batched_split',
        'block_gradient': 'linear_interpolation_geometric_midpoint',
        'reference_levels': 2, 'reference_trajectory_weight': '1/N',
        'reference_block_weight': '(1/N)*(owned_legal_starts/trajectory_legal_starts)',
        'reference_total_weight': 1, 'weights_apply_to': 'cosine_score_only',
        'probability': 'softmax(score/T)', 'source_length_weighting': False,
        'post_softmax_weighting': False, 'p10_filters_split': False,
        'retains_all_legal_starts': True,
    },
    'rags-traj': {
        'gating': False, 'competition': 'global', 'scoring_unit': 'trajectory',
        'anchors': 'CRB_coverage_error_closest_zero_batched_split',
        'trajectory_gradient': 'arithmetic_mean_of_raw_sampled_window_gradients',
        'reference_trajectory_weight': '1/N', 'reference_total_weight': 1,
        'reference_aggregation': 'mean_cosine', 'probability': 'softmax(score/T)',
        'source_length_weighting': False, 'post_softmax_weighting': False,
        'same_per_trajectory_budget_as_less': True,
        'difference_from_less_main': 'anchor_positions_only',
    },
}



def opinion_method_protocols(arms=V3_MATRIX):
    """New defaults are versioned separately; historical V3 stays immutable."""
    from copy import deepcopy
    from .reference_opinions import OPINION_BASES, selection_options
    protocols = {}
    for arm in arms:
        family = ({'less': 'less-main', 'rags': 'rags-main', 'rags_traj': 'rags-traj',
                   'rags_uniform': 'rags-main'}.get(OPINION_BASES.get(arm)) or arm)
        protocol = deepcopy(V3_METHOD_PROTOCOLS[family])
        if family in V3_GRADIENT_ARMS:
            options = selection_options(arm)
            # Preserve the preceding, never-launched v1 design as historical.
            if arm in ('rags-main', 'rags-traj'):
                options = dict(aggregation='zeroClipSum', competition='global')
            protocol.update(reference_aggregation=options['aggregation'],
                            competition=options['competition'],
                            reference_aggregation_revision='reference-opinion-v1')
            protocol['post_softmax_weighting'] = options['competition']=='local'
            protocol['weights_apply_to'] = ('fixed_reference_row_mass' if options['competition']=='local'
                else 'not_used_by_max' if options['aggregation']=='maxSum' else 'transformed_score_sum')
            if options['aggregation']=='expSum':
                protocol['exp_inner_scale'] = 1.
                protocol['probability'] = 'softmax(sum(alpha*exp(raw_cosine))/T)'
            if family=='rags-traj':
                protocol['difference_from_less_main'] = 'anchor_positions_and_reference_aggregation'
                protocol['difference_from_less_with_same_aggregation'] = 'anchor_positions_only'
            if OPINION_BASES.get(arm)=='rags_uniform':
                protocol['anchors'] = 'CRB_error_stage_uniform_same_actual_K'
        protocols[arm] = protocol
    return protocols


def validate_opinion_config(config):
    from copy import deepcopy
    from .reference_opinions import OPINION_ARMS
    allowed = set(V3_MATRIX + OPINION_ARMS)
    arms = config['arms']
    if not arms or len(arms)!=len(set(arms)) or any(arm not in allowed for arm in arms):
        raise ValueError('Unknown/duplicate reference opinion arm')
    if config.get('method_protocols') != opinion_method_protocols(arms):
        raise ValueError('Reference opinion method_protocols differ from the named definitions')
    if config.get('temperature_protocol') != 'reference-mean-t005':
        raise ValueError('Reference opinion protocol defaults to T005')
    for student in config['students']:
        expected = {arm: .05 for arm in arms if arm not in ('orig','rand','tsds')}
        if config['settings'][student]['temperature'] != expected:
            raise ValueError('Reference opinion methods require shared T005')
    historical = deepcopy(config)
    historical.pop('opinion_aggregation_protocol')
    historical.update(arms=list(V3_MATRIX),method_protocols=deepcopy(V3_METHOD_PROTOCOLS))
    for student in historical['students']:
        historical['settings'][student]['temperature'] = v3_temperatures(historical)
    validate_v3_config(historical)


NEUTRAL_MATRIX = ('rand', 'orig', 'less-main',
    'rags-block-linearSum', 'rags-block-zeroClipSum', 'rags-block-maxSum', 'rags-block-expSum',
    'rags-traj-linearSum', 'rags-traj-zeroClipSum', 'rags-traj-maxSum', 'rags-traj-expSum',
    'tsds', 'lp-mirror')


def neutral_method_protocols():
    protocols = opinion_method_protocols(NEUTRAL_MATRIX)
    for name, protocol in protocols.items():
        if name.startswith('rags-'):
            protocol.update(method_role='unranked_variant', main_method_selected=False)
            if name.startswith('rags-traj-'):
                protocol['difference_from_less_main'] = ('anchor_positions_only'
                    if protocol['reference_aggregation']=='maxSum'
                    else 'anchor_positions_and_reference_aggregation')
    return protocols


def validate_neutral_config(config):
    from copy import deepcopy
    if config.get('arms') != list(NEUTRAL_MATRIX):
        raise ValueError('Neutral matrix requires exactly the thirteen named methods')
    if config.get('rags_main_method') is not None or config.get('rags_main_method_selected') is not False:
        raise ValueError('Neutral matrix cannot declare a RAGS main method')
    if config.get('method_protocols') != neutral_method_protocols():
        raise ValueError('Neutral matrix method definitions differ from named variants')
    compatible = deepcopy(config)
    compatible['opinion_aggregation_protocol'] = 'reference-opinion-v1'
    compatible['method_protocols'] = opinion_method_protocols(compatible['arms'])
    validate_opinion_config(compatible)

def validate_v3_config(config):
    if config.get("opinion_aggregation_protocol") == "reference-opinion-v2-neutral":
        return validate_neutral_config(config)
    if config.get("opinion_aggregation_protocol"):
        if config["opinion_aggregation_protocol"] != "reference-opinion-v1":
            raise ValueError("Unknown reference opinion protocol")
        return validate_opinion_config(config)
    from .resampling import REVISION
    from .students import STUDENTS
    required = dict(experiment_version='v3', purpose='formal', resampling_revision=REVISION,
        students=list(STUDENTS), datasets=list(V3_DATASETS), arms=list(V3_MATRIX), seeds=[0],
        pool_equivalents=16284, context_length=512, prediction_length=64,
        task_protocol='shared-ten-formal-20260915', method_protocols=V3_METHOD_PROTOCOLS)
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f'V3 requires {key}={value!r}')
    for key, value in dict(cover_density=.02, error_density=.01, split_density=.01,
                           split_policy='closest_zero_v2', epsilon_policy='trajectory_p10').items():
        if config['crb'].get(key) != value:
            raise ValueError(f'V3 frozen CRB requires {key}={value}')
    decisions = json.loads(Path(config['shared_task_decisions']).read_text())
    if decisions['datasets'] != list(V3_DATASETS):
        raise ValueError('V3 shared task decisions differ from the frozen nine tasks')
    for student in config['students']:
        settings, recipe = config['settings'][student], decisions['models'][student]
        expected = dict(learning_rate=recipe['learning_rates'][0], epochs=recipe['epochs'][0],
            batch_size=recipe['batch_size'], weight_decay=recipe['weight_decay'],
            optimizer=recipe['optimizer'], schedule_epochs=recipe['schedule_epochs'])
        for key, value in expected.items():
            if settings.get(key) != value:
                raise ValueError(f'V3 frozen training recipe differs: {student}.{key}')
        if not settings.get('development_evidence'):
            raise ValueError(f'Missing development evidence: {student}')
        if settings.get('temperature') != v3_temperatures(config):
            raise ValueError(f'V3 temperature profile differs: {student}')
