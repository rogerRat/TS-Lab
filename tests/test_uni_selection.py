"""RAGE-uni: intra-trajectory uniformization and v5-uni experiment identity."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rage.experiments import full_pool
from rage.experiments.full_pool import ARMS, validate_cached_selection
from rage.experiments.protocol import (V5_FROZEN_ARMS, V5_UNI_ARMS, method_protocols,
                                       validate_config)
from rage.experiments.ranking import (DEFAULT_TEMPERATURE, rank_mass,
                                      uniformize_intra_trajectory, UNIFORMIZATION_REVISION)
from rage.experiments.reference_opinions import (UNI_ARMS, LEGACY_UNI_ARMS, BLOCK_UNI_ARMS, WIN_UNI_ARMS, is_uni_arm,
                                                 selection_options, weight_placement_for)
from rage.experiments.resampling import ResampledWindows
from rage.crb import CRBConfig


def _selection(probabilities, counts, trajectory_ids, scores=None,
               placement='before_transform', aggregation='linearSum'):
    counts = list(counts)
    probabilities = list(probabilities)
    values = np.arange(8, dtype=np.float32)
    rows = int(sum(counts))
    window_locations = np.column_stack((np.zeros(rows, dtype=np.int64),
                                        np.arange(rows, dtype=np.int64)))
    metadata = dict(competition='global', source_weighting='legal_window_counts',
        temperature=float(DEFAULT_TEMPERATURE),
        probability_mapping='object-rank-exponential-interval-v5',
        scoring_revision='full-pool-rank-volume-v5',
        opinion_aggregation=aggregation, opinion_weight_placement=placement,
        cosine_numerics=full_pool.COSINE_NUMERICS, pool_manifest=dict(total_equivalents=rows),
        model=dict(model_id='fake'), crb={}, native_scores=list(probabilities if scores is None else scores))
    selected = ResampledWindows(values, window_locations, np.arange(rows),
        np.cumsum([0] + counts), np.arange(rows), probabilities,
        window_length=2, pool_n=rows, metadata=metadata)
    selected.native_scores = np.asarray(scores if scores is not None else probabilities)
    return selected, np.asarray(trajectory_ids, dtype=np.int64)


def test_uniformization_sums_blocks_and_erases_intra_trajectory_differences():
    selected, ids = _selection([.5, .3, .1, .1], [1, 3, 2, 4], [7, 7, 9, 9])
    uniform = uniformize_intra_trajectory(selected, ids)
    # Trajectory totals are preserved exactly: traj 7 -> .8, traj 9 -> .2.
    assert uniform.probabilities[0] + uniform.probabilities[1] == pytest.approx(.8)
    assert uniform.probabilities[2] + uniform.probabilities[3] == pytest.approx(.2)
    # Interior block differences are erased: equal share per block in a trajectory.
    assert uniform.probabilities[0] == pytest.approx(.4)
    assert uniform.probabilities[1] == pytest.approx(.4)
    assert uniform.probabilities[2] == pytest.approx(.1)
    assert uniform.probabilities[3] == pytest.approx(.1)
    assert uniform.probabilities.sum() == pytest.approx(1.)
    # Audit and identity.
    audit = uniform.metadata['uniformization_audit']
    assert audit['revision'] == UNIFORMIZATION_REVISION
    assert audit['trajectory_count'] == 2
    assert audit['blocks_per_trajectory_min'] == 2
    assert audit['total_variation_distance'] == pytest.approx(
        .5 * np.abs(uniform.probabilities - selected.probabilities).sum())
    assert uniform.metadata['intra_trajectory_uniformization'] == UNIFORMIZATION_REVISION
    assert uniform.metadata['pre_uniformization_probabilities'] == [.5, .3, .1, .1]
    # Parent is untouched and windows remain sampleable and identical.
    np.testing.assert_allclose(selected.probabilities, [.5, .3, .1, .1])
    np.testing.assert_array_equal(uniform.unit_offsets, selected.unit_offsets)
    np.testing.assert_array_equal(uniform.unit_positions, selected.unit_positions)


def test_uniformization_preserves_per_trajectory_mass_under_volume_prior():
    scores = np.array([3., 1., 2., 0.])
    counts = np.array([1, 3, 2, 4])
    _, block_probabilities, _ = rank_mass(scores, counts)
    selected, ids = _selection(block_probabilities, counts, [5, 5, 8, 8], scores=scores)
    uniform = uniformize_intra_trajectory(selected, ids)
    totals = {}
    for probability, trajectory in zip(uniform.probabilities, ids):
        totals[trajectory] = totals.get(trajectory, 0.) + probability
    base_totals = {}
    for probability, trajectory in zip(selected.probabilities, ids):
        base_totals[trajectory] = base_totals.get(trajectory, 0.) + probability
    assert totals == pytest.approx(base_totals)
    # Uniformized per-window exposure no longer follows legal-window counts
    # inside a trajectory: every block of a trajectory gets the same unit mass.
    assert uniform.probabilities[0] == pytest.approx(uniform.probabilities[1])


def test_uniformization_single_block_trajectories_are_unchanged():
    selected, ids = _selection([.6, .4], [2, 5], [3, 4])
    uniform = uniformize_intra_trajectory(selected, ids)
    np.testing.assert_allclose(uniform.probabilities, selected.probabilities)
    assert uniform.metadata['uniformization_audit']['total_variation_distance'] == 0.


def test_uniformization_roundtrip_through_payload():
    selected, ids = _selection([.5, .3, .1, .1], [1, 3, 2, 4], [7, 7, 9, 9])
    uniform = uniformize_intra_trajectory(selected, ids)
    restored = ResampledWindows.from_payload(uniform.payload())
    np.testing.assert_allclose(restored.probabilities, uniform.probabilities)
    assert restored.metadata['intra_trajectory_uniformization'] == UNIFORMIZATION_REVISION


@pytest.mark.parametrize('ids', [
    [1], [1, -2], [1, np.nan], np.zeros((4, 1), dtype=np.int64),
])
def test_uniformization_rejects_invalid_trajectory_ids(ids):
    selected, _ = _selection([.5, .3, .1, .1], [1, 3, 2, 4], [7, 7, 9, 9])
    with pytest.raises(ValueError):
        uniformize_intra_trajectory(selected, np.asarray(ids))


def test_uni_arms_registered_with_base_aggregation_and_placement():
    assert UNI_ARMS == LEGACY_UNI_ARMS + BLOCK_UNI_ARMS + WIN_UNI_ARMS
    for arm in UNI_ARMS:
        assert arm in ARMS
        base_mode = arm[len('rage-block-'):].split('-', 1)[0]
        assert selection_options(arm) == dict(aggregation=base_mode, competition='global')
        assert weight_placement_for(arm) == ('after_transform' if base_mode == 'expSum'
                                             else 'before_transform')
        assert is_uni_arm(arm)
        protocols = method_protocols([arm])[arm]
        assert protocols['intra_trajectory_uniformization'] == ('uniform_trajectory_window' if arm in WIN_UNI_ARMS else 'uniform_block_share')
        assert protocols['aggregation'] == base_mode
    assert not is_uni_arm('rage-block-expSum')
    # Base arms gain no uni identity.
    assert 'intra_trajectory_uniformization' not in method_protocols(['rage-block-expSum'])['rage-block-expSum']


def test_rage_uni_dispatch_groups_blocks_by_trajectory():
    rng = np.random.default_rng(4)
    dim = 13
    reference_blocks = [SimpleNamespace(fragment_id=1, start_positions=[0, 1],
        direction=rng.normal(size=dim)), SimpleNamespace(fragment_id=2, start_positions=[0],
        direction=rng.normal(size=dim))]
    source_blocks = [SimpleNamespace(start_positions=[0], direction=rng.normal(size=dim)),
                     SimpleNamespace(start_positions=[1, 2], direction=rng.normal(size=dim)),
                     SimpleNamespace(start_positions=[3], direction=rng.normal(size=dim))]
    for rows in ([b.direction for b in reference_blocks]
                 + [b.direction for b in source_blocks]):
        rows /= np.linalg.norm(rows)
    assets = SimpleNamespace(reference_blocks=reference_blocks,
        reference_results=[SimpleNamespace(fragment_id=1, valid_starts=[0, 1]),
                           SimpleNamespace(fragment_id=2, valid_starts=[0])],
        source_blocks=source_blocks, reference_trajectories=[], source_trajectories=[])
    index = SimpleNamespace(fragment_ids=np.array([10, 10, 10, 20]))
    metadata = dict(model=dict(model_id='fake'), temperature=float(DEFAULT_TEMPERATURE),
        crb={}, cosine_numerics=full_pool.COSINE_NUMERICS, task='t', scoring_revision='rev',
        opinion_aggregation='linearSum', competition='global',
        opinion_weight_placement='before_transform', additional_bp_allowed=False)
    original = full_pool.ranked_selection

    def fake_ranked_selection(index, units, scores, *, pool_manifest, metadata, temperature):
        selected, ids_unused = _selection([.6, .3, .1], [len(u) for u in units], [0, 0, 0])
        return selected

    full_pool.ranked_selection = fake_ranked_selection
    try:
        output = full_pool._rage('rage-block-linearSum-uni', index, dict(total_equivalents=4),
                                 assets, metadata, DEFAULT_TEMPERATURE, trajectory=False,
                                 uniformize="block")
    finally:
        full_pool.ranked_selection = original
    selected = output.selected
    # Source blocks 0 and 1 belong to trajectory 10, block 2 to trajectory 20.
    assert selected.metadata['intra_trajectory_uniformization'] == UNIFORMIZATION_REVISION
    assert selected.probabilities[0] == pytest.approx(.45)
    assert selected.probabilities[1] == pytest.approx(.45)
    assert selected.probabilities[2] == pytest.approx(.1)
    with pytest.raises(ValueError, match='block representation only'):
        full_pool._rage('rage-traj-linearSum', index, dict(total_equivalents=4), assets,
                        metadata, DEFAULT_TEMPERATURE, trajectory=True, uniformize="block")


def test_validate_cached_selection_distinguishes_uni_identity():
    from dataclasses import asdict
    selected, _ = _selection([.5, .5], [1, 3], [7, 9])
    crb = CRBConfig(context_length=512, prediction_length=64)
    manifest = dict(total_equivalents=4)
    identity = dict(model_id='fake')
    base_metadata = dict(selected.metadata)
    base_metadata['pool_manifest'] = manifest
    base_metadata['model'] = identity
    base_metadata['scoring_revision'] = 'full-pool-rank-volume-v5'
    base_metadata['probability_mapping'] = 'object-rank-exponential-interval-v5'
    base_metadata['crb'] = asdict(crb)
    from rage.experiments.ranking import frozen_rank_protocol
    base_metadata['frozen_rank_protocol'] = frozen_rank_protocol()
    validate_cached_selection(base_metadata, arm='rage-block-linearSum', identity=identity,
        manifest=manifest, crb=crb, temperature=DEFAULT_TEMPERATURE)
    with pytest.raises(ValueError, match='uni method identity'):
        validate_cached_selection(base_metadata, arm='rage-block-linearSum-uni', identity=identity,
            manifest=manifest, crb=crb, temperature=DEFAULT_TEMPERATURE)
    uni = uniformize_intra_trajectory(selected, np.array([7, 9]))
    uni_metadata = dict(uni.metadata, pool_manifest=manifest, model=identity,
        crb=asdict(crb), frozen_rank_protocol=frozen_rank_protocol())
    validate_cached_selection(uni_metadata, arm='rage-block-linearSum-uni', identity=identity,
        manifest=manifest, crb=crb, temperature=DEFAULT_TEMPERATURE)
    with pytest.raises(ValueError, match='uni method identity'):
        validate_cached_selection(uni_metadata, arm='rage-block-linearSum', identity=identity,
            manifest=manifest, crb=crb, temperature=DEFAULT_TEMPERATURE)


def test_v5_uni_config_validates_and_v5_frozen_matrix_unchanged():
    from rage.experiments.ranking import DEFAULT_TEMPERATURE
    config = json.loads(Path('configs/rage/main_v5_uni.json').read_text())
    validate_config(config)
    assert config['experiment_version'] == 'v5-uni'
    assert tuple(config['arms']) == V5_UNI_ARMS
    # The approved V5 formal matrix is not extended by the uni experiment.
    assert V5_FROZEN_ARMS == tuple(a for a in ARMS if not a.endswith('-uni'))
    v5 = json.loads(Path('configs/rage/main.json').read_text())
    validate_config(v5)

    def with_uni_temperatures(source):
        result = json.loads(json.dumps(source))
        for settings in result['settings'].values():
            for arm in V5_UNI_ARMS:
                if arm.startswith('rage-'):
                    settings['temperature'][arm] = DEFAULT_TEMPERATURE
        return result

    with pytest.raises(ValueError, match='frozen matrix differs: arms'):
        validate_config(with_uni_temperatures(dict(v5, arms=list(V5_UNI_ARMS),
            method_protocols=method_protocols(list(V5_UNI_ARMS)))))
    with pytest.raises(ValueError, match='frozen matrix differs: arms'):
        validate_config(with_uni_temperatures(dict(config, experiment_version='v5',
            method_protocols=method_protocols(list(V5_UNI_ARMS)))))
