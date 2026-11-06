import json
from pathlib import Path

import numpy as np
import pytest

from rage.experiments.v7_extension import (
    V7_E_CANDIDATES, make_config, v7_e_native_arms, validate_extension,
)
from rage.experiments.v7_shape import scores_many
from rage.experiments.v7_shape_backfill import sidecar_config


def test_v7_e_two_shape_choice_freezes_eleven_reported_arms(tmp_path):
    base = json.loads(Path('configs/rage/main_v7_curve4.json').read_text())
    chosen = ('rage-shape-expSum', 'rage-shape-linearSum')
    config = make_config(base, str(tmp_path/'parent'), str(tmp_path/'run'),
                         V7_E_CANDIDATES[0], shape_arms=chosen)
    assert len(config['arms']) == 9
    assert config['arms'] == ['orig', 'rand', 'rage-block-expSum', 'rage-block-linearSum',
                              'rage-traj-expSum', 'rage-traj-linearSum',
                              'lp-mirror', 'tsds', 'less-main']
    validate_extension(config)
    sidecar = sidecar_config(config, tmp_path/'run'/'shape_backfill_20260925')
    assert sidecar['arms'] == ['orig', 'rand', *chosen]
    assert sidecar['shape_amendment']['methods'] == list(chosen)
    assert len(set(config['arms'] + list(chosen))) == 11
    with pytest.raises(ValueError, match='two distinct'):
        v7_e_native_arms((chosen[0], chosen[0]))


def test_v7_e_shape_subset_scores_are_exact_subset_of_full():
    source = np.asarray([[1., 0.], [0., 1.], [-1., 0.]], dtype=np.float32)
    reference = np.asarray([[1., 0.], [0., 1.]], dtype=np.float32)
    weights = np.asarray([.25, .75])
    full = scores_many(reference, source, weights)
    selected = scores_many(reference, source, weights,
                           aggregations=('expSum', 'linearSum'))
    assert set(selected) == {'expSum', 'linearSum'}
    for mode in selected:
        np.testing.assert_array_equal(selected[mode][0], full[mode][0])
