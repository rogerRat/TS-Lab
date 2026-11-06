import numpy as np
import pytest

from rage.core import TimeSeriesFragment
from rage.crb import CRBConfig, CRBEngine, CRBPipeline, match_blocks
from rage.experiments.crb_main import _evaluation_windows


def fragment_with_positions(count: int, *, context: int = 2, horizon: int = 1,
                            stride: int = 1, fragment_id: int = 7) -> TimeSeriesFragment:
    length = context + horizon + (count - 1) * stride
    return TimeSeriesFragment(np.arange(length, dtype=np.float32), fragment_id=fragment_id)


class Field:
    def __init__(self, function):
        self.function = function
        self.calls = []

    def __call__(self, fragment, physical_starts):
        physical_starts = np.asarray(physical_starts)
        self.calls.extend((fragment.fragment_id, int(value)) for value in physical_starts)
        return np.stack([np.asarray(self.function(int(value)), dtype=np.float32)
                         for value in physical_starts])


@pytest.mark.parametrize("scale", [1., 1e-17, 1e-30])
def test_block_direction_is_cosine_unit_even_for_native_tiny_gradients(scale):
    fragment = fragment_with_positions(9)
    config = CRBConfig(context_length=2, prediction_length=1, cover_density=3/9,
        error_density=0, split_density=0, mean_epsilon=scale/10, endpoint_epsilon=scale/10)
    result = CRBEngine(config, Field(lambda position: np.array([3., 4.]) * scale)).process_fragment(fragment)
    assert result.blocks
    assert sum(len(block.start_positions) for block in result.blocks) == 9
    for block in result.blocks:
        np.testing.assert_allclose(block.direction, [.6, .8], rtol=1e-6)
        assert np.dot(block.direction, np.array([.6, .8])) == pytest.approx(1.)


def test_protocol_stride_and_strict_budget_accounting():
    fragment = fragment_with_positions(11, stride=3)
    assert fragment.valid_starts(2, 1, 3).tolist() == list(range(1, 32, 3))
    field = Field(lambda position: [position, 1.0])
    config = CRBConfig(
        context_length=2, prediction_length=1, window_stride=3,
        cover_density=4 / 11, error_density=3 / 11, split_density=0,
        mean_epsilon=0.0,
    )
    result = CRBEngine(config, field).process_fragment(fragment)
    assert result.cover_computations == 4
    assert result.error_computations == 3
    assert result.split_computations == 0
    assert result.computed_gradients == 7
    assert len(field.calls) == 7
    assert len(set(field.calls)) == 7


def test_curvature_risk_targets_nonlinear_region():
    fragment = fragment_with_positions(17)

    def gradient(position):
        value = float(position - 1)
        return [8.0 + (value - 8.0) ** 2 if value >= 8 else value, 1.0]

    field = Field(gradient)
    config = CRBConfig(
        context_length=2, prediction_length=1,
        cover_density=5 / 17, error_density=4 / 17, split_density=0,
        mean_epsilon=0.0,
    )
    result = CRBEngine(config, field).process_fragment(fragment)
    # At least two error-stage anchors land in the curved right half.
    cover_positions = {1, 5, 9, 13, 17}
    extra = {physical for _fragment, physical in field.calls} - cover_positions
    assert sum(position >= 9 for position in extra) >= 2
    assert result.error_computations == 4


def test_cancellation_splits_keep_every_block():
    fragment = fragment_with_positions(9)
    vectors = {
        1: [1.0, 0.0],
        3: [-1.0, 0.0],
        5: [0.0, 0.0],
        7: [1.0, 0.0],
        9: [-1.0, 0.0],
    }
    # Coverage anchors are 1, 9, 5, 3, 7.  Three minimum intervals are
    # antagonistic; a one-item split budget processes only the first.
    field = Field(lambda position: vectors.get(position, [1.0, 0.0]))
    config = CRBConfig(
        context_length=2, prediction_length=1,
        cover_density=5 / 9, error_density=0, split_density=1 / 9,
        mean_epsilon=0.1, endpoint_epsilon=0.1,
    )
    result = CRBEngine(config, field).process_fragment(fragment)
    assert result.split_computations == 1
    assert result.banned_budget == result.discarded_antagonistic == 0
    assert len(result.blocks) == 5
    assert np.concatenate([b.start_positions for b in result.blocks]).tolist() == result.valid_starts.tolist()


def test_blocks_cover_positions_without_overlap_when_all_pass():
    fragment = fragment_with_positions(13, stride=2)
    field = Field(lambda position: [1.0, position / 100.0])
    result = CRBEngine(CRBConfig(
        context_length=2, prediction_length=1, window_stride=2,
        cover_density=5 / 13, error_density=2 / 13, split_density=0,
        mean_epsilon=1e-9,
    ), field).process_fragment(fragment)
    flattened = [int(position) for block in result.blocks for position in block.start_positions]
    assert flattened == result.valid_starts.tolist()
    assert len(flattened) == len(set(flattened))


def test_identical_reference_candidate_pipeline_ranks_matching_direction_first():
    references = [fragment_with_positions(9, fragment_id=1)]
    candidates = [fragment_with_positions(9, fragment_id=2)]

    def computer(fragment, starts):
        sign = 1.0 if fragment.fragment_id == 1 else 1.0
        return np.stack([[sign, float(start) / 100.0] for start in starts]).astype(np.float32)

    report = CRBPipeline(CRBConfig(
        context_length=2, prediction_length=1,
        cover_density=5 / 9, error_density=0, split_density=0,
        mean_epsilon=1e-9,
    ), computer).run(references, candidates, block_budget=2)
    assert report.reference_backward_count == 5
    assert report.candidate_backward_count == 5
    assert report.matches_by_reference
    assert all(len(row) == 2 for row in report.matches_by_reference)
    assert all(row[0].score >= row[1].score for row in report.matches_by_reference)


def test_match_budget_zero():
    assert match_blocks([], [], 0) == []


def test_experiment_evaluation_mode_defaults_to_latest_true_history_window():
    fragments = [fragment_with_positions(5, fragment_id=1), fragment_with_positions(5, fragment_id=2)]
    single = _evaluation_windows(fragments, 2, 1, 1, "single_window")
    rolling = _evaluation_windows(fragments, 2, 1, 1, "rolling")
    assert single.shape == (2, 3)
    assert rolling.shape == (10, 3)
    np.testing.assert_array_equal(single[0], rolling[4])
    np.testing.assert_array_equal(single[1], rolling[9])
