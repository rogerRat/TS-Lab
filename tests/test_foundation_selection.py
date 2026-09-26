"""Static deployment tests for W/S long-context valuation."""

from __future__ import annotations

import numpy as np
import pytest

from rags.experiments.foundation_selection import select_long_context_candidates


class _SketchEngine:
    def batch_gradient_sketch(self, windows):
        values = np.asarray(windows, dtype=np.float32)
        return np.zeros(len(values)), values[:, :2]


def _inputs():
    return {
        "source_windows": np.asarray([[1, 0, 3], [0, 1, 3], [1, 1, 3], [-1, 0, 3]], dtype=np.float32),
        "source_ids": np.asarray([[1, 0], [1, 10], [1, 20], [1, 30]]),
        "source_groups": np.asarray([10, 10, 20, 20]),
        "reference_windows": np.asarray([[1, 0, 2]], dtype=np.float32),
        "reference_groups": np.asarray([30]),
    }


def test_window_and_segment_change_only_scoring_not_static_identities() -> None:
    window = select_long_context_candidates(
        _SketchEngine(), **_inputs(), n_select=2, deployment_window_length=5,
        aggregation="window",
    )
    segment = select_long_context_candidates(
        _SketchEngine(), **_inputs(), n_select=2, deployment_window_length=5,
        aggregation="segment",
    )
    allowed = set(map(tuple, _inputs()["source_ids"]))
    assert set(map(tuple, window.selected_ids)).issubset(allowed)
    assert set(map(tuple, segment.selected_ids)).issubset(allowed)
    np.testing.assert_allclose(segment.event_scores[:2], segment.event_scores[0])
    assert not np.array_equal(window.event_scores, segment.event_scores)


def test_duplicate_static_identity_is_rejected() -> None:
    inputs = _inputs()
    inputs["source_ids"][1] = inputs["source_ids"][0]
    with pytest.raises(ValueError, match="unique"):
        select_long_context_candidates(
            _SketchEngine(), **inputs, n_select=1, aggregation="window",
        )
