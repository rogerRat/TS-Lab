"""Dependency-free contracts for foundation-model integration."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from adapters.foundation import ModelCapability, build_adapter, get_model_spec, list_model_specs
from adapters.foundation.components import (
    grouped_mean_sketches,
    median_forecast,
    shifted_windows,
    split_static_windows,
)
from adapters.foundation.timesfm import _load_dtype


def test_registry_has_latest_and_trainable_timesfm_plus_sundial() -> None:
    keys = {spec.key for spec in list_model_specs()}
    assert keys == {"timesfm-2.5", "timesfm-3.0", "sundial-base-128m", "timemoe-200m", "timerxl-84m"}
    assert get_model_spec("timesfm-3.0").supports(ModelCapability.FORECAST)
    assert not get_model_spec("timesfm-3.0").supports(ModelCapability.TRAINING_LOSS)
    assert get_model_spec("timesfm-2.5").supports(ModelCapability.FINE_TUNE)
    assert not get_model_spec("sundial-base-128m").official_training_recipe


def test_unavailable_training_capability_fails_before_model_load() -> None:
    for key in ("timesfm-3.0",):
        adapter = build_adapter(key, device="cpu")
        with pytest.raises(NotImplementedError, match="training_loss"):
            adapter.training_loss(np.zeros((1, 8), dtype=np.float32), context_length=4, prediction_length=4)


def test_sundial_training_fallback_is_distinct_from_official_recipe() -> None:
    spec = get_model_spec("sundial-base-128m")
    assert spec.supports(ModelCapability.TRAINING_LOSS)
    assert not spec.official_training_recipe


def test_static_window_split_preserves_values() -> None:
    windows = np.arange(24, dtype=np.float32).reshape(3, 8)
    context, future = split_static_windows(windows, context_length=5, prediction_length=3)
    np.testing.assert_array_equal(context, windows[:, :5])
    np.testing.assert_array_equal(future, windows[:, 5:])


def test_median_forecast_reduces_only_sample_axis() -> None:
    samples = np.asarray([[[1, 2, 3], [3, 4, 5], [2, 3, 4]]], dtype=np.float32)
    np.testing.assert_array_equal(
        median_forecast(samples, prediction_length=2),
        np.asarray([[2, 3]], dtype=np.float32),
    )


def test_bottom_and_middle_layers_do_not_import_cli() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "adapters" / "foundation"
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(name == "scripts" or name.startswith("scripts.") for name in imports), path


def test_timesfm_dtype_requires_native_bf16_architecture() -> None:
    class FakeCuda:
        @staticmethod
        def is_available(): return True
        @staticmethod
        def get_device_capability(_index): return (7, 5)

    class FakeTorch:
        cuda = FakeCuda()
        bfloat16 = "bf16"
        float32 = "fp32"

    assert _load_dtype(FakeTorch, "cuda") == "bf16"
    FakeCuda.get_device_capability = staticmethod(lambda _index: (8, 0))
    assert _load_dtype(FakeTorch, "cuda:0") == "bf16"


def test_shifted_windows_keep_explicit_prediction_events() -> None:
    series = np.arange(30, dtype=np.float32)
    windows = shifted_windows(
        series, np.asarray([8, 12]), context_length=6, prediction_length=3,
    )
    np.testing.assert_array_equal(windows[0], series[2:11])
    np.testing.assert_array_equal(windows[1], series[6:15])


def test_grouped_mean_sketches_preserve_first_seen_group_order() -> None:
    groups, means = grouped_mean_sketches(
        np.asarray([[1, 3], [3, 5], [10, 20]], dtype=np.float32),
        np.asarray([7, 7, 2]),
    )
    np.testing.assert_array_equal(groups, [7, 2])
    np.testing.assert_array_equal(means, [[2, 4], [10, 20]])
