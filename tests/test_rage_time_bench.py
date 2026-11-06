"""Tests for the TIME-benchmark task adapter (skipped when data is absent)."""

from pathlib import Path
import pytest

from rage.experiments.time_bench import (
    TIME_DATA_ROOT, TIME_DATASETS_YAML,
    build_time_task, build_time_task_specs, load_time_datasets,
    time_seasonality,
)


def test_datasets_yaml_loadable_and_pinned_layout():
    assert TIME_DATASETS_YAML.name == "datasets.yaml"
    datasets = load_time_datasets()
    assert datasets, "TIME datasets.yaml must not be empty"
    first = next(iter(datasets.values()))
    assert "test_length" in first


@pytest.mark.parametrize("suffix,expected", [
    ("15T", 96), ("1H", 24), ("5T", 288), ("D", 7), ("W", 52), ("", 24),
])
def test_seasonality_from_frequency_suffix(suffix, expected):
    assert time_seasonality("some_dataset/%s" % suffix, "") == expected


def test_task_specs_resolve_and_reject_unknown():
    specs = build_time_task_specs()
    assert specs, "at least one TIME task spec expected"
    with pytest.raises(KeyError):
        build_time_task_specs(["no_such_dataset/1H"])


@pytest.mark.skipif(not TIME_DATA_ROOT.exists(), reason="TIME data not fetched")
def test_build_time_task_shapes():
    specs = build_time_task_specs()
    spec = specs[0]
    task = build_time_task(spec, context_length=512, prediction_length=64,
                           series_cap=2)
    assert task.test_windows.ndim == 2
    assert task.test_windows.shape[1] == 512 + 64
    assert len(task.test_entities) == len(task.test_windows)
    assert len(task.test_scales) == len(task.test_windows)
    assert task.reference_fragments, "selection region must be non-empty"
    for fragment in task.reference_fragments:
        # exam span strictly excluded from the selection region
        assert len(fragment.data) + task.test_windows.shape[1] >= 576
