import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluate.quito import evaluate_task, run_quito, validate_config
from evaluate.quito_data import (CUTOFF, REPO_ID, REVISION, TARGETS,
                                data_file, iter_series, window_starts)
from evaluate.schemas import EvaluationRunConfig
from evaluate.time import _project_relative


class LastValueAdapter:
    @classmethod
    def parse_config(cls, config, base):
        return config

    @classmethod
    def from_config(cls, config, checkpoint):
        return cls()

    def predict_quantiles(self, inputs, prediction_length):
        return np.array([np.full((3, prediction_length), entry["target"][-1])
                         for entry in inputs])


@pytest.fixture
def snapshot(tmp_path):
    path = data_file(tmp_path, "hour")
    path.parent.mkdir()
    dates = pd.date_range(CUTOFF - pd.Timedelta(hours=48), periods=56, freq="h")
    pd.DataFrame({"item_id": 7, "date_time": dates,
                  **{target: np.arange(56, dtype=float) for target in TARGETS}}).to_parquet(path)
    (tmp_path / "data_manifest.json").write_text(json.dumps({"repo_id": REPO_ID, "revision": REVISION}))
    return tmp_path


def inference():
    return {"context_length": 8, "batch_size": 3,
            "quantile_levels": [0.1, 0.5, 0.9], "evaluation_mode": "rolling"}


def bench(root):
    return {"dataset": {"repo_id": REPO_ID, "revision": REVISION, "root": str(root)},
            "tasks": [{"subset": "hour", "prediction_length": 3}], "stride": 3}


def test_cutoff_and_incomplete_tail(snapshot):
    entity, values, boundary = next(iter_series(snapshot, "hour"))
    assert entity == "7/ind_1" and boundary == 48
    assert list(window_starts(len(values), boundary, 8, 3, "rolling", 3)) == [48, 51]
    assert window_starts(len(values), boundary, 8, 3, "single_window", 3) == [53]
    assert window_starts(len(values), boundary, 8, 9, "single_window", 9) == []


def test_missing_timestamp_preserves_time_axis(snapshot):
    path = data_file(snapshot, "hour")
    frame = pd.read_parquet(path).drop(index=50)
    frame.to_parquet(path)
    _, values, boundary = next(iter_series(snapshot, "hour"))
    assert boundary == 48 and len(values) == 56 and np.isnan(values[50])
    with pytest.raises(ValueError, match="no finite"):
        evaluate_task(LastValueAdapter(), snapshot, bench(snapshot)["tasks"][0], inference(), 3)


def test_metrics_and_counts(snapshot):
    result = evaluate_task(LastValueAdapter(), snapshot, bench(snapshot)["tasks"][0], inference(), 3)
    assert result["windows"] == 10
    assert result["candidate_windows"] == 10 and result["invalid_windows"] == 0
    assert result["MAE"] == pytest.approx(2)
    assert result["MSE"] == pytest.approx(14 / 3)
    assert result["MASE"] == pytest.approx(2 / 24)
    assert result["SQL"] == pytest.approx(2 / 24)
    assert result["WAPE"] == pytest.approx(result["WQL"])


def test_missing_values_counted_without_imputation(snapshot):
    path = data_file(snapshot, "hour")
    frame = pd.read_parquet(path)
    frame.loc[49, "ind_1"] = np.nan
    frame.to_parquet(path)
    result = evaluate_task(LastValueAdapter(), snapshot, bench(snapshot)["tasks"][0], inference(), 3)
    assert result["candidate_windows"] == 10
    assert result["invalid_windows"] == 2 and result["windows"] == 8


def test_duplicate_timestamps_rejected(snapshot):
    path = data_file(snapshot, "hour")
    frame = pd.read_parquet(path)
    pd.concat([frame, frame.iloc[:1]]).to_parquet(path)
    with pytest.raises(ValueError, match="duplicate"):
        list(iter_series(snapshot, "hour"))


def test_validation_and_revision(snapshot):
    config = bench(snapshot)
    assert validate_config(config)[0] == snapshot
    config["dataset"]["revision"] = "main"
    with pytest.raises(ValueError, match="pinned"):
        validate_config(config)


def test_run_resume_and_changed_protocol(snapshot, tmp_path):
    mapping = {"schema_version": 1, "benchmark": "quito", "model": {"id": "base"},
               "adapter": {"class": "test_evaluate_quito:LastValueAdapter",
                           "config": {"inference": inference()}}, "bench_config": bench(snapshot)}
    config = EvaluationRunConfig.from_mapping(mapping, tmp_path / "run.yaml")
    out = tmp_path / "out"
    out.mkdir()
    checkpoint = tmp_path / "checkpoint"
    result = run_quito(config, checkpoint, out, False)
    original = result.read_text()
    assert run_quito(config, checkpoint, out, True).read_text() == original
    assert (out / "summary.csv").is_file()
    mapping["bench_config"]["stride"] = 1
    changed = EvaluationRunConfig.from_mapping(mapping, tmp_path / "run.yaml")
    with pytest.raises(ValueError, match="resume"):
        run_quito(changed, checkpoint, out, True)


def test_time_accepts_external_disk_and_symlinks(tmp_path):
    actual = tmp_path / "external"
    actual.mkdir()
    link = tmp_path / "link"
    link.symlink_to(actual, target_is_directory=True)
    assert _project_relative(str(link), "dataset.root") == actual


def test_base_checkpoint_and_unified_dispatch(snapshot, tmp_path):
    from evaluate.run import run, validate_checkpoint
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").touch()
    assert validate_checkpoint(checkpoint, require_manifest=False) == checkpoint
    with pytest.raises(FileNotFoundError):
        validate_checkpoint(checkpoint)
    mapping = {"schema_version": 1, "benchmark": "quito", "model": {"id": "base"},
               "adapter": {"class": "test_evaluate_quito:LastValueAdapter",
                           "config": {"inference": inference()}}, "bench_config": bench(snapshot)}
    config = EvaluationRunConfig.from_mapping(mapping, tmp_path / "run.yaml")
    from configs.paths import PathPolicyWarning
    with pytest.warns(PathPolicyWarning):
        assert run(config, checkpoint, tmp_path / "unified").is_file()


def test_quito_reference_never_includes_test_values(snapshot):
    from rags.experiments.quito_bench import build_quito_task, build_quito_task_specs
    spec = build_quito_task_specs(["hour"], root=snapshot)[0]
    task = build_quito_task(spec, context_length=8, prediction_length=3)
    assert task.test_windows.shape == (10, 11)
    assert len(task.reference_fragments) == 5
    for fragment in task.reference_fragments:
        np.testing.assert_array_equal(fragment.data, np.arange(48))
    np.testing.assert_array_equal(task.test_windows[0, 8:], [48, 49, 50])
    np.testing.assert_array_equal(task.test_windows[1, 8:], [51, 52, 53])


def test_time_rolling_does_not_extend_before_test_boundary(tmp_path, monkeypatch):
    import sys
    import types
    from rags.experiments import time_bench
    from rags.experiments.tasks import TaskSpec
    (tmp_path / "fixture").mkdir()
    monkeypatch.setattr(time_bench, "TIME_DATA_ROOT", tmp_path)
    monkeypatch.setattr(time_bench, "load_time_datasets", lambda: {
        "fixture": {"test_length": 8, "short": {"prediction_length": 3}}})
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(
        load_from_disk=lambda path: [{"item_id": "x", "target": np.arange(56), "freq": "h"}]))
    task = time_bench.build_time_task(TaskSpec("fixture", tmp_path),
                                     context_length=8, prediction_length=3)
    assert task.test_windows.shape == (2, 11)
    np.testing.assert_array_equal(task.reference_fragments[0].data, np.arange(48))
    np.testing.assert_array_equal(task.test_windows[:, 8:], [[48, 49, 50], [51, 52, 53]])


def test_shipped_config_and_chronos_quantile_contract(tmp_path):
    import torch
    from adapters.chronos.eval.time import ChronosTIMEEvaluationAdapter
    source = Path(__file__).resolve().parents[1] / "configs/eval/quito/default.yaml"
    config = EvaluationRunConfig.from_file(source)
    parsed = ChronosTIMEEvaluationAdapter.parse_config(dict(config.adapter_config), source.parent)
    assert config.benchmark == "quito" and parsed["inference"]["evaluation_mode"] == "single_window"
    parsed["inference"]["batch_size"] = 2
    parsed["inference"]["context_length"] = 4
    calls = []

    class Pipeline:
        def predict_quantiles(self, batch, **kwargs):
            calls.append((batch, kwargs))
            return torch.ones(len(batch), kwargs["prediction_length"], 9), None

    adapter = ChronosTIMEEvaluationAdapter(tmp_path, parsed)
    adapter._pipeline = Pipeline()
    result = adapter.predict_quantiles([{"target": np.arange(8)} for _ in range(3)], 3)
    assert result.shape == (3, 9, 3) and len(calls) == 2
    assert all(len(context) == 4 for batch, _ in calls for context in batch)
    assert all(kwargs["num_samples"] == 20 for _, kwargs in calls)
