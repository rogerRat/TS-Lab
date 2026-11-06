import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from rage.core import TimeSeriesFragment
from rage.experiments.development_inputs import save_fragments, load_fragments
from rage.experiments.full_pool_exploration import run_exploration
from rage.experiments.optimization import learning_rate_factor, optimizer_options
from rage.experiments.prepare_probe_inputs import prepare
from rage.experiments.train_eval import fine_tune


def engine():
    torch.manual_seed(8)
    model = torch.nn.Linear(4, 2)
    return SimpleNamespace(model=model, device="cpu", training_loss=lambda w:
        (model(torch.tensor(w[:, :4])) - torch.tensor(w[:, 4:])).square().mean())


def test_microbatch_accumulation_matches_effective_batch():
    windows = np.arange(120, dtype=np.float32).reshape(20, 6) / 100
    a, b = engine(), engine()
    kwargs = dict(steps=5, batch_size=7, learning_rate=.001, seed=4)
    fine_tune(a, windows, **kwargs)
    fine_tune(b, windows, **kwargs, training_options={"micro_batch_size": 2})
    for left, right in zip(a.model.parameters(), b.model.parameters()):
        torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-6)


def test_bias_and_norm_can_be_excluded_from_weight_decay(tmp_path):
    tiny = engine()
    path = tmp_path / "resume.pt"
    fine_tune(tiny, np.ones((4, 6), dtype=np.float32), steps=1, batch_size=2,
              learning_rate=.001, seed=0, weight_decay=.1,
              training_options={"exclude_bias_norm": True}, resume_path=path)
    groups = torch.load(path, weights_only=False)["optimizer"]["param_groups"]
    assert [group["weight_decay"] for group in groups] == [.1, 0.]
    assert all(len(group["params"]) == 1 for group in groups)


def test_custom_optimizer_and_schedule_resume_exactly(tmp_path):
    windows = np.arange(120, dtype=np.float32).reshape(20, 6) / 100
    options = dict(betas=[.9, .95], eps=1e-7, max_grad_norm=.1, micro_batch_size=2,
                   schedule="cosine", min_lr_ratio=.5, warmup_ratio=.25, schedule_steps=8)
    kwargs = dict(steps=8, batch_size=4, learning_rate=.001, seed=4,
                  weight_decay=.1, training_options=options)
    a, b = engine(), engine()
    fine_tune(a, windows, **kwargs)
    path = tmp_path / "resume.pt"
    def interrupt(step):
        if step == 5:
            raise RuntimeError("interrupted")
    with pytest.raises(RuntimeError, match="interrupted"):
        fine_tune(b, windows, **kwargs, resume_path=path, checkpoint_steps=[4, 8], on_step=interrupt)
    fine_tune(b, windows, **kwargs, resume_path=path)
    for left, right in zip(a.model.parameters(), b.model.parameters()):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    state = torch.load(path, weights_only=False)
    group = state["optimizer"]["param_groups"][0]
    assert group["betas"] == (.9, .95) and group["eps"] == 1e-7 and group["weight_decay"] == .1
    assert group["lr"] == pytest.approx(.001 * learning_rate_factor(7, optimizer_options(options, batch_size=4, steps=8)))
    with pytest.raises(ValueError, match="contract"):
        fine_tune(engine(), windows, **dict(kwargs, training_options={**options, "max_grad_norm": 1.}), resume_path=path)


def test_fractional_epoch_runtime_excludes_validation_and_exam(tmp_path, monkeypatch):
    from rage.experiments import full_pool_exploration as module
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    save_fragments(inputs / "fixture__reference.npz", [TimeSeriesFragment(
        np.arange(100, dtype=np.float32), entity_id="fixture", fragment_id=0)])
    (inputs / "fixture__manifest.json").write_text(json.dumps({"seasonality": 2}))
    (inputs / "fixture__exam.npz").write_text("POISON: final exam must never be opened")
    seen = []
    tiny = engine()
    original_loss = tiny.training_loss
    def loss(w):
        seen.append(w.copy())
        return original_loss(w)
    tiny.training_loss = loss
    monkeypatch.setattr(module, "evaluate", lambda model, **kw: {
        "mase": float(model.model.weight.detach().square().mean())})
    config = dict(student="chronos-base", inputs=str(inputs), datasets=["fixture"],
        training_data="safe_history_random", learning_rates=[.001], epochs=[1., 1.25, 1.3125, 2.],
        seeds=[0], batch_size=2, context_length=4, prediction_length=2, validation_windows=3,
        optimizer={"schedule": "linear", "micro_batch_size": 1}, schedule_epochs=2)
    report = run_exploration(config, tmp_path / "out", model_bundle=(tiny, tiny, {"model_id": "fixture"}))
    assert [p["epoch"] for p in report["trials"][0]["curve"]] == config["epochs"]
    assert np.max(np.concatenate(seen)) < 90
    assert not report["formal_exam_used"]
    state = torch.load(tmp_path / "out/lr-0.001-seed-0/resume.pt", weights_only=False)
    assert state["contract"]["training_options"]["schedule_steps"] == 15


def test_prepare_quito_only_exports_pre_cutoff_and_native_seasonality(tmp_path):
    from evaluate.quito_data import CUTOFF, REPO_ID, REVISION, TARGETS, data_file
    root = tmp_path / "data"
    path = data_file(root, "min")
    path.parent.mkdir(parents=True)
    (root / "data_manifest.json").write_text(json.dumps(dict(repo_id=REPO_ID, revision=REVISION)))
    dates = pd.date_range(CUTOFF - pd.Timedelta(minutes=600), periods=80, freq="10min")
    values = np.concatenate([np.arange(60.), np.full(20, 1e9)])
    pd.DataFrame(dict(item_id=1, date_time=dates, **{target: values for target in TARGETS})).to_parquet(path)
    config = dict(context_length=4, prediction_length=2, validation_windows=3,
                  tasks=[dict(id="quito_min", benchmark="quito", subset="min", root=str(root))])
    out = tmp_path / "prepared"
    prepare(config, out)
    for fragment in load_fragments(out / "quito_min__reference.npz"):
        np.testing.assert_array_equal(fragment.data, np.arange(60))
    manifest = json.loads((out / "quito_min__manifest.json").read_text())
    assert manifest["seasonality"] == 144 and manifest["formal_exam_excluded"]
    assert not list(out.glob("*exam*"))


def test_prepare_time_reserves_full_native_exam_tail(tmp_path, monkeypatch):
    import sys
    source = tmp_path / "eval.yaml"
    source.write_text("bench_config: {}")
    monkeypatch.setattr("evaluate.time._validate_bench_config", lambda _: dict(
        tasks=[{"dataset": "fixture/h"}], datasets={"fixture/h": {"test_length": 7}},
        dataset_root=tmp_path, dataset_values={"revision": "test"}))
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_from_disk=lambda _: [
        {"item_id": "x", "target": np.arange(47.)}]))
    out = tmp_path / "prepared"
    prepare(dict(context_length=4, prediction_length=2, validation_windows=3, tasks=[dict(
        id="time_fixture", benchmark="time", dataset="fixture/h", seasonality=24,
        evaluation_config=str(source))]), out)
    fragments = load_fragments(out / "time_fixture__reference.npz")
    np.testing.assert_array_equal(fragments[0].data, np.arange(40))


def test_prepare_fev_scalar_arrow_reserves_declared_exam(tmp_path):
    import datasets
    import yaml
    source = tmp_path / "raw.arrow"
    ds = datasets.Dataset.from_dict({"id": ["x"], "target": [list(range(60))]})
    ds.save_to_disk(str(tmp_path / "raw"))
    source = next((tmp_path / "raw").glob("*.arrow"))
    declaration = dict(dataset_config="fixture", horizon=4, num_windows=5, seasonality=7)
    protocol = tmp_path / "tasks.yaml"
    protocol.write_text(yaml.safe_dump(dict(tasks=[declaration])))
    task = dict(id="fev_scalar", benchmark="fev", dataset_config="fixture",
                dataset_file=str(source), evaluation_config=str(protocol), revision="fixture-v1")
    config = dict(context_length=4, prediction_length=2, validation_windows=3, tasks=[task])
    prepare(config, tmp_path / "prepared")
    rows = load_fragments(tmp_path / "prepared/fev_scalar__reference.npz")
    np.testing.assert_array_equal(rows[0].data, np.arange(40))
    manifest = json.loads((tmp_path / "prepared/fev_scalar__manifest.json").read_text())
    assert manifest["excluded_tail"] == 20
    assert manifest["seasonality"] == 7
    assert manifest["source_target_dims"] == [1]
    assert manifest["formal_exam_excluded"] is True
    declaration["target"] = ["target"]
    protocol.write_text(yaml.safe_dump(dict(tasks=[declaration])))
    with pytest.raises(ValueError, match="scalar target"):
        prepare(config, tmp_path / "invalid_multivariate")
    declaration.pop("target")
    declaration["initial_cutoff"] = -20
    protocol.write_text(yaml.safe_dump(dict(tasks=[declaration])))
    with pytest.raises(ValueError, match="cutoff/stride"):
        prepare(config, tmp_path / "invalid_cutoff")


def test_prepare_fev_requires_existing_reference_manifest(tmp_path):
    source = tmp_path / "reference.npz"
    save_fragments(source, [TimeSeriesFragment(np.arange(40.), entity_id="x", fragment_id=0)])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"source_training_only": True}))
    task = dict(id="fev_fixture", benchmark="fev", reference_file=str(source),
                manifest_file=str(manifest), seasonality=24)
    config = dict(context_length=4, prediction_length=2, validation_windows=3, tasks=[task])
    prepare(config, tmp_path / "prepared")
    np.testing.assert_array_equal(load_fragments(tmp_path / "prepared/fev_fixture__reference.npz")[0].data, np.arange(40))
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="provenance"):
        prepare(config, tmp_path / "invalid")


def test_time_named_subset_and_native_univariate_guard(tmp_path, monkeypatch):
    import sys
    source = tmp_path / "eval.yaml"
    source.write_text("bench_config: {}")
    monkeypatch.setattr("evaluate.time._validate_bench_config", lambda _: dict(
        tasks=[{"dataset": "fixture/h"}], datasets={"fixture/h": {"test_length": 7}},
        dataset_root=tmp_path, dataset_values={"revision": "test"}))
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_from_disk=lambda _: [
        {"item_id": "x", "target": np.stack([np.arange(47.), np.arange(47.) + 100]),
         "variate_names": ["demand", "price"]}]))
    task = dict(id="price", benchmark="time", dataset="fixture/h", seasonality=24,
                evaluation_config=str(source), variate_names=["price"])
    settings = dict(context_length=4, prediction_length=2, validation_windows=3, tasks=[task])
    prepare(settings, tmp_path / "selected")
    rows = load_fragments(tmp_path / "selected/price__reference.npz")
    assert len(rows) == 1 and rows[0].entity_id == "x/1"
    np.testing.assert_array_equal(rows[0].data, np.arange(40.) + 100)
    manifest = json.loads((tmp_path / "selected/price__manifest.json").read_text())
    assert manifest["source_target_dims"] == [2] and manifest["selected_variate_names"] == ["price"]
    task["variate_name_aliases"] = {"price": ["price[sp,rw]"]}
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_from_disk=lambda _: [
        {"item_id": "annotated", "target": np.stack([np.arange(47.), np.arange(47.) + 200]),
         "variate_names": ["demand", "price[sp,rw]"]}]))
    prepare(settings, tmp_path / "aliased")
    np.testing.assert_array_equal(load_fragments(tmp_path / "aliased/price__reference.npz")[0].data,
                                  np.arange(40.) + 200)
    manifest = json.loads((tmp_path / "aliased/price__manifest.json").read_text())
    assert manifest["resolved_alias_series"] == [dict(entity="annotated", canonical="price", actual="price[sp,rw]")]
    task["require_native_univariate"] = True
    with pytest.raises(ValueError, match="native univariate"):
        prepare(settings, tmp_path / "not_native")
    task["require_native_univariate"] = False
    task.pop("variate_name_aliases")
    task["variate_names"] = ["unknown"]
    with pytest.raises(ValueError, match="absent"):
        prepare(settings, tmp_path / "unknown")
