import json
import shutil
import numpy as np
import pytest

from test_full_pool_resampling import Probe, inputs
from rags.experiments.development_inputs import save_fragments
from rags.experiments.full_pool_worker import run_worker
from rags.experiments.resampling import REVISION
from rags.experiments.students import atomic_json


def fixture(tmp_path):
    data = inputs()
    for f in data["pool"] + data["reference_fragments"]:
        f.entity_id = f"fixture::{f.fragment_id}"
    folder = tmp_path / "inputs"
    folder.mkdir()
    save_fragments(folder / "pool.npz", data["pool"])
    save_fragments(folder / "tiny__reference.npz", data["reference_fragments"])
    atomic_json(folder / "pool_manifest.json", data["pool_manifest"])
    np.savez(folder / "tiny__exam.npz", windows=data["index"].windows_at(np.array([0, 1])),
             entities=np.array(["a", "b"]), scales=np.ones(2))
    identity = dict(model_id="tiny", revision="fixture", projection_dimension=3, projection_seed=42,
                    precision_policy="cpu-test", ckpt="fixture")
    config = dict(experiment_version="v5",purpose="development",resampling_revision=REVISION, students=["chronos-base"], datasets=["tiny"],
        arms=["rand"], seeds=[0, 1], context_length=4, prediction_length=2, train_batch=32,
        inputs=str(folder), crb={}, settings={"chronos-base": dict(
            epochs=.5, learning_rate=1e-7, temperature={})}, selection_execution={"cpu_threads": 2})
    from rags.experiments.ranking import frozen_rank_protocol
    from rags.experiments.protocol import method_protocols
    from rags.experiments.full_pool import ARMS
    from rags.experiments.ranking import DEFAULT_TEMPERATURE
    config["settings"]["chronos-base"]["temperature"]={a:DEFAULT_TEMPERATURE for a in ARMS if a.startswith("rags-") or a=="less-main"}
    config.update(rank_protocol=frozen_rank_protocol(),method_protocols=method_protocols(config["arms"]))
    atomic_json(tmp_path / "execution_control.json",dict(mode="run",selection_enabled=True,training_enabled=True,evaluation_enabled=True))
    atomic_json(tmp_path / "protocol.json", config)
    engine = Probe()
    return config, (engine, engine, identity)


def test_worker_production_path_and_completed_cell_resume(tmp_path):
    config, bundle = fixture(tmp_path)
    out = run_worker(tmp_path, "chronos-base", model_bundle=bundle)
    paths = sorted((out / "cells").glob("*.json"))
    assert len(paths) == 3
    before = [p.read_bytes() for p in paths]
    for path in paths:
        record = json.loads(path.read_text())
        if record["arm"] != "orig":
            assert record["checkpoints"][0]["epoch"] == .5
            assert record["train"]["source"]["actual_epochs"] >= .5
    run_worker(tmp_path, "chronos-base", model_bundle=bundle)
    assert [p.read_bytes() for p in paths] == before
    costs = [json.loads(line) for line in (out / "selection_metrics.jsonl").read_text().splitlines()]
    assert costs[-1]["cache_state"] == "selection_cache_hit"
    assert json.loads((out / "state.json").read_text())["status"] == "complete"


def test_worker_trains_task_before_preparing_the_next_task(tmp_path, monkeypatch):
    config, bundle = fixture(tmp_path)
    folder = tmp_path / "inputs"
    shutil.copyfile(folder / "tiny__reference.npz", folder / "later__reference.npz")
    shutil.copyfile(folder / "tiny__exam.npz", folder / "later__exam.npz")
    config["datasets"] = ["tiny", "later"]
    atomic_json(tmp_path / "protocol.json", config)
    events = []
    from rags.experiments import full_pool_worker as module
    select = module.select_full_pool
    train = module.run_arm_training

    def record_select(arm, probe, **kwargs):
        events.append(("selection", kwargs["dataset_tag"]))
        return select(arm, probe, **kwargs)

    def record_train(*args, checkpoint_dir, **kwargs):
        events.append(("training", checkpoint_dir.parts[-3]))
        return train(*args, checkpoint_dir=checkpoint_dir, **kwargs)

    monkeypatch.setattr(module, "select_full_pool", record_select)
    monkeypatch.setattr(module, "run_arm_training", record_train)
    run_worker(tmp_path, "chronos-base", model_bundle=bundle)
    assert events.index(("training", "tiny")) < events.index(("selection", "later"))


def test_worker_rejects_historical_protocol_before_model_loading(tmp_path):
    config,bundle=fixture(tmp_path)
    config['experiment_version']='v4'
    atomic_json(tmp_path/'protocol.json',config)
    with pytest.raises(ValueError,match='Only the frozen V5'):
        run_worker(tmp_path,'chronos-base',model_bundle=bundle)


def test_original_first_does_not_select_or_train(tmp_path, monkeypatch):
    config, bundle = fixture(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("Original-only stage must not select or train")
    monkeypatch.setattr("rags.experiments.full_pool_worker.select_full_pool", forbidden)
    monkeypatch.setattr("rags.experiments.full_pool_worker.run_arm_training", forbidden)
    out = run_worker(tmp_path, "chronos-base", model_bundle=bundle, original_only=True)
    assert len(list((out / "cells").glob("*.json"))) == 1
    assert json.loads((out / "state.json").read_text())["status"] == "original_complete"


@pytest.mark.parametrize("online", [False, True])
def test_completed_cleanup_preserves_results_and_resume_skips_training(tmp_path, monkeypatch, online):
    config, bundle = fixture(tmp_path)
    config["checkpoint_retention"] = "completed_cell_cleanup"
    config["online_evaluation"] = online
    atomic_json(tmp_path / "protocol.json", config)
    out = run_worker(tmp_path, "chronos-base", model_bundle=bundle)
    assert not list((out / "checkpoints").rglob("*.pt"))
    cells = list((out / "cells").glob("*__rand__*.json"))
    for cell in cells:
        record = json.loads(cell.read_text())
        assert record["training_weights_retained"] is False
        assert record["checkpoints"][0]["metrics"]
    assert list((out / "selections").glob("*.npz"))
    def must_not_train(*args, **kwargs):
        raise AssertionError("Completed cells must not retrain")
    monkeypatch.setattr("rags.experiments.full_pool_worker.run_arm_training", must_not_train)
    run_worker(tmp_path, "chronos-base", model_bundle=bundle)


def test_failed_cell_keeps_training_state(tmp_path, monkeypatch):
    config, bundle = fixture(tmp_path)
    config["checkpoint_retention"] = "completed_cell_cleanup"
    atomic_json(tmp_path / "protocol.json", config)
    def fail(*args, checkpoint_dir, **kwargs):
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "resume.pt").touch()
        raise RuntimeError("evaluation interrupted")
    monkeypatch.setattr("rags.experiments.full_pool_worker.run_arm_training", fail)
    with pytest.raises(RuntimeError, match="evaluation interrupted"):
        run_worker(tmp_path, "chronos-base", model_bundle=bundle)
    assert len(list(tmp_path.rglob("resume.pt"))) == 1
    assert not list(tmp_path.rglob("tiny__rand__seed0.json"))


def test_interruption_between_result_and_cleanup_does_not_retrain(tmp_path, monkeypatch):
    from rags.experiments import full_pool_worker as module
    config, bundle = fixture(tmp_path)
    config.update(checkpoint_retention="completed_cell_cleanup", seeds=[0])
    atomic_json(tmp_path / "protocol.json", config)
    cleanup = module.cleanup_completed_weights
    def interrupt_cleanup(*args):
        raise RuntimeError("shutdown before cleanup")
    monkeypatch.setattr(module, "cleanup_completed_weights", interrupt_cleanup)
    with pytest.raises(RuntimeError, match="shutdown before cleanup"):
        run_worker(tmp_path, "chronos-base", model_bundle=bundle)
    assert list(tmp_path.rglob("resume.pt"))
    cell = next(tmp_path.rglob("tiny__rand__seed0.json"))
    metrics = json.loads(cell.read_text())["checkpoints"]
    monkeypatch.setattr(module, "cleanup_completed_weights", cleanup)
    def must_not_train(*args, **kwargs):
        raise AssertionError("Durable results must not be retrained after shutdown")
    monkeypatch.setattr(module, "run_arm_training", must_not_train)
    run_worker(tmp_path, "chronos-base", model_bundle=bundle)
    assert not list(tmp_path.rglob("resume.pt"))
    assert json.loads(cell.read_text())["checkpoints"] == metrics
