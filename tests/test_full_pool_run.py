from pathlib import Path
import pytest
import yaml
import json

from rage.experiments.full_pool_run import validate_config, worker_command
from rage.experiments.run import parse_args, main


def formal_config():
    return json.loads(Path("configs/rage/main.json").read_text())


def test_default_entry_is_new_protocol_without_legacy_subset_budget(capsys):
    assert parse_args([]).config == "configs/rage/main.json"
    assert main(["--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "full_pool_worker" in output and "chronos-base" in output
    assert "n-select" not in output and "main_table" not in output


def test_formal_config_requires_frozen_v5_matrix():
    config=formal_config();validate_config(config)
    for key,value in (("pool_equivalents",16384),("students",["chronos-small"]),
            ("datasets",["boomlet_1676"]),("arms",["random"]),("seeds",[3])):
        with pytest.raises(ValueError):validate_config(dict(config,**{key:value}))


def test_explicit_environment_uses_native_worker():
    config = dict(student_environments={"timemoe-200m": "/tmp/env/bin/python"})
    command = worker_command("timemoe-200m", config, "/tmp/run")
    assert command[:3] == ["/tmp/env/bin/python", "-m", "rage.experiments.full_pool_worker"]


def test_new_formal_cleanup_policy_is_frozen_and_resumable(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from rage.experiments import full_pool_run as module
    config = dict(purpose="formal", students=["chronos-base"], experiment_version="v2", crb={"split_policy":"closest_zero_v2"})
    monkeypatch.setattr(module, "validate_config", lambda config: None)
    monkeypatch.setattr(module, "prepare_inputs", lambda *args: None)
    monkeypatch.setattr(module, "summarize", lambda *args: {"complete": True})
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: "fixture")
    assert module.run_suite(config, tmp_path) == 0
    assert json.loads((tmp_path / "protocol.json").read_text())["checkpoint_retention"] == "completed_cell_cleanup"
    assert module.run_suite(config, tmp_path) == 0


def test_execution_override_parallel_workers_preserve_protocol(tmp_path, monkeypatch):
    import json
    import threading
    from types import SimpleNamespace
    from rage.experiments import full_pool_run as module
    students = ["chronos-base", "timesfm-2.5", "timemoe-200m"]
    config = dict(purpose="test", students=students)
    barrier = threading.Barrier(3, timeout=5)
    def run(*args, **kwargs):
        barrier.wait()
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module, "validate_config", lambda config: None)
    monkeypatch.setattr(module, "prepare_inputs", lambda *args: None)
    monkeypatch.setattr(module, "summarize", lambda *args: {"complete": True})
    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: "fixture")
    execution = {"student_parallelism": 3, "student_selection_execution": {students[0]: {"gradient_batch": 8}}}
    assert module.run_suite(config, tmp_path, execution=execution) == 0
    assert json.loads((tmp_path / "execution.json").read_text()) == execution
    assert "student_parallelism" not in json.loads((tmp_path / "protocol.json").read_text())
    assert module.run_suite(config, tmp_path) == 0
