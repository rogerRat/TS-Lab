"""Actual peak measurement and append-only cold/warm/resume accounting."""
import json

import numpy as np
import pytest
import torch

from rags.experiments.arms import ArmOutput, SelectionCost
from rags.experiments.selection_metrics import SelectionMetrics


def output(notes=None):
    return ArmOutput("test", np.zeros((1, 4)), [(0, 1)],
                     SelectionCost(wall_seconds=123, notes=notes or {}))


def test_cold_warm_and_resume_preserve_original_cost(tmp_path):
    meter = SelectionMetrics(tmp_path, device="cpu", protocol={"n_select": 1})
    cold = meter.measure("task", "random", output)
    for key in ("pool_cached", "proxy_cached", "scores_cached"):
        meter.measure("task", key, lambda: output({key: True}))
    saved_cost = cold.cost.wall_seconds
    resumed = SelectionMetrics(tmp_path, device="cpu", protocol={"n_select": 1})
    assert resumed.measure("task", "random", lambda: cold, selection_cached=True) is cold
    assert cold.cost.wall_seconds == saved_cost
    resumed.original("task")
    rows = [json.loads(line) for line in meter.path.read_text().splitlines()]
    assert [r["cache_state"] for r in rows] == [
        "cold", "warm", "warm", "warm", "selection_cache_hit", "not_applicable"]
    assert rows[0]["invocation_id"] != rows[-1]["invocation_id"]
    assert rows[0]["selection_seconds"] == saved_cost
    assert all(r["peak_gpu_memory_gib"] == 0 for r in rows)
    assert rows[-1]["selection_seconds"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_peak_captures_freed_tensor_and_resets_between_arms(tmp_path):
    meter = SelectionMetrics(tmp_path, device="cuda:0", protocol={})
    resident = torch.zeros(1024, device="cuda:0")
    baseline = torch.cuda.memory_allocated(0)

    def compute():
        transient = torch.ones(4 * 1024 * 1024, device="cuda:0")
        transient.mul_(2)
        return output()  # transient freed before the wrapper reads the peak

    meter.measure("task", "large", compute)
    meter.measure("task", "small", output)
    rows = [json.loads(line) for line in meter.path.read_text().splitlines()]
    assert rows[0]["peak_gpu_memory_gib"] * 2**30 >= baseline + 16 * 1024**2
    assert rows[1]["peak_gpu_memory_gib"] < rows[0]["peak_gpu_memory_gib"]
    assert rows[0]["selection_seconds"] > 0
    assert rows[0]["gpu_name"] == torch.cuda.get_device_name(0)
    del resident
