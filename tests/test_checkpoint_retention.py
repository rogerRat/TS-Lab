import json

import pytest

from rags.experiments.checkpoint_retention import cleanup_completed_weights
from rags.experiments.students import atomic_json


def test_requires_result_and_only_removes_known_training_files(tmp_path):
    result = tmp_path / "result.json"
    names = ["epoch-01.pt", "epoch-0.500.pt", "resume.pt", "resume.pt.tmp"]
    for name in names + ["curve.json", "probability.npz", "model.json", "unrelated.pt"]:
        (tmp_path / name).touch()
    with pytest.raises(FileNotFoundError):
        cleanup_completed_weights(result, tmp_path)
    assert all((tmp_path / name).exists() for name in names)
    atomic_json(result, {"mase": 1.2, "curve": [{"epoch": .5}]})
    cleanup_completed_weights(result, tmp_path)
    assert all(not (tmp_path / name).exists() for name in names)
    assert all((tmp_path / name).exists() for name in ["curve.json", "probability.npz", "model.json", "unrelated.pt"])
    before = result.read_bytes()
    cleanup_completed_weights(result, tmp_path)
    assert result.read_bytes() == before
    assert json.loads(before)["mase"] == 1.2
