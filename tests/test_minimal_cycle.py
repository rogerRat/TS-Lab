from pathlib import Path

import json

import numpy as np
import torch

from assets.dataset_processing.load.bundle import BundleLoadWriter
from assets.dataset_processing.shard_dataset import ShardDatasetReader
from rage.experiments.students import run_student_task
from rage.static_selection import selection_size, select_round_robin, select_topk


class TinyForecastEngine:
    def __init__(self, context_length, prediction_length):
        self.device = "cpu"
        self.model = torch.nn.Linear(context_length, prediction_length)

    def training_loss(self, windows):
        values = torch.as_tensor(windows, dtype=torch.float32)
        return torch.nn.functional.mse_loss(
            self.model(values[:, :4]), values[:, 4:8],
        )

    def forecast(self, contexts):
        with torch.no_grad():
            values = torch.as_tensor(contexts, dtype=torch.float32)
            return self.model(values).numpy()


def test_preparation_selection_training_evaluation_cycle(tmp_path):
    bundle = tmp_path / "data" / "load-bundle"
    writer = BundleLoadWriter(bundle, scope="source", shard_bytes=4096)
    rows = []
    for group in range(3):
        values = np.sin(np.arange(20, dtype=np.float64) * 0.2 + group)
        rows.append({"dataset": "smoke", "subset": "minimal",
                     "series_id": "series-%d" % group, "values": values})
    writer.write_rows(rows)
    writer.finalize()

    reader = ShardDatasetReader(bundle)
    windows, groups, scores, candidate_ids = [], [], [], []
    for group in range(3):
        values = reader.read("source/smoke/minimal/series-%d" % group)
        for start in range(12):
            windows.append(values[start:start + 8])
            candidate_ids.append((group, start))
            groups.append(group)
            scores.append(float(abs(values[start + 7])))
    reader.close()
    windows = np.asarray(windows, dtype=np.float32)
    count = selection_size(len(windows), ratio=0.25)
    test_windows = windows[-2:]
    exam_path = tmp_path / "data" / "exam.npz"
    exam_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(exam_path, windows=test_windows,
                        entities=np.asarray(["a", "b"]), scales=np.ones(2))
    selection_paths = {}
    for method, indices in {
        "rr": select_round_robin(groups, count, scores),
        "topk": select_topk(scores, count),
    }.items():
        path = tmp_path / "data" / (method + ".npz")
        np.savez_compressed(path, windows=windows[indices], ids=np.asarray(candidate_ids)[indices])
        selection_paths[method] = {"path": str(path)}
    manifest = {
        "dataset": "smoke", "context_length": 4, "prediction_length": 4,
        "training": {"seeds": [7], "epochs": 1, "batch_size": 1,
                     "learning_rate": 1e-3},
        "exam": {"path": str(exam_path)}, "selections": selection_paths,
    }
    manifest_path = tmp_path / "task-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    results = run_student_task(
        "chronos-base", manifest_path, tmp_path / "outputs",
        engine=TinyForecastEngine(4, 4),
    )
    assert results == tmp_path / "outputs"
    cells = list((results / "cells").glob("*.json"))
    assert len(cells) == 3
    assert (results / "cells" / "smoke__original__seed0.json").is_file()
    for method in ("rr", "topk"):
        cell = json.loads((results / "cells" / ("smoke__%s__seed7.json" % method)).read_text())
        assert cell["train"]["source"]["steps"] == count
        assert np.isfinite(cell["mase"])
