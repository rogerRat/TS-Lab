"""Validate the feasible reference gradients and close the preparation batch."""
from __future__ import annotations

import json
from pathlib import Path

from configs.logger import configure_logger
from configs.paths import PROJECT_ROOT, output_path

TASKS = ("fev-09", "time-01", "time-02", "time-03", "time-04", "time-06", "time-07", "time-08")
ABANDONED = ("fev-11", "fev-18", "fev-19", "time-05")
ROOT = PROJECT_ROOT


def main():
    logger = configure_logger("v1_crb_redo_finalize_ref", "v1-crb-redo",
                              "12tasks-ref-grad-prepare-finalize", "validation")
    out = output_path("train", ROOT / "outputs/train/v1-crb-redo/12tasks-ref-grad-prepare")
    inputs = ROOT / "data/ref/shard_dataset/v1-crb-redo-12tasks"
    records = []
    for task in TASKS:
        source, result = inputs / task, out / task
        if not (source / ".complete").is_file() or not (result / ".complete").is_file():
            raise FileNotFoundError("Reference task incomplete: " + task)
        data = json.loads((source / "manifest.json").read_text())
        gradient = json.loads((result / "manifest.json").read_text())
        n = data["fragments"]
        crb, uniform = sorted((result / "crb").glob("*.json")), sorted((result / "uniform").glob("*.json"))
        if data["answer_values_included"] or len(crb) != n or len(uniform) != n:
            raise ValueError("Reference safety or method manifests differ: " + task)
        a = sum(json.loads(path.read_text())["count"] for path in crb)
        b = sum(json.loads(path.read_text())["count"] for path in uniform)
        if a != b or a != gradient["anchors_per_method"] or gradient["legal_windows"] != data["legal_windows"]:
            raise ValueError("Reference budgets or windows differ: " + task)
        cache = Path(gradient["cache"])
        if not (cache / "cache_manifest.json").is_file() or not (cache / "window_index.npz").is_file():
            raise FileNotFoundError("Reference gradient cache incomplete: " + task)
        records.append(dict(task=task, fragments=n, legal_windows=data["legal_windows"],
                            anchors_per_method=a, cache=str(cache)))
        logger.info("stage=task-validated task=%s fragments=%d legal_windows=%d anchors_per_method=%d",
                    task, n, data["legal_windows"], a)
    for task in ABANDONED:
        if (out / task / ".complete").exists():
            raise ValueError("Abandoned task has a completion marker: " + task)
    manifest = dict(experiment="v1-crb-redo", batch="12tasks-ref-grad-prepare",
        eligible_tasks=records, abandoned_tasks=[dict(task=task,
            reason="safe_history_shorter_than_576_no_selection_support",
            future_use=False) for task in ABANDONED],
        methods_prepared=["crb_1_0_1", "matched_uniform"], required_side_symmetry=True)
    path = out / "manifest.json"
    temporary = out / "manifest.partial.json"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii")
    temporary.replace(path)
    (out / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=complete eligible=%d abandoned=%d output=%s", len(TASKS), len(ABANDONED), out)


if __name__ == "__main__":
    main()
