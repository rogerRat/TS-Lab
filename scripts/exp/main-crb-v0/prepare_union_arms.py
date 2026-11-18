"""Prepare CRB+REFT and UNI+REFT datasets and configs for the eight-task batch."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import yaml
ROOT = Path(__file__).resolve().parents[3]
BATCH = "q16-src256-eight-task"
TASKS = ("FEV09", "TIME1", "TIME2", "TIME3", "TIME4", "TIME6", "TIME7", "TIME8")
ARMS = {"crb_reft": "crb", "uni_reft": "uni"}
SELECTION = ROOT / "outputs/selection/main-crb-v0" / BATCH
CONFIG_ROOT = ROOT / "configs/experiments/main-crb-v0/eight-task"
TRAIN_ROOT = ROOT / "outputs/train/main-crb-v0" / BATCH
EVAL_ROOT = ROOT / "outputs/eval/main-crb-v0" / BATCH

def read_yaml(path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)

def write_yaml(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=False)

def main():
    for task in TASKS:
        task_config_dir = CONFIG_ROOT / task
        for arm, source_arm in ARMS.items():
            source_windows_path = SELECTION / task / source_arm / "training_windows.npz"
            ref_windows_path = SELECTION / task / "reference/training_windows.npz"
            dataset_dir = SELECTION / task / arm
            dataset_dir.mkdir(parents=True, exist_ok=True)
            dataset_path = dataset_dir / "training_windows.npz"
            with np.load(source_windows_path) as source_file, np.load(ref_windows_path) as ref_file:
                source_windows = np.asarray(source_file["windows"])
                ref_windows = np.asarray(ref_file["windows"])
            if source_windows.ndim != 2 or ref_windows.ndim != 2 or source_windows.shape[1:] != ref_windows.shape[1:]:
                raise ValueError("incompatible window arrays for " + task + "/" + arm)
            if not np.isfinite(source_windows).all() or not np.isfinite(ref_windows).all():
                raise ValueError("non-finite windows for " + task + "/" + arm)
            merged = np.concatenate((source_windows, ref_windows), axis=0)
            np.savez(dataset_path, windows=merged)
            (dataset_dir / ".complete").write_text("complete\n", encoding="ascii")
            metadata = {
                "task": task, "arm": arm, "source_arm": source_arm,
                "source_windows": int(source_windows.shape[0]),
                "reference_windows": int(ref_windows.shape[0]),
                "training_windows": int(merged.shape[0]),
                "merge_rule": "concatenate selected source windows followed by all task reference windows",
                "deduplicate": False, "windows_shape": list(merged.shape),
                "windows_dtype": str(merged.dtype), "source_path": str(source_windows_path),
                "reference_path": str(ref_windows_path),
            }
            (dataset_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            base_train = read_yaml(task_config_dir / (source_arm + "-train.yaml"))
            base_train["run_id"] = "main-crb-v0-eight-task-" + task.lower() + "-" + arm
            base_train["data"]["id"] = task.lower() + "-" + arm + "-source-plus-reference"
            base_train["data"]["windows"] = str(dataset_path)
            base_train["output_dir"] = str(TRAIN_ROOT / task / arm)
            write_yaml(task_config_dir / (arm + "-train.yaml"), base_train)
            base_eval = read_yaml(task_config_dir / (source_arm + "-eval.yaml"))
            base_eval["checkpoint"] = str(TRAIN_ROOT / task / arm / "checkpoint")
            base_eval["output_dir"] = str(EVAL_ROOT / task / arm)
            write_yaml(task_config_dir / (arm + "-eval.yaml"), base_eval)
            pipeline = {
                "schema_version": 1, "run_id": base_train["run_id"],
                "stages": [
                    {"name": "train", "config": "configs/experiments/main-crb-v0/eight-task/" + task + "/" + arm + "-train.yaml"},
                    {"name": "eval", "config": "configs/experiments/main-crb-v0/eight-task/" + task + "/" + arm + "-eval.yaml"},
                ],
            }
            write_yaml(task_config_dir / (arm + "-pipeline.yaml"), pipeline)
            print("prepared task=%s arm=%s source=%d reference=%d total=%d" % (
                task, arm, source_windows.shape[0], ref_windows.shape[0], merged.shape[0]), flush=True)

if __name__ == "__main__":
    main()
