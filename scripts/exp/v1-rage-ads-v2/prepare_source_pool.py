"""Materialize the frozen src-256 trajectory pool from src-96 and src-4096."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "data/src/shard_dataset"
CONFIG_ROOT = ROOT / "configs/data"
TARGET_NAME = "src-256"
SEED = 0


def log(stage: str, message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"{stamp}-[prepare_source_pool]-[{stage}]-[{message}]", flush=True)


def read_config(name: str) -> dict:
    path = CONFIG_ROOT / f"{name}.yaml"
    return yaml.safe_load(path.read_text(encoding="ascii"))


def key(row: dict) -> str:
    return str(row["source"]) + "/" + str(row["subset"])


def largest_remainder(weights: dict[str, int], total: int) -> dict[str, int]:
    denominator = sum(weights.values())
    if denominator <= 0:
        raise ValueError("No eligible source trajectories remain")
    raw = {name: total * count / denominator for name, count in weights.items()}
    quota = {name: math.floor(value) for name, value in raw.items()}
    remaining = total - sum(quota.values())
    order = sorted(weights, key=lambda name: (-(raw[name] - quota[name]), name))
    for name in order[:remaining]:
        quota[name] += 1
    return quota


def main() -> None:
    started = time.monotonic()
    source96 = read_config("src-96")
    source4096 = read_config("src-4096")
    pool96 = DATA_ROOT / "src-96"
    pool4096 = DATA_ROOT / "src-4096"
    target = DATA_ROOT / TARGET_NAME
    target_config = CONFIG_ROOT / f"{TARGET_NAME}.yaml"
    if target.exists() or target_config.exists():
        raise FileExistsError("src-256 output already exists; refusing to overwrite")

    log("start", "trajectories=256 inherited=96 additions=160 seed=0")
    parent_rows = source4096["trajectories"]
    inherited_rows = source96["trajectories"]
    inherited_ordinals = {int(row["parent_ordinal"]) for row in inherited_rows}
    if len(inherited_ordinals) != 96:
        raise ValueError("src-96 parent ordinals are not unique")
    parent_by_ordinal = {int(row["ordinal"]): row for row in parent_rows}
    if not inherited_ordinals.issubset(parent_by_ordinal):
        raise ValueError("src-96 references a missing src-4096 ordinal")

    inherited_by_key: dict[str, list[dict]] = {}
    remaining_by_key: dict[str, list[dict]] = {}
    for row in inherited_rows:
        inherited_by_key.setdefault(key(row), []).append(row)
    for row in parent_rows:
        if int(row["ordinal"]) not in inherited_ordinals:
            remaining_by_key.setdefault(key(row), []).append(row)
    if sum(map(len, remaining_by_key.values())) != 4000:
        raise ValueError("Expected exactly 4000 parent trajectories after excluding src-96")
    weights = {name: len(rows) for name, rows in remaining_by_key.items()}
    additions_quota = largest_remainder(weights, 160)
    additions: list[dict] = []
    rng = np.random.default_rng(SEED)
    for name in sorted(remaining_by_key):
        rows = remaining_by_key[name]
        count = additions_quota[name]
        chosen = rng.choice(len(rows), size=count, replace=False) if count else []
        additions.extend(rows[int(index)] for index in chosen)
        log("allocation", f"configuration={name} candidates={len(rows)} selected={count}")

    selected_ordinals = inherited_ordinals | {int(row["ordinal"]) for row in additions}
    if len(selected_ordinals) != 256:
        raise ValueError("Selected trajectory count is not 256")
    ordered_parent = sorted(selected_ordinals)
    parent_offsets = np.load(pool4096 / "offsets.npy", mmap_mode="r")
    parent_values = np.load(pool4096 / "values.npy", mmap_mode="r")
    config_rows: list[dict] = []
    values_chunks: list[np.ndarray] = []
    for ordinal in ordered_parent:
        source_row = parent_by_ordinal[ordinal]
        start, end = int(parent_offsets[ordinal]), int(parent_offsets[ordinal + 1])
        if end - start != int(source_row["length"]):
            raise ValueError(f"Parent offset length mismatch at ordinal={ordinal}")
        chunk = np.asarray(parent_values[start:end])
        if not np.isfinite(chunk).all():
            raise ValueError(f"Non-finite source values at ordinal={ordinal}")
        values_chunks.append(chunk)
        config_rows.append({
            "ordinal": len(config_rows),
            "source": source_row["source"],
            "subset": source_row["subset"],
            "series_id": source_row["series_id"],
            "length": int(source_row["length"]),
            "source_ordinal": int(source_row["source_ordinal"]),
            "parent_ordinal": ordinal,
        })

    config_counts: dict[str, int] = {}
    for row in config_rows:
        config_counts[key(row)] = config_counts.get(key(row), 0) + 1
    values = np.concatenate(values_chunks)
    offsets = np.concatenate(([0], np.cumsum([len(chunk) for chunk in values_chunks]))).astype(np.int64)
    manifest = {
        "name": TARGET_NAME,
        "parent": "src-4096",
        "trajectory_count": len(config_rows),
        "configuration_count": len(config_counts),
        "total_points": int(offsets[-1]),
        "configuration_counts": dict(sorted(config_counts.items())),
        "selection_seed": SEED,
        "selection_rule": "retain src-96; allocate 160 additions by remaining src-4096 configuration counts with largest remainder; seed 0 without replacement",
        "inherited_trajectory_count": len(inherited_rows),
        "addition_trajectory_count": len(additions),
    }
    target.mkdir(parents=True)
    try:
        np.save(target / "values.npy", values)
        np.save(target / "offsets.npy", offsets)
        (target / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii")
        config = {
            "name": TARGET_NAME,
            "parent": "src-4096",
            "selection_seed": SEED,
            "selection_level": "trajectory",
            "allocation": "retain src-96; remaining src-4096 configuration proportional allocation with largest remainder",
            "trajectory_count": len(config_rows),
            "configuration_count": len(config_counts),
            "total_points": int(offsets[-1]),
            "window_length": 576,
            "configuration_counts": dict(sorted(config_counts.items())),
            "values_path": f"data/src/shard_dataset/{TARGET_NAME}/values.npy",
            "offsets_path": f"data/src/shard_dataset/{TARGET_NAME}/offsets.npy",
            "trajectories": config_rows,
        }
        temp_config = target_config.with_suffix(".yaml.partial")
        temp_config.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=False), encoding="ascii")
        os.replace(temp_config, target_config)

        loaded_offsets = np.load(target / "offsets.npy", mmap_mode="r")
        loaded_values = np.load(target / "values.npy", mmap_mode="r")
        if len(loaded_offsets) != 257 or int(loaded_offsets[-1]) != len(loaded_values):
            raise ValueError("Materialized packed arrays failed dimensions check")
        materialized = {int(row["parent_ordinal"]): row for row in config_rows}
        for inherited in inherited_rows:
            actual = materialized.get(int(inherited["parent_ordinal"]))
            if actual is None or actual["series_id"] != inherited["series_id"]:
                raise ValueError("src-96 trajectory identity was not retained")
        if sum(additions_quota.values()) != 160:
            raise ValueError("Addition quota does not sum to 160")
        (target / ".complete").write_text("complete\n", encoding="ascii")
    except Exception:
        raise

    log("complete", f"trajectories=256 points={int(offsets[-1])} duration={time.monotonic() - started:.1f}s output={target}")


if __name__ == "__main__":
    main()
