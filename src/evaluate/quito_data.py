"""Pinned QuitoBench data and causal windows for project evaluation/selection."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

REPO_ID = "hq-bench/quitobench"
REVISION = "ed4bf8eea27862aa01a063393ee48732910ecd87"
UPSTREAM_COMMIT = "262e7428db6ec186ba712ab5a4973501946d26ae"
CUTOFF = pd.Timestamp("2023-07-28", tz="UTC")
TARGETS = tuple("ind_%d" % i for i in range(1, 6))
FREQUENCIES = {"hour": "1h", "min": "10min"}
SEASONALITY = {"hour": 24, "min": 144}


def validate_snapshot(root: Path):
    manifest = json.loads((Path(root) / "data_manifest.json").read_text())
    if manifest.get("repo_id") != REPO_ID or manifest.get("revision") != REVISION:
        raise ValueError("Quito data manifest does not match pinned revision")


def data_file(root: Path, subset: str) -> Path:
    if subset not in FREQUENCIES:
        raise ValueError("Quito subset must be hour or min")
    return Path(root) / "v20260315" / ("test_%s-00001-of-00001.parquet" % subset)


def iter_series(root: Path, subset: str, targets=TARGETS):
    """Yield channels and the first test index; missing timestamps retain their slots.

    Values stay in original units. No interpolation, normalization or filtering
    based on future values is applied. Callers explicitly count invalid windows.
    """
    if not targets or len(set(targets)) != len(targets) or not set(targets) <= set(TARGETS):
        raise ValueError("targets must be a unique non-empty subset of ind_1..ind_5")
    frame = pd.read_parquet(data_file(root, subset), columns=["item_id", "date_time", *targets])
    frame["date_time"] = pd.to_datetime(frame["date_time"], utc=True)
    if frame[["item_id", "date_time"]].isna().any().any():
        raise ValueError("Quito item_id/date_time must not be missing")
    for item_id, rows in frame.groupby("item_id", sort=True):
        rows = rows.set_index("date_time").sort_index()
        if rows.index.has_duplicates:
            raise ValueError("duplicate Quito timestamps for item_id=%s" % item_id)
        grid = pd.date_range(rows.index[0], rows.index[-1], freq=FREQUENCIES[subset])
        if not rows.index.isin(grid).all():
            raise ValueError("Quito timestamps do not match declared frequency")
        rows = rows.reindex(grid)
        boundary = int(grid.searchsorted(CUTOFF))
        for target in targets:
            yield "%s/%s" % (item_id, target), rows[target].to_numpy(dtype=np.float64), boundary


def window_starts(length, boundary, context_length, prediction_length, mode, stride):
    """Forecast labels start at/after cutoff, including the single latest window."""
    if min(context_length, prediction_length, stride) <= 0:
        raise ValueError("context, horizon and stride must be positive")
    if mode not in {"single_window", "rolling"}:
        raise ValueError("evaluation_mode must be single_window or rolling")
    if boundary < context_length or length - boundary < prediction_length:
        return []
    if mode == "single_window":
        return [length - prediction_length]
    return range(boundary, length - prediction_length + 1, stride)


def seasonal_scale(history, seasonality):
    differences = np.abs(history[seasonality:] - history[:-seasonality])
    valid = differences[np.isfinite(differences)]
    return float(valid.mean()) if valid.size else float("nan")
