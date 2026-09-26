"""Task construction for the main-table experiment.

Six FEv reference datasets (the six with the most usable reference series in
``profile-10b-segl512s512``), one shared ratio-sampled source pool, and the
fixed window protocol C=512, H=64, D=1 for selection, training, evaluation.

Domains A3/A4 (all available series, full history, FEV-native exams): every
usable series enters with its complete history; the exam is the series' last
``test_window_count`` head-to-tail horizon windows; the remaining history is
the selection region.  Reference data never enters training (domain A7) and
there is no target-adaptation stage (domain D1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json

import numpy as np

from rags.core import TimeSeriesFragment

from .data import (
    BundleIndex,
    load_bundle_index,
    native_test_rows,
    reference_load_dirs,
    sample_ratio_pool,
)
from .metrics import DATASET_SEASONALITY, history_seasonal_scale, seasonality_for

# The six reference datasets with the most usable series (>= 576 points).
MAIN_DATASETS: tuple[str, ...] = (
    "boomlet_1676",  # 100 usable series, 30-minute frequency
    "boomlet_619",   # 52 usable series, 1-minute frequency
    "boomlet_1282",  # 35 usable series, 1-minute frequency
    "jena_weather_1H",  # 21 usable series, hourly
    "ETT_1H",        # 14 usable series, hourly
    "ETT_15T",       # 14 usable series, 15-minute
)


@dataclass
class TaskSpec:
    """One (dataset) cell of the main table."""

    dataset: str
    load_dir: Path
    fragments: list[TimeSeriesFragment] = field(default_factory=list)

    @property
    def entity_ids(self) -> list[str]:
        return [f.entity_id or "" for f in self.fragments]


@dataclass
class TaskData:
    """Fully materialised windows for one dataset cell."""

    dataset: str
    # Selection region (queries); full history minus the native exam tail.
    reference_fragments: list[TimeSeriesFragment]
    # Held-out native exam windows [K, C+H] with per-window entity ids.
    test_windows: np.ndarray
    test_entities: list[str]
    # Full-history seasonal-naive scale per test window (domain E1).
    test_scales: np.ndarray
    # Constant series (zero scale) are tracked for separate MAE reporting.
    constant_series: list[str]
    # Full series history per entity id (scale provenance and diagnostics).
    histories: dict[str, np.ndarray]
    # Source pool shared across all datasets.
    pool_fragments: list[TimeSeriesFragment]


def build_task_specs(series_cap: int | None = None,
                     keywords: tuple[str, ...] = MAIN_DATASETS,
                     reference_root: Path | None = None) -> list[TaskSpec]:
    dirs = reference_load_dirs(root=reference_root, keywords=list(keywords))
    by_dataset: dict[str, Path] = {}
    for load_dir in dirs:
        from .data import _dataset_label

        by_dataset[_dataset_label(load_dir)] = load_dir
    missing = [name for name in keywords if name not in by_dataset]
    if missing:
        raise FileNotFoundError("reference datasets missing from profile10b: %s" % missing)
    return [TaskSpec(dataset=name, load_dir=by_dataset[name]) for name in keywords]


def _usable_full_series(bundle: BundleIndex, dataset: str, *, window_length: int,
                        series_cap: int | None) -> list[tuple[str, np.ndarray]]:
    rows: list[tuple[str, np.ndarray]] = []
    from .data import _read_series

    for row in range(bundle.series_count()):
        if series_cap is not None and len(rows) >= series_cap:
            break
        values = _read_series(bundle, row)
        if len(values) < window_length:
            continue
        if not np.isfinite(values).all() or float(np.std(values)) == 0.0:
            continue
        rows.append(("%s::%s" % (dataset, bundle.meta["series_id"][row]), values))
    return rows


def build_task_data(spec: TaskSpec, *,
                    context_length: int,
                    prediction_length: int,
                    series_cap: int | None = None,
                    slice_length: int | None = None,
                    pool_fragments: list[TimeSeriesFragment]) -> TaskData:
    """Load one dataset: native exam at the series end, full history as the
    selection region, per-series full-history seasonal scales.

    ``slice_length`` (dev subsets only, e.g. mechanism experiments' 4096)
    tails every series before the native exam split; the seasonal scale is
    still computed from the untouched full history.
    """
    window_length = context_length + prediction_length
    bundle: BundleIndex = load_bundle_index(spec.load_dir)
    series_rows = _usable_full_series(bundle, spec.dataset,
                                      window_length=window_length,
                                      series_cap=series_cap)
    if not series_rows:
        raise RuntimeError("no usable reference series for %s" % spec.dataset)

    reference_fragments: list[TimeSeriesFragment] = []
    histories: dict[str, np.ndarray] = {}
    test_rows: list[np.ndarray] = []
    test_entities: list[str] = []
    test_scales: list[float] = []
    constant_series: list[str] = []
    for entity, values in series_rows:
        histories[entity] = np.asarray(values, dtype=np.float64)
        seasonality = seasonality_for(entity)
        work = values if slice_length is None else values[-slice_length:]
        rows, offsets = native_test_rows(
            work, dataset=spec.dataset, context_length=context_length,
            prediction_length=prediction_length)
        span = len(rows) * prediction_length
        selection = np.asarray(work[:-span], dtype=np.float32)
        reference_fragments.append(TimeSeriesFragment(
            data=selection, entity_id=entity,
            fragment_id=1000 * (1 + len(reference_fragments)),
        ))
        test_rows.extend(rows)
        test_entities.extend([entity] * len(rows))
        # FEV uses y_past at each forecast origin, never the held-out future.
        # Preserve full available history even when dev scoring uses a tail slice.
        scales = [history_seasonal_scale(values[:len(values)-len(work)+int(start)+context_length], seasonality)
                  for start in offsets]
        if any(scale == 0.0 for scale in scales):
            constant_series.append(entity)
        test_scales.extend(scales)
    if not test_rows:
        raise RuntimeError("dataset %s produced no test windows" % spec.dataset)
    return TaskData(
        dataset=spec.dataset,
        reference_fragments=reference_fragments,
        test_windows=np.stack(test_rows).astype(np.float32, copy=False),
        test_entities=test_entities,
        test_scales=np.asarray(test_scales, dtype=np.float64),
        constant_series=constant_series,
        histories=histories,
        pool_fragments=pool_fragments,
    )


def dataset_seasonality(dataset: str) -> int:
    return DATASET_SEASONALITY.get(dataset, 24)


def build_ratio_pool(*, context_length: int, prediction_length: int,
                     equivalents: int, seed: int = 0,
                     cache_dir: Path | None = None,
                     ) -> tuple[list[TimeSeriesFragment], dict]:
    """The shared source pool (A5) with an on-disk manifest cache.

    The pool draw itself is deterministic given (seed, equivalents), so the
    cache only stores the manifest; fragments are re-read from the corpus.
    """
    window_length = context_length + prediction_length
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = hashlib.sha1(json.dumps(
            {"seed": seed, "equivalents": equivalents,
             "window_length": window_length}).encode()).hexdigest()[:16]
        manifest_path = cache_dir / ("pool_manifest_%s.json" % fingerprint)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("total_equivalents", 0) >= equivalents:
                return _pool_from_manifest(manifest, window_length), manifest
    fragments, manifest = sample_ratio_pool(
        window_length=window_length, equivalents=equivalents, seed=seed)
    if cache_dir is not None:
        manifest_path.write_text(json.dumps(manifest))
    return fragments, manifest


def _pool_from_manifest(manifest: dict, window_length: int) -> list[TimeSeriesFragment]:
    from .data import SOURCE_LOAD, _read_series

    bundle = load_bundle_index(SOURCE_LOAD)
    fragments: list[TimeSeriesFragment] = []
    for fragment_id, row in enumerate(manifest["rows"]):
        values = _read_series(bundle, int(row))
        fragments.append(TimeSeriesFragment(
            data=np.asarray(values, dtype=np.float32),
            entity_id="pool-row-%d" % int(row),
            fragment_id=fragment_id,
        ))
    return fragments


def pool_windows(pool: list[TimeSeriesFragment], *, context_length: int,
                 prediction_length: int, stride: int = 1,
                 ) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """All legal pool windows (D=stride) and their (fragment_id, start) ids.

    Only for small pools / dev subsets; the formal pool uses WindowIndex.
    """
    rows = []
    ids: list[tuple[int, int]] = []
    for fragment in pool:
        starts = fragment.valid_starts(context_length, prediction_length, stride)
        for start in starts:
            rows.append(fragment.extract_window(int(start), context_length,
                                                prediction_length))
            ids.append((fragment.fragment_id, int(start)))
    if not rows:
        raise RuntimeError("source pool produced no windows")
    return np.stack(rows).astype(np.float32, copy=False), ids


__all__ = [
    "MAIN_DATASETS",
    "TaskSpec",
    "TaskData",
    "build_task_specs",
    "build_task_data",
    "build_ratio_pool",
    "dataset_seasonality",
    "pool_windows",
]
