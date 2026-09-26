"""Load profile10b shard-dataset bundles as RAGS fragments.

The bundles follow ``ts3.shard-dataset.v1``: ``index.arrow`` maps each
trajectory to (shard_id, value_offset, length), ``meta.arrow`` carries
series identity, and ``shards/values_XXXXXX.bin`` stores float64 values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import numpy as np
import pyarrow.ipc as ipc

from rags.core import TimeSeriesFragment

REFERENCE_ROOT = Path(os.environ.get("RAGS_REFERENCE_ROOT",
    "data/ref/shard_dataset/profile-10b-segl512s512/fev"))
SOURCE_LOAD = Path(os.environ.get("RAGS_SOURCE_LOAD",
    "/root/autodl-tmp/TS3/data/src/shard_dataset/profile-10b-segl512s512/load"))


@dataclass
class BundleIndex:
    meta: dict[str, list]
    index: dict[str, list]
    load_dir: Path

    def series_count(self) -> int:
        return len(self.index["trajectory_id"])


def load_bundle_index(load_dir: Path) -> BundleIndex:
    with open(load_dir / "meta.arrow", "rb") as handle:
        meta = ipc.open_file(handle).read_all().to_pydict()
    with open(load_dir / "index.arrow", "rb") as handle:
        index = ipc.open_file(handle).read_all().to_pydict()
    return BundleIndex(meta=meta, index=index, load_dir=Path(load_dir))


def _read_series(bundle: BundleIndex, row: int) -> np.ndarray:
    shard_id = bundle.index["shard_id"][row]
    offset = bundle.index["value_offset"][row]
    length = bundle.index["length"][row]
    if bundle.index["dropped"][row]:
        return np.array([], dtype=np.float64)
    path = bundle.load_dir / "shards" / ("values_%06d.bin" % shard_id)
    values = np.memmap(path, dtype=np.float64, mode="r")
    return np.asarray(values[offset : offset + length])


def reference_load_dirs(root: Path | None = None, keywords: list[str] | None = None) -> list[Path]:
    """Reference load directories, optionally filtered by dataset keywords."""
    root = Path(root) if root is not None else Path(REFERENCE_ROOT)
    dirs = []
    for child in sorted(root.iterdir()):
        load = child / "load"
        if not load.is_dir():
            continue
        if keywords and not any(keyword in child.name for keyword in keywords):
            continue
        dirs.append(load)
    return dirs


def _dataset_label(load_dir: Path) -> str:
    """Decode the double-URL-encoded asset directory to a dataset name."""
    from urllib.parse import unquote

    name = unquote(unquote(load_dir.parent.name))
    parts = [part for part in name.split("/") if part]
    # .../autogluon___fev_datasets/<DATASET>/0.0.0/<hash>/<file>.arrow
    for index, part in enumerate(parts):
        if part.endswith("___fev_datasets") and index + 1 < len(parts):
            return parts[index + 1]
    return parts[-1] if parts else load_dir.parent.name


def build_reference_fragments(
    load_dirs: list[Path],
    *,
    window_length: int,
    slice_length: int,
    series_cap: int,
    min_length: int | None = None,
) -> list[TimeSeriesFragment]:
    """Tail slices of real reference series as fragments.

    Each fragment keeps one series' last ``slice_length`` points; entity_id
    records dataset and series identity for later reporting.
    """
    fragments: list[TimeSeriesFragment] = []
    for load_dir in load_dirs:
        bundle = load_bundle_index(load_dir)
        dataset_name = _dataset_label(load_dir)
        taken = 0
        for row in range(bundle.series_count()):
            if taken >= series_cap:
                break
            values = _read_series(bundle, row)
            if len(values) < (min_length or window_length):
                continue
            tail = np.asarray(values[-slice_length:], dtype=np.float64)
            if not np.isfinite(tail).all() or tail.std() == 0.0:
                continue
            series_id = bundle.meta["series_id"][row]
            fragments.append(
                TimeSeriesFragment(
                    data=tail.astype(np.float32),
                    entity_id="%s::%s" % (dataset_name, series_id),
                    fragment_id=len(fragments),
                )
            )
            taken += 1
    return fragments


def build_candidate_fragments(
    *,
    window_length: int,
    slice_length: int,
    num_series: int,
    seed: int = 0,
    load_dir: Path = SOURCE_LOAD,
) -> list[TimeSeriesFragment]:
    """Random source series tail slices as the candidate pool."""
    bundle = load_bundle_index(load_dir)
    rng = np.random.default_rng(seed)
    rows = rng.choice(bundle.series_count(), size=num_series * 4, replace=False)
    fragments: list[TimeSeriesFragment] = []
    for row in rows:
        if len(fragments) >= num_series:
            break
        values = _read_series(bundle, int(row))
        if len(values) < window_length:
            continue
        tail = np.asarray(values[-slice_length:], dtype=np.float64)
        if not np.isfinite(tail).all() or tail.std() == 0.0:
            continue
        fragments.append(
            TimeSeriesFragment(
                data=tail.astype(np.float32),
                entity_id=bundle.meta["series_id"][row][:120],
                fragment_id=len(fragments),
            )
        )
    return fragments


def sample_pool_windows(
    num: int,
    *,
    window_length: int,
    seed: int,
    load_dir: Path = SOURCE_LOAD,
) -> tuple[np.ndarray, list[str]]:
    """Uniformly sample windows from the entire source pool, no preprocessing.

    Each window comes from a uniformly chosen trajectory and a uniformly
    chosen valid start; this is the "no selection, no extra processing"
    baseline arm.
    """
    bundle = load_bundle_index(load_dir)
    rng = np.random.default_rng(seed)
    windows: list[np.ndarray] = []
    origins: list[str] = []
    attempts = 0
    while len(windows) < num and attempts < num * 50:
        attempts += 1
        row = int(rng.integers(0, bundle.series_count()))
        values = _read_series(bundle, row)
        if len(values) < window_length:
            continue
        max_start = len(values) - window_length
        start = int(rng.integers(0, max_start + 1))
        window = np.asarray(values[start : start + window_length], dtype=np.float64)
        if not np.isfinite(window).all() or window.std() == 0.0:
            continue
        windows.append(window.astype(np.float32))
        origins.append(str(bundle.meta["series_id"][row])[:120])
    if len(windows) < num:
        raise RuntimeError("could not sample %d finite windows from the pool" % num)
    return np.stack(windows), origins


def split_reference_fragment(
    fragment: TimeSeriesFragment,
    *,
    test_fraction: float,
    window_length: int,
) -> tuple[TimeSeriesFragment, TimeSeriesFragment]:
    """Split one reference series into a selection region and a test region.

    The selection fragment keeps the earlier history (used for gradients and
    perturbations); the test fragment keeps the later tail (held-out windows
    for downstream evaluation only). The two never overlap.
    """
    test_length = max(int(fragment.length * test_fraction), window_length)
    cut = fragment.length - test_length
    if cut < window_length:
        raise ValueError(
            "fragment of length %d cannot yield non-empty selection and test regions "
            "with window length %d" % (fragment.length, window_length)
        )
    selection = TimeSeriesFragment(
        data=fragment.data[:cut], entity_id=fragment.entity_id, fragment_id=fragment.fragment_id,
    )
    test = TimeSeriesFragment(
        data=fragment.data[cut:], entity_id=fragment.entity_id,
        fragment_id=10_000 + fragment.fragment_id,
    )
    return selection, test


# FEV-native exam sizes (domain A4): boomlet and ETT series end in 20
# head-to-tail rolling windows, jena in 10; window length 64, stride 64.
TEST_WINDOWS_PER_DATASET: dict[str, int] = {
    "boomlet_1676": 20,
    "boomlet_619": 20,
    "boomlet_1282": 20,
    "jena_weather_1H": 10,
    "ETT_1H": 20,
    "ETT_15T": 20,
}
DEFAULT_TEST_WINDOWS = 20


def test_window_count(dataset: str) -> int:
    return TEST_WINDOWS_PER_DATASET.get(dataset, DEFAULT_TEST_WINDOWS)


def native_test_rows(
    values: np.ndarray,
    *,
    dataset: str,
    context_length: int,
    prediction_length: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """FEV-native exam windows at the end of one full series.

    Targets are the last ``test_window_count(dataset) * H`` points split into
    head-to-tail horizon windows; each target window is evaluated with the C
    points immediately preceding it as context.  Returns (window rows, row
    start offsets into the series).
    """
    count = test_window_count(dataset)
    values = np.asarray(values, dtype=np.float64)
    span = count * prediction_length
    if len(values) < span + context_length:
        raise ValueError(
            "series of length %d too short for %d native test windows" % (
                len(values), count))
    rows: list[np.ndarray] = []
    starts: list[int] = []
    base = len(values) - span
    for index in range(count):
        start = base + index * prediction_length - context_length
        rows.append(values[start : start + context_length + prediction_length]
                    .astype(np.float32))
        starts.append(start)
    return rows, np.asarray(starts, dtype=np.int64)


def sample_ratio_pool(
    *,
    window_length: int,
    equivalents: int,
    seed: int = 0,
    load_dir: Path = SOURCE_LOAD,
) -> tuple[list[TimeSeriesFragment], dict]:
    """Ratio-based whole-trajectory source pool (domain A5).

    Draws complete trajectories from the source corpus with a fixed seed
    (no truncation, no splicing) until the cumulative non-overlapping window
    equivalents ``floor(L / window_length)`` reach ``equivalents``.  Returns
    the pool fragments and a manifest (trajectory origins and lengths) for
    provenance and cache fingerprinting.
    """
    bundle = load_bundle_index(load_dir)
    rng = np.random.default_rng(seed)
    order = rng.permutation(bundle.series_count())
    fragments: list[TimeSeriesFragment] = []
    manifest = {"seed": seed, "window_length": window_length,
                "equivalents_target": equivalents,
                "rows": [], "total_points": 0, "total_equivalents": 0}
    accumulated = 0
    for row in order:
        if accumulated >= equivalents:
            break
        values = _read_series(bundle, int(row))
        if len(values) < window_length:
            continue
        if not np.isfinite(values).all() or float(np.std(values)) == 0.0:
            continue
        fragments.append(TimeSeriesFragment(
            data=np.asarray(values, dtype=np.float32),
            entity_id=str(bundle.meta["series_id"][row])[:160],
            fragment_id=len(fragments),
        ))
        manifest["rows"].append(int(row))
        manifest["total_points"] += int(len(values))
        accumulated += len(values) // window_length
    manifest["total_equivalents"] = accumulated
    if accumulated < equivalents:
        raise RuntimeError(
            "source corpus exhausted at %d/%d window equivalents" % (
                accumulated, equivalents))
    return fragments, manifest


__all__ = [
    "load_bundle_index",
    "reference_load_dirs",
    "build_reference_fragments",
    "build_candidate_fragments",
    "sample_pool_windows",
    "split_reference_fragment",
    "TEST_WINDOWS_PER_DATASET",
    "test_window_count",
    "native_test_rows",
    "sample_ratio_pool",
    "SOURCE_LOAD",
]
