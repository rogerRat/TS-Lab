"""TIME-benchmark task adapter (integration prep, 2026-09-09).

Builds the same ``TaskSpec``/``TaskData`` objects the FEV main table uses,
but from the pinned TIME benchmark checkout (``third_parties/TIME`` commit
b6e7ba9, datasets.yaml) and its Hugging Face snapshot (``Real-TSF/TIME``
revision 83e3d0b3, fetched by ``scripts/fetch_time_dataset.sh`` under
``data/bench/time/<rev>/dataset``).

Protocol mapping (mirrors the FEV path in ``tasks.py``, domain A):
- usable series: univariate channels with length >= C+H, finite, non-constant;
- exam: head-to-tail rolling windows over the dataset's native ``test_length``
  tail, stride = prediction_length (H=64 uniform protocol, disclosed);
- selection region: full history minus the exam span;
- seasonal scale: full-history seasonal-naive scale with seasonality derived
  from the dataset frequency suffix (points per day, capped like domain E1).

The adapter is deliberately side-effect free: it never touches the FEV
main-table code path unless a caller asks for ``benchmark == "time"``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from rags.core import TimeSeriesFragment

from .metrics import history_seasonal_scale
from .tasks import TaskData, TaskSpec

from configs.paths import PROJECT_ROOT
TIME_CHECKOUT = PROJECT_ROOT / "third_parties" / "TIME"
TIME_COMMIT = "b6e7ba9862acb9233c8c55b314c1777052c420ff"
TIME_DATASETS_YAML = TIME_CHECKOUT / "src" / "timebench" / "config" / "datasets.yaml"
TIME_DATA_REVISION = "83e3d0b3be28d11c7182bffcc1892d19b36c4da1"
TIME_DATA_ROOT = (PROJECT_ROOT / "data" / "bench" / "time" / TIME_DATA_REVISION
                  / "dataset")

# Seasonal period in points per day for the frequency suffixes TIME uses.
_POINTS_PER_DAY = {
    "T": 1440, "5T": 288, "10T": 144, "15T": 96, "30T": 48,
    "H": 24, "h": 24, "D": 7, "B": 7, "W": 52, "M": 12,
}
_DEFAULT_SEASONALITY = 24


def load_time_datasets() -> dict:
    """Pinned TIME datasets.yaml content (``{name: {test_length, terms...}}``)."""
    if not TIME_DATASETS_YAML.is_file():
        raise FileNotFoundError(
            "TIME checkout missing (expected %s at commit %s); clone "
            "git@github.com:zqiao11/TIME.git and pin the commit"
            % (TIME_CHECKOUT, TIME_COMMIT))
    loaded = yaml.safe_load(TIME_DATASETS_YAML.read_text(encoding="ascii"))
    return loaded["datasets"]


def time_seasonality(dataset: str, freq: str) -> int:
    """Points-per-day seasonality from the frequency string, >= 2."""
    suffix = dataset.rsplit("/", 1)[-1]
    per_day = _POINTS_PER_DAY.get(suffix, _POINTS_PER_DAY.get(freq))
    if per_day is None:
        return _DEFAULT_SEASONALITY
    seasonality = max(2, int(round(per_day)))
    return min(seasonality, 1440)


def verify_time_checkout() -> None:
    """Fail fast if the pinned checkout drifted (evaluation provenance)."""
    import subprocess

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(TIME_CHECKOUT),
                          capture_output=True, text=True, check=True)
    if head.stdout.strip() != TIME_COMMIT:
        raise RuntimeError("TIME checkout is not at the pinned commit %s"
                           % TIME_COMMIT)


def build_time_task(spec: TaskSpec, *, term: str = "short",
                    context_length: int = 512, prediction_length: int = 64,
                    series_cap: int | None = None,
                    pool_fragments: list[TimeSeriesFragment] | None = None
                    ) -> TaskData:
    """Materialise one TIME dataset cell under the rags window protocol."""
    import datasets as hf_datasets

    settings = load_time_datasets()
    if spec.dataset not in settings:
        raise KeyError("unknown TIME dataset: %s" % spec.dataset)
    entry = settings[spec.dataset]
    test_length = int(entry["test_length"])
    term_settings = entry.get(term)
    if not isinstance(term_settings, dict) or "prediction_length" not in term_settings:
        raise KeyError("TIME dataset %s has no term %r" % (spec.dataset, term))

    dataset_path = TIME_DATA_ROOT / spec.dataset
    if not dataset_path.exists():
        raise FileNotFoundError(
            "TIME data missing: %s (run scripts/fetch_time_dataset.sh)"
            % dataset_path)
    hf = hf_datasets.load_from_disk(str(dataset_path))

    window_length = context_length + prediction_length
    reference_fragments: list[TimeSeriesFragment] = []
    histories: dict[str, np.ndarray] = {}
    test_rows: list[np.ndarray] = []
    test_entities: list[str] = []
    test_scales: list[float] = []
    constant_series: list[str] = []
    usable = 0
    for row in hf:
        target = np.asarray(row["target"], dtype=np.float64)
        freq = str(row.get("freq", ""))
        channels = target if target.ndim == 2 else target[None, :]
        for channel_index, channel in enumerate(channels):
            if series_cap is not None and usable >= series_cap:
                break
            if len(channel) < window_length + test_length:
                continue
            if not np.isfinite(channel).all() or float(np.std(channel)) == 0.0:
                continue
            entity = "%s[%d]" % (row.get("item_id", spec.dataset), channel_index)
            usable += 1
            histories[entity] = channel
            seasonality = time_seasonality(spec.dataset, freq)
            # Native exam: head-to-tail windows over the test tail.
            n_windows = test_length // prediction_length
            span = test_length
            tail = channel[-span:]
            for w in range(n_windows):
                start = w * prediction_length
                horizon = tail[start:start + prediction_length]
                context = channel[len(channel) - span + start - context_length:
                                  len(channel) - span + start]
                test_rows.append(np.concatenate([context, horizon]))
                test_entities.append(entity)
                scale = history_seasonal_scale(channel[:len(channel) - span + start], seasonality)
                test_scales.append(scale)
                if scale == 0.0 and entity not in constant_series:
                    constant_series.append(entity)
            selection = np.asarray(channel[:-span], dtype=np.float32)
            reference_fragments.append(TimeSeriesFragment(
                data=selection, entity_id=entity,
                fragment_id=1000 * (1 + len(reference_fragments)),
            ))
    if not test_rows:
        raise RuntimeError("TIME dataset %s produced no test windows"
                           % spec.dataset)
    return TaskData(
        dataset="time/%s/%s" % (spec.dataset, term),
        reference_fragments=reference_fragments,
        test_windows=np.stack(test_rows).astype(np.float32, copy=False),
        test_entities=test_entities,
        test_scales=np.asarray(test_scales, dtype=np.float64),
        constant_series=constant_series,
        histories=histories,
        pool_fragments=pool_fragments or [],
    )


def build_time_task_specs(datasets: list[str] | None = None) -> list[TaskSpec]:
    """TaskSpecs for TIME datasets (names like ``Water_Quality_Darwin/15T``)."""
    available = load_time_datasets()
    names = datasets or list(available)
    missing = [name for name in names if name not in available]
    if missing:
        raise KeyError("unknown TIME datasets: %s" % missing)
    return [TaskSpec(dataset=name, load_dir=TIME_DATA_ROOT / name)
            for name in names]


__all__ = [
    "TIME_CHECKOUT", "TIME_COMMIT", "TIME_DATA_ROOT",
    "build_time_task", "build_time_task_specs", "load_time_datasets",
    "time_seasonality", "verify_time_checkout",
]
