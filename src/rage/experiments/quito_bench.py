"""Quito reference prefixes and held-out windows for the FEV-style main table."""
from pathlib import Path

import numpy as np

from configs.paths import PROJECT_ROOT
from evaluate.quito_data import (REVISION, SEASONALITY, iter_series,
                                window_starts, seasonal_scale, validate_snapshot)
from rage.core import TimeSeriesFragment
from .tasks import TaskData, TaskSpec

DEFAULT_ROOT = PROJECT_ROOT / "data" / "bench" / "quito" / REVISION


def build_quito_task_specs(datasets=None, root=None):
    names = ["hour", "min"] if datasets is None else datasets
    if not names or len(set(names)) != len(names) or not set(names) <= {"hour", "min"}:
        raise ValueError("Quito datasets must be hour and/or min")
    validate_snapshot(Path(root or DEFAULT_ROOT))
    return [TaskSpec(dataset=name, load_dir=Path(root or DEFAULT_ROOT)) for name in names]


def build_quito_task(spec, *, context_length=512, prediction_length=64,
                     series_cap=None, pool_fragments=None):
    fragments, windows, entities, scales, constants = [], [], [], [], []
    histories = {}
    for entity, values, boundary in iter_series(spec.load_dir, spec.dataset):
        if series_cap is not None and len(fragments) >= series_cap:
            break
        # Selection uses only the pre-cutoff prefix, even when later exam
        # windows use observed test history as their forecasting context.
        prefix = values[:boundary]
        if len(prefix) < context_length + prediction_length or not np.isfinite(prefix).all():
            continue
        starts = window_starts(len(values), boundary, context_length,
                               prediction_length, "rolling", prediction_length)
        usable = [start for start in starts
                  if np.isfinite(values[start-context_length:start+prediction_length]).all()]
        if not usable:
            continue
        fragments.append(TimeSeriesFragment(data=prefix.astype(np.float32),
                         entity_id=entity, fragment_id=1000 * (len(fragments) + 1)))
        histories[entity] = values
        for start in usable:
            windows.append(values[start-context_length:start+prediction_length])
            entities.append(entity)
            scale = seasonal_scale(values[:start], SEASONALITY[spec.dataset])
            scales.append(scale)
            if scale == 0 and entity not in constants:
                constants.append(entity)
    if not windows:
        raise ValueError("Quito produced no usable reference/exam pairs")
    return TaskData(dataset="quito/%s" % spec.dataset, reference_fragments=fragments,
                    test_windows=np.asarray(windows, dtype=np.float32),
                    test_entities=entities, test_scales=np.asarray(scales),
                    constant_series=constants, histories=histories,
                    pool_fragments=pool_fragments or [])
