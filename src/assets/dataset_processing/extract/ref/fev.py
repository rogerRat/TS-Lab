"""FEV reference adapter: publish only FEV's official past evaluation window."""

from __future__ import annotations

import logging
import os
from numbers import Integral
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np

from tsl.spec.ref_extraction import ReferenceExtractionSpecification


log = logging.getLogger("extract.ref")


def _save_records(records, destination: Path, *, contains_non_safe_history: bool) -> tuple[int, int]:
    from datasets import Dataset, DatasetInfo

    rows = list(records)
    if not rows:
        raise ValueError("FEV extraction produced no trajectories")
    value_count = sum(len(row["target"]) for row in rows)
    log.info(
        "Saving FEV asset: destination=%s trajectories=%d values=%d",
        destination, len(rows), value_count,
    )
    description = "contains_non_safe_history=%s" % str(contains_non_safe_history).lower()
    Dataset.from_list(rows, info=DatasetInfo(description=description)).save_to_disk(str(destination))
    log.info("FEV asset saved: destination=%s", destination)
    return len(rows), value_count


def _normalize_boundary(value: Any, history_lengths: list[int]) -> int | str:
    """Convert FEV cutoffs to manifest-native absolute positions or timestamps."""
    if isinstance(value, Integral):
        position = int(value)
        if position < 0:
            if not history_lengths:
                raise ValueError("FEV produced no history rows for a negative cutoff")
            # FEV uses Python negative slicing for relative cutoffs.  The
            # manifest stores an absolute, non-negative safe-history length.
            return min(history_lengths)
        return position
    return value if isinstance(value, str) else str(value)


def _records(task: Any, *, include_future: bool) -> tuple[list[dict[str, Any]], Any, Any]:
    """Use FEV's first window, optionally retaining its answer interval for debugging."""
    window = task.get_window(0, num_proc=1)
    past, _future_inputs = window.get_input_data()
    future = window.get_ground_truth() if include_future else None
    records = []
    history_lengths = []
    future_rows = future.with_format("numpy") if future is not None else (None for _ in range(len(past)))
    for row_number, (row, future_row) in enumerate(zip(past.with_format("numpy"), future_rows)):
        item_id = str(row[task.id_column])
        for target_name in task.target_columns:
            target = row[target_name]
            history_lengths.append(len(target))
            if include_future:
                target = np.concatenate((target, future_row[target_name]))
            target = target.tolist()
            records.append({"item_id": "%s:%s" % (item_id, target_name), "target": target, "freq": str(task.freq)})
    boundary = _normalize_boundary(task.cutoffs[0], history_lengths)
    return records, boundary, boundary


def extract_assets(spec: ReferenceExtractionSpecification, declaration_path: Path, data_root: Path, declaration: Mapping[str, Any], output_dir: Path, digest: Callable[..., str], *, include_future: bool = False) -> list[dict[str, Any]]:
    """Return manifest assets in the schema-owned asset mapping format."""
    os.environ["HF_HOME"] = str(data_root)
    from evaluate.fev import load_task_declarations
    import fev

    tasks = load_task_declarations(declaration_path)
    selection = spec.mapping["task_selection"]
    indices = range(len(tasks)) if selection["mode"] == "all" else [int(item) for item in selection["items"]]
    indices = list(indices)
    if include_future:
        log.warning(
            "FEV DEBUG ASSET INCLUDES NON-SAFE HISTORY: it contains benchmark answer values and must never be used for training or evaluation"
        )
    log.info(
        "FEV extraction prepared: declaration=%s tasks_declared=%d tasks_selected=%d selection_mode=%s",
        declaration_path, len(tasks), len(indices), selection["mode"],
    )
    assets = []
    from assets.dataset_processing.progress import ProcessingProgress

    with ProcessingProgress(
        log, "EXTRACT_REF", "fev_tasks", len(indices),
        detail="data_root=%s output_dir=%s" % (data_root, output_dir),
    ) as progress:
        for index in indices:
            started = perf_counter()
            log.info("FEV task starting: index=%d declaration_keys=%s", index, sorted(tasks[index]))
            task = fev.Task(**tasks[index])
            # FEV exposes metadata such as ``freq`` only after the complete
            # dataset has been loaded.  Load before logging metadata or reading
            # the first evaluation window so the provider lifecycle is explicit.
            task.load_full_dataset()
            log.info(
                "FEV task loaded: index=%d task_name=%s dataset=%s config=%s targets=%s freq=%s",
                index, task.task_name, task.dataset_path, task.dataset_config or "none",
                list(task.target_columns), task.freq,
            )
            log.info("FEV task window loading: index=%d window=0", index)
            records, training_boundary, test_boundary = _records(task, include_future=include_future)
            log.info(
                "FEV task window loaded: index=%d trajectories=%d values=%d boundary=%s",
                index, len(records), sum(len(row["target"]) for row in records),
                training_boundary,
            )
            identity = "%s:%s" % (task.dataset_path, task.dataset_config or task.task_name)
            projection = digest(identity, ",".join(task.target_columns))
            asset_id = digest("fev", identity, spec.mapping["upstream"]["data_revision"], projection, spec.mapping["prefix_policy"])
            native = Path("assets") / asset_id
            count, values = _save_records(records, output_dir / native, contains_non_safe_history=include_future)
            kind = "position" if isinstance(training_boundary, int) else "timestamp"
            assets.append({"asset_id": asset_id, "asset_identity": identity, "series_projection_id": projection, "task_ids": [str(index)], "earliest_test_boundary": {"kind": kind, "value": test_boundary, "provider_evidence": "fev.Task.get_window(0)"}, "training_boundary": {"kind": kind, "value": training_boundary}, "native_output": {"location": str(native), "format": "huggingface-dataset-on-disk.v1"}, "source": {"identity": identity, "revision": spec.mapping["upstream"]["data_revision"]}, "trajectory_count": count, "value_count": values})
            log.info(
                "FEV task completed: index=%d asset_id=%s identity=%s trajectories=%d values=%d boundary_kind=%s boundary=%s elapsed_seconds=%.1f",
                index, asset_id, identity, count, values, kind, training_boundary,
                perf_counter() - started,
            )
            progress.advance(detail="last_task=%d asset_id=%s" % (index, asset_id))
    return assets
