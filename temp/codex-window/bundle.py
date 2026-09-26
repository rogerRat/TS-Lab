"""Arrow-only Feature stage for a ShardDatasetBundle.

Features are a parallel Arrow interpretation of immutable Load bins rather
than a new data representation, keeping storage and provenance centralized.

Every valid trajectory is covered by analysis segments of
ANALYSIS_SEGMENT_LENGTH. ANALYSIS_SEGMENT_STRIDE may make adjacent segments
overlap; the default stride preserves non-overlapping segments. A segment is
only an analysis proxy, never the minimum legal sample. Therefore a
trajectory shorter than the analysis segment still gets one segment covering
its complete length; the runtime window specification decides later whether
the trajectory can produce a training sample.
"""

from __future__ import annotations

import json
from pathlib import Path
import logging
import time

import numpy as np

from assets.dataset_processing.shard_dataset import ShardDatasetReader
from ts3.features import compute_descriptors
from ts3.features.cuda import FeatureCudaExecutor, cuda_available
from ts3.features.config import ANALYSIS_SEGMENT_LENGTH, ANALYSIS_SEGMENT_STRIDE
from ts3.spec.files.shard_dataset import (
    FEATURE_METADATA_FILENAME,
    FEATURE_METADATA_SCHEMA_VERSION_V1,
    FEATURE_SCHEMA_V1,
    validate_shard_dataset_bundle,
    write_arrow,
)

log = logging.getLogger("assets.dataset_processing.features")

FEATURE_NAMES = (
    "trend_strength",
    "trend_profile_strength",
    "trend_shape_1",
    "trend_shape_2",
    "trend_shape_3",
    "trend_shape_4",
    "roughness",
    "memory",
    "volatility_clustering",
    "spectral_entropy",
    "spectral_centroid",
    "spectral_spread",
    "asymmetry",
    "tail_weight",
    "intermittency",
    "change_rate",
)
SEGMENT_FEATURE_INVALID_REASON = "segment_feature_invalid"


def _segment_starts(trajectory_length: int, segment_length: int, segment_stride: int) -> tuple[int, ...]:
    if trajectory_length <= segment_length:
        return (0,)
    starts = list(range(0, trajectory_length - segment_length + 1, segment_stride))
    final_start = trajectory_length - segment_length
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(starts)


def _feature_mapping(values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(FEATURE_NAMES, values)}


def _segment_rows(values: np.ndarray, segment_length: int, *, segment_stride: int = ANALYSIS_SEGMENT_STRIDE) -> list[dict] | None:
    """Compute v2 full-segment and boundary-transition descriptors."""
    if not np.isfinite(values).all():
        return None
    segments = []
    for segment_index, start in enumerate(_segment_starts(len(values), segment_length, segment_stride)):
        segment_values = values[start:min(start + segment_length, len(values))]
        result = compute_descriptors(segment_values)
        feature_values = {name: float(result[name]) for name in FEATURE_NAMES}
        valid = (
            result.get("failure_reason") is None
            and all(np.isfinite(value) and 0 <= value <= 1 for value in feature_values.values())
        )
        if not valid:
            return None
        segments.append({
            "segment_index": segment_index,
            "start": start,
            "length": int(len(segment_values)),
            "feature_values": feature_values,
        })
    return segments


def _row_result(trajectory_id: str, segments: list[dict] | None, *, error: bool = False) -> dict:
    valid = segments is not None
    return {
        "trajectory_id": trajectory_id,
        "dropped": not valid,
        "drop_stage": None if valid else "feature",
        "drop_reason": None if valid else ("feature_invalid" if error else SEGMENT_FEATURE_INVALID_REASON),
        "segments": segments if valid else None,
    }


def _cuda_segment_record(output, output_index: int, segment_index: int, start: int, length: int) -> dict | None:
    feature_values = {name: float(value) for name, value in zip(FEATURE_NAMES, output.feature_values[output_index])}
    valid = all(np.isfinite(value) and 0 <= value <= 1 for value in feature_values.values())
    if not valid:
        return None
    return {
        "segment_index": segment_index,
        "start": start,
        "length": int(length),
        "feature_values": feature_values,
    }


def _cuda_fallback_row(state: dict, segment_length: int, segment_stride: int) -> dict:
    try:
        segments = _segment_rows(
            state["values"],
            segment_length,
            segment_stride=segment_stride,
        )
        return _row_result(state["trajectory_id"], segments)
    except Exception:
        log.exception("Feature CPU fallback failed: trajectory_id=%s", state["trajectory_id"])
        return _row_result(state["trajectory_id"], None, error=True)


def _flush_cuda_batch(
    executor: FeatureCudaExecutor,
    entries: list[tuple[dict, int, int, np.ndarray]],
    row_results: list[dict | None],
    segment_length: int,
    segment_stride: int,
    batch_number: int,
) -> None:
    if not entries:
        return
    states = {id(entry[0]): entry[0] for entry in entries}
    started = time.perf_counter()
    try:
        values = np.stack([entry[3] for entry in entries], axis=0)
        output, timing = executor.compute(values)
        invalid = any(
            _cuda_segment_record(output, index, entry[1], entry[2], entry[3].shape[0]) is None
            for index, entry in enumerate(entries)
        )
        if invalid:
            raise ValueError("CUDA descriptor output failed the finite [0, 1] protocol")
        for index, (state, segment_index, start, segment_values) in enumerate(entries):
            state["segments"][segment_index] = _cuda_segment_record(output, index, segment_index, start, segment_values.shape[0])
            state["pending"] -= 1
        log.info(
            "Feature GPU storage transfer: batch=%d segments=%d host_bytes=%d h2d_ms=%.2f d2h_ms=%.2f",
            batch_number,
            len(entries),
            values.nbytes,
            timing.h2d_ms,
            timing.d2h_ms,
        )
        log.info(
            "Feature GPU compute: batch=%d segments=%d compute_ms=%.2f wall_ms=%.2f",
            batch_number,
            len(entries),
            timing.compute_ms,
            (time.perf_counter() - started) * 1000.0,
        )
    except Exception:
        log.exception("Feature GPU batch failed; falling back to CPU: batch=%d segments=%d", batch_number, len(entries))
        for state in states.values():
            if state["done"]:
                continue
            row_results[state["row_index"]] = _cuda_fallback_row(state, segment_length, segment_stride)
            state["done"] = True
            state["pending"] = 0
            state["values"] = None
        return
    for state in states.values():
        if state["pending"] == 0 and not state["done"]:
            row_results[state["row_index"]] = _row_result(state["trajectory_id"], state["segments"])
            state["done"] = True
            state["values"] = None


def derive_bundle_features(bundle_root: Path, role: str | None = None, *, use_gpu: bool | None = None, gpu_batch_size: int | None = None) -> Path:
    """Write segment-level feature status and values without altering Load shards."""
    log.info("Features starting: bundle=%s", bundle_root)
    bundle = validate_shard_dataset_bundle(bundle_root)
    log.info("Bundle validated: load_rows=%d dedup_rows=%d", len(bundle.load_index), len(bundle.dedup or ()))
    feature_directory = bundle.root / "feature"
    target = feature_directory / "feature.arrow"
    metadata_target = feature_directory / FEATURE_METADATA_FILENAME
    if target.exists() or metadata_target.exists():
        raise FileExistsError(target if target.exists() else metadata_target)
    prior = {row["trajectory_id"]: row for row in bundle.dedup or ()}
    scope_by_id = {row["trajectory_id"]: row["scope"] for row in bundle.meta}
    if use_gpu is None:
        use_gpu = cuda_available()
    executor = None
    if use_gpu:
        try:
            executor = FeatureCudaExecutor(batch_size=gpu_batch_size)
        except Exception:
            log.exception("Feature GPU backend unavailable; using CPU feature computation")
    if executor is None:
        log.info("Feature CPU backend active")
    else:
        log.info("Feature GPU backend active: device=%s batch_size=%d", executor.device, executor.batch_size)
    reader, row_results = ShardDatasetReader(bundle.root), [None] * len(bundle.load_index)
    pending_entries: list[tuple[dict, int, int, np.ndarray]] = []
    batch_number = 0
    try:
        for number, load in enumerate(bundle.load_index, 1):
            row_index = number - 1
            tid = load["trajectory_id"]; blocked = load if load["dropped"] else prior.get(tid)
            if blocked and blocked["dropped"]:
                row_results[row_index] = {"trajectory_id": tid, "dropped": True, "drop_stage": blocked["drop_stage"], "drop_reason": blocked["drop_reason"], "segments": None}
                continue
            try:
                input_values = reader.read(tid)
                starts = _segment_starts(len(input_values), ANALYSIS_SEGMENT_LENGTH, ANALYSIS_SEGMENT_STRIDE)
                gpu_eligible = (
                    executor is not None
                    and np.isfinite(input_values).all()
                    and all(len(input_values[start:min(start + ANALYSIS_SEGMENT_LENGTH, len(input_values))]) == ANALYSIS_SEGMENT_LENGTH for start in starts)
                )
                if not gpu_eligible:
                    segments = _segment_rows(
                        input_values,
                        ANALYSIS_SEGMENT_LENGTH,
                        segment_stride=ANALYSIS_SEGMENT_STRIDE,
                    )
                    row_results[row_index] = _row_result(tid, segments)
                else:
                    state = {
                        "row_index": row_index,
                        "trajectory_id": tid,
                        "values": input_values,
                        "segments": [None] * len(starts),
                        "pending": len(starts),
                        "done": False,
                    }
                    for segment_index, start in enumerate(starts):
                        if state["done"]:
                            break
                        segment_values = input_values[start:start + ANALYSIS_SEGMENT_LENGTH]
                        pending_entries.append((state, segment_index, start, segment_values))
                        if len(pending_entries) >= executor.batch_size:
                            batch_number += 1
                            _flush_cuda_batch(
                                executor,
                                pending_entries,
                                row_results,
                                ANALYSIS_SEGMENT_LENGTH,
                                ANALYSIS_SEGMENT_STRIDE,
                                batch_number,
                            )
                            pending_entries.clear()
            except Exception:
                log.exception("Feature computation failed: trajectory_id=%s", tid)
                row_results[row_index] = {"trajectory_id": tid, "dropped": True, "drop_stage": "feature", "drop_reason": "feature_invalid", "segments": None}
            if number == 1 or number % 1000 == 0:
                log.info(
                    "Feature extraction progress: records=%d/%d gpu_segments=%d gpu_batches=%d pending_segments=%d",
                    number,
                    len(bundle.load_index),
                    executor.processed_segments if executor is not None else 0,
                    executor.batch_count if executor is not None else 0,
                    len(pending_entries),
                )
        if pending_entries and executor is not None:
            batch_number += 1
            _flush_cuda_batch(
                executor,
                pending_entries,
                row_results,
                ANALYSIS_SEGMENT_LENGTH,
                ANALYSIS_SEGMENT_STRIDE,
                batch_number,
            )
            pending_entries.clear()
    finally: reader.close()
    if any(row is None for row in row_results):
        raise RuntimeError("Feature extraction did not produce one result per Load row")
    rows = [row for row in row_results if row is not None]
    write_arrow(target, rows, FEATURE_SCHEMA_V1)
    metadata_payload = {
        "schema_version": FEATURE_METADATA_SCHEMA_VERSION_V1,
        "analysis_segment_length": ANALYSIS_SEGMENT_LENGTH,
        "analysis_segment_stride": ANALYSIS_SEGMENT_STRIDE,
    }
    metadata_target.write_text(json.dumps(metadata_payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    log.info("Features output written: path=%s rows=%d valid=%d dropped=%d metadata=%s", target, len(rows), sum(not row["dropped"] for row in rows), sum(row["dropped"] for row in rows), metadata_target)
    if executor is not None:
        log.info("Feature GPU summary: batches=%d segments=%d", executor.batch_count, executor.processed_segments)
    return target
