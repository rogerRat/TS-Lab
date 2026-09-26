"""Schema and structural validation for the non-compatible ShardDataset layout.

The Load stage is the sole owner of numeric shards.  Dedup and Feature are
Arrow-only, append-only interpretations of the Load trajectory index.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.ipc as ipc



FORMAT_VERSION = "ts3.shard-dataset.v1"
# Batch-level Load identity is kept beside index/meta so protocol-specific
# bundles can safely share immutable numeric shards without changing row schema.
LOAD_METADATA_FILENAME = "metadata.json"
LOAD_METADATA_SCHEMA_VERSION = "ts3.load-identity.v1"
DEFAULT_SHARD_BYTES = 256 * 1024 * 1024
DEFAULT_ARROW_BATCH_ROWS = 4096

LOAD_INDEX_SCHEMA = pa.schema([
    pa.field("trajectory_id", pa.string(), nullable=False),
    pa.field("shard_id", pa.int32(), nullable=False),
    pa.field("value_offset", pa.int64(), nullable=False),
    pa.field("length", pa.int64(), nullable=False),
    pa.field("dropped", pa.bool_(), nullable=False),
    pa.field("drop_stage", pa.string()),
    pa.field("drop_reason", pa.string()),
])
META_SCHEMA = pa.schema([
    pa.field("trajectory_id", pa.string(), nullable=False),
    pa.field("scope", pa.string(), nullable=False),
    pa.field("dataset", pa.string(), nullable=False),
    pa.field("subset", pa.string(), nullable=False),
    pa.field("series_id", pa.string(), nullable=False),
])
DEDUP_SCHEMA = pa.schema([
    pa.field("trajectory_id", pa.string(), nullable=False),
    pa.field("dropped", pa.bool_(), nullable=False),
    pa.field("drop_stage", pa.string()),
    pa.field("drop_reason", pa.string()),
    pa.field("canonical_trajectory_id", pa.string()),
])
FEATURE_VALUES = pa.struct([
    pa.field("trend_strength", pa.float64(), nullable=False),
    pa.field("trend_profile_strength", pa.float64(), nullable=False),
    pa.field("trend_shape_1", pa.float64(), nullable=False),
    pa.field("trend_shape_2", pa.float64(), nullable=False),
    pa.field("trend_shape_3", pa.float64(), nullable=False),
    pa.field("trend_shape_4", pa.float64(), nullable=False),
    pa.field("roughness", pa.float64(), nullable=False),
    pa.field("memory", pa.float64(), nullable=False),
    pa.field("volatility_clustering", pa.float64(), nullable=False),
    pa.field("spectral_entropy", pa.float64(), nullable=False),
    pa.field("spectral_centroid", pa.float64(), nullable=False),
    pa.field("spectral_spread", pa.float64(), nullable=False),
    pa.field("asymmetry", pa.float64(), nullable=False),
    pa.field("tail_weight", pa.float64(), nullable=False),
    pa.field("intermittency", pa.float64(), nullable=False),
    pa.field("change_rate", pa.float64(), nullable=False),
])
TRANSITION_VALUES = pa.struct([
    pa.field(name, pa.float32(), nullable=False)
    for name in (
        "trend_strength", "trend_profile_strength", "trend_shape_1", "trend_shape_2",
        "trend_shape_3", "trend_shape_4", "roughness", "memory",
        "volatility_clustering", "spectral_entropy", "spectral_centroid",
        "spectral_spread", "asymmetry", "tail_weight", "intermittency", "change_rate",
    )
])
ANALYSIS_SEGMENT_RECORD = pa.struct([
    pa.field("segment_index", pa.int64(), nullable=False),
    pa.field("start", pa.int64(), nullable=False),
    pa.field("length", pa.int64(), nullable=False),
    pa.field("feature_values", FEATURE_VALUES),
    pa.field("before_features", TRANSITION_VALUES),
    pa.field("after_features", TRANSITION_VALUES),
    pa.field("change_features", TRANSITION_VALUES),
    pa.field("level_shift", pa.float32()),
    pa.field("scale_change", pa.float32()),
    pa.field("is_near_constant", pa.float32()),
])
ANALYSIS_SEGMENT_RECORD_V1 = pa.struct([
    pa.field("segment_index", pa.int64(), nullable=False),
    pa.field("start", pa.int64(), nullable=False),
    pa.field("length", pa.int64(), nullable=False),
    pa.field("feature_values", FEATURE_VALUES),
])
FEATURE_SCHEMA_V1 = pa.schema([
    pa.field("trajectory_id", pa.string(), nullable=False),
    pa.field("dropped", pa.bool_(), nullable=False),
    pa.field("drop_stage", pa.string()), pa.field("drop_reason", pa.string()),
    pa.field("segments", pa.list_(ANALYSIS_SEGMENT_RECORD_V1)),
])
FEATURE_SCHEMA = pa.schema([
    pa.field("trajectory_id", pa.string(), nullable=False),
    pa.field("dropped", pa.bool_(), nullable=False),
    pa.field("drop_stage", pa.string()), pa.field("drop_reason", pa.string()),
    pa.field("segments", pa.list_(ANALYSIS_SEGMENT_RECORD)),
])
FEATURE_METADATA_FILENAME = "metadata.json"
FEATURE_METADATA_SCHEMA_VERSION_V1 = "ts3.feature-asset.descriptors.v1"
FEATURE_METADATA_SCHEMA_VERSION = "ts3.feature-asset.descriptors.v2"


def read_arrow(path: Path, schema: pa.Schema) -> list[dict[str, Any]]:
    """Read exactly one Arrow IPC file with the required schema."""
    with Path(path).open("rb") as stream:
        reader = ipc.open_file(stream)
        if reader.schema != schema:
            raise ValueError("Arrow schema mismatch: %s" % path)
        return [row for batch in range(reader.num_record_batches)
                for row in reader.get_batch(batch).to_pylist()]


def iter_arrow_rows(
    path: Path,
    schema: pa.Schema,
    columns: tuple[str, ...] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield selected Arrow row fields without materializing the full table."""
    selected = schema.names if columns is None else tuple(columns)
    if any(name not in schema.names for name in selected):
        raise ValueError("Arrow column selection contains an unknown field")
    with Path(path).open("rb") as stream:
        reader = ipc.open_file(stream)
        if reader.schema != schema:
            raise ValueError("Arrow schema mismatch: %s" % path)
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            selected_columns = tuple(batch.column(name) for name in selected)
            for row_index in range(batch.num_rows):
                yield {
                    name: column[row_index].as_py()
                    for name, column in zip(selected, selected_columns)
                }


def iter_arrow_batches(path: Path, schema: pa.Schema) -> Iterator[pa.RecordBatch]:
    """Yield validated IPC record batches without converting cells to Python."""
    with Path(path).open("rb") as stream:
        reader = ipc.open_file(stream)
        if reader.schema != schema:
            raise ValueError("Arrow schema mismatch: %s" % path)
        for batch_index in range(reader.num_record_batches):
            yield reader.get_batch(batch_index)


def write_arrow(
    path: Path,
    rows: list[dict[str, Any]],
    schema: pa.Schema,
    batch_rows: int = DEFAULT_ARROW_BATCH_ROWS,
) -> None:
    """Write rows as bounded IPC batches without changing row semantics."""
    if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows <= 0:
        raise ValueError("batch_rows must be a positive integer")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        with ipc.new_file(stream, schema) as writer:
            for start in range(0, len(rows), batch_rows):
                table = pa.Table.from_pylist(rows[start:start + batch_rows], schema=schema)
                for batch in table.to_batches(max_chunksize=batch_rows):
                    writer.write_batch(batch)


def read_feature_metadata(bundle_root: Path) -> dict[str, Any]:
    """Read and validate feature/metadata.json of a validated Feature Bundle."""
    target = Path(bundle_root).resolve() / "feature" / FEATURE_METADATA_FILENAME
    import json as _json
    try:
        payload = _json.loads(target.read_text(encoding="ascii"))
    except (OSError, ValueError) as error:
        raise ValueError("feature metadata is missing or invalid: %s" % target) from error
    allowed = (
        {"schema_version", "analysis_segment_length"},
        {"schema_version", "analysis_segment_length", "analysis_segment_stride"},
    )
    if not isinstance(payload, dict) or set(payload) not in allowed:
        raise ValueError("feature metadata must contain schema_version and analysis_segment_length")
    schema_version = payload["schema_version"]
    segment_length = payload["analysis_segment_length"]
    if not isinstance(schema_version, str) or not schema_version.isascii():
        raise ValueError("feature metadata schema_version is invalid")
    if isinstance(segment_length, bool) or not isinstance(segment_length, int) or segment_length <= 0:
        raise ValueError("feature metadata analysis_segment_length must be a positive integer")
    segment_stride = payload.get("analysis_segment_stride", segment_length)
    if isinstance(segment_stride, bool) or not isinstance(segment_stride, int) or segment_stride <= 0:
        raise ValueError("feature metadata analysis_segment_stride must be a positive integer")
    return {"schema_version": schema_version, "analysis_segment_length": int(segment_length), "analysis_segment_stride": int(segment_stride)}


def _validate_segment_records(row: dict[str, Any], index_row: dict[str, Any], stage: str, *, transition_required: bool) -> None:
    """Validate one feature row's ordered AnaSeg records against Load geometry.

    Valid rows must carry non-empty ordered segments that cover the full
    trajectory without gaps and hold finite [0,1] feature values. Dropped
    rows must carry no segments. Any invalid segment invalidates the whole
    trajectory, so feature rows never contain a partial segment list.
    """
    trajectory_id = row["trajectory_id"]
    segments = row.get("segments")
    if row["dropped"]:
        if segments is not None:
            raise ValueError("%s dropped row must have null segments: %s" % (stage, trajectory_id))
        return
    if not segments:
        raise ValueError("%s valid row requires non-empty segments: %s" % (stage, trajectory_id))
    trajectory_length = index_row["length"]
    covered_until = 0
    previous_start = -1
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError("%s segment must be a record: %s" % (stage, trajectory_id))
        if segment.get("segment_index") != index:
            raise ValueError("%s segment_index must be ordered from 0: %s" % (stage, trajectory_id))
        start = segment.get("start")
        segment_length = segment.get("length")
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise ValueError("%s segment start is invalid: %s" % (stage, trajectory_id))
        if isinstance(segment_length, bool) or not isinstance(segment_length, int) or segment_length <= 0:
            raise ValueError("%s segment length is invalid: %s" % (stage, trajectory_id))
        if start < previous_start or start > covered_until:
            raise ValueError("%s segments must cover the trajectory without gaps: %s" % (stage, trajectory_id))
        if start + segment_length > trajectory_length:
            raise ValueError("%s segment exceeds trajectory: %s" % (stage, trajectory_id))
        values = segment.get("feature_values")
        if not isinstance(values, dict) or not values:
            raise ValueError("%s segment feature_values are missing: %s" % (stage, trajectory_id))
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not (0.0 <= value <= 1.0)
            for value in values.values()
        ):
            raise ValueError("%s segment feature_values must be finite and in [0,1]: %s" % (stage, trajectory_id))
        if transition_required:
            for field_name in ("before_features", "after_features", "change_features"):
                transition = segment.get(field_name)
                if not isinstance(transition, dict) or set(transition) != set(values):
                    raise ValueError("%s %s is missing or has an invalid shape: %s" % (stage, field_name, trajectory_id))
                if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in transition.values()):
                    raise ValueError("%s %s must be finite: %s" % (stage, field_name, trajectory_id))
            if any(not isinstance(segment.get(name), (int, float)) or isinstance(segment.get(name), bool) or not math.isfinite(segment.get(name)) for name in ("level_shift", "scale_change", "is_near_constant")):
                raise ValueError("%s transition auxiliary values are invalid: %s" % (stage, trajectory_id))
            if float(segment["is_near_constant"]) not in (0.0, 1.0):
                raise ValueError("%s is_near_constant must be 0 or 1: %s" % (stage, trajectory_id))
        covered_until = max(covered_until, start + segment_length)
        previous_start = start
    if covered_until != trajectory_length:
        raise ValueError("%s segments must cover the full trajectory: %s" % (stage, trajectory_id))


@dataclass(frozen=True)
class ShardDatasetBundle:
    """Validated Load shards and optional Arrow-only derivative stages."""

    root: Path
    load_index: tuple[dict[str, Any], ...]
    meta: tuple[dict[str, Any], ...]
    dedup: tuple[dict[str, Any], ...] | None
    feature: tuple[dict[str, Any], ...] | None


def validate_shard_dataset_bundle(root: Path, *, include_feature: bool = True) -> ShardDatasetBundle:
    root = Path(root).resolve()
    load = root / "load"
    shards = load / "shards"
    if not load.is_dir() or not shards.is_dir():
        raise ValueError("ShardDatasetBundle requires load/shards: %s" % root)
    index, meta = read_arrow(load / "index.arrow", LOAD_INDEX_SCHEMA), read_arrow(load / "meta.arrow", META_SCHEMA)
    index_ids, meta_ids = [row["trajectory_id"] for row in index], [row["trajectory_id"] for row in meta]
    if len(index_ids) != len(set(index_ids)) or set(index_ids) != set(meta_ids) or len(meta_ids) != len(set(meta_ids)):
        raise ValueError("load index/meta trajectory identities must be one-to-one")
    expected_roots = {"load", "dedup", "feature"}
    unexpected = {item.name for item in root.iterdir()} - expected_roots
    if unexpected:
        raise ValueError("ShardDatasetBundle has unknown top-level entries: %s" % sorted(unexpected))
    for row in index:
        if row["shard_id"] < 0 or row["value_offset"] < 0 or row["length"] <= 0:
            raise ValueError("invalid Load shard locator")
        shard = shards / ("values_%06d.bin" % row["shard_id"])
        if not shard.is_file():
            raise ValueError("Load shard is missing")
        if (row["value_offset"] + row["length"]) * 8 > shard.stat().st_size:
            raise ValueError("Load shard locator exceeds the binary file")
        if row["dropped"] != (row["drop_stage"] is not None and row["drop_reason"] is not None):
            raise ValueError("Load drop state is inconsistent")
    def optional(stage: str, name: str, schema: pa.Schema):
        directory = root / stage
        if not directory.exists(): return None
        allowed = {name}
        if stage == "feature":
            allowed.add(FEATURE_METADATA_FILENAME)
        if set(item.name for item in directory.iterdir()) != allowed:
            raise ValueError("%s may contain only %s" % (stage, sorted(allowed)))
        if stage == "feature" and not include_feature:
            return None
        transition_required = False
        if stage == "feature":
            metadata = read_feature_metadata(root)
            if metadata["schema_version"] == FEATURE_METADATA_SCHEMA_VERSION:
                schema = FEATURE_SCHEMA
                transition_required = True
            elif metadata["schema_version"] == FEATURE_METADATA_SCHEMA_VERSION_V1:
                schema = FEATURE_SCHEMA_V1
            else:
                raise ValueError("unsupported feature metadata schema: %s" % metadata["schema_version"])
        rows = read_arrow(directory / name, schema)
        if [row["trajectory_id"] for row in rows] != index_ids:
            raise ValueError("%s must have exactly Load index order" % stage)
        for row, index_row in zip(rows, index):
            if row["dropped"] != (row["drop_stage"] is not None and row["drop_reason"] is not None):
                raise ValueError("%s drop state is inconsistent" % stage)
            if stage == "feature":
                _validate_segment_records(row, index_row, stage, transition_required=transition_required)
        return tuple(rows)
    return ShardDatasetBundle(root, tuple(index), tuple(meta), optional("dedup", "dedup.arrow", DEDUP_SCHEMA), optional("feature", "feature.arrow", FEATURE_SCHEMA))
