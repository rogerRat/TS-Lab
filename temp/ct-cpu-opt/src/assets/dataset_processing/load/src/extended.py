"""Load adapters for the additional heterogeneous time-series corpora."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from assets.dataset_processing.fetch.src.hub_files import _marker
from assets.dataset_processing.series import (
    BATCH_SIZE,
    SQLiteSeriesReservoir,
    log_skipped_series,
    split_univariate_series,
)

log = logging.getLogger("load")


def _raw_files(input_dir: Path, subset: str) -> list[Path] | None:
    """Return validated raw files recorded for one extended selection.

    @input: input_dir is an extended raw source root; subset is the explicit
            selection path used by Fetch.
    @output: Sorted raw file paths, or None when the marker is incomplete.
    @scenario: Load consumes only atomically completed Fetch selections.
    """
    marker = _marker(Path(input_dir), subset)
    if not marker.is_file():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="ascii"))
        files = payload["files"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    paths = [Path(input_dir) / str(name) for name in files] if isinstance(files, list) else []
    return paths if paths and all(path.is_file() for path in paths) else None


def _iter_batches(path: Path):
    """Yield bounded Arrow batches without materializing Python row dictionaries."""
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq

    if path.suffix.lower() == ".parquet":
        yield from pq.ParquetFile(path).iter_batches(batch_size=4096)
        return
    source = pa.memory_map(str(path), "r")
    try:
        try:
            reader = ipc.open_stream(source)
            yield from reader
        except pa.ArrowInvalid:
            source.seek(0)
            reader = ipc.open_file(source)
            for index in range(reader.num_record_batches):
                yield reader.get_batch(index)
    finally:
        source.close()


def _iter_rows(path: Path):
    """Yield raw records from one Arrow or Parquet file."""
    for batch in _iter_batches(path):
        yield from batch.to_pylist()


def _numeric_sequence_fields(row: dict, dataset: str) -> list[tuple[str, object]]:
    """Return explicit numeric sequence fields from one source record.

    @input: row is one decoded source record; dataset identifies its adapter.
    @output: Ordered ``(field_name, values)`` pairs.
    @scenario: Keeps metadata, labels, and timestamps out of Canonical values.
    """
    excluded = {
        "id", "item_id", "start", "freq", "date", "date_time",
        "timestamp", "time", "label", "labels", "target_class",
        "_original_filename",
    }
    import numpy as np

    result = []
    for name, value in row.items():
        if name in excluded or value is None:
            continue
        if isinstance(value, (list, tuple)):
            array = np.asarray(value)
            if array.ndim == 1:
                result.append((str(name), value))
            elif array.ndim == 2:
                # The raw record remains one indivisible multivariate
                # trajectory during Fetch.  Load explicitly maps its outer
                # variable axis to independent Canonical rows.
                for index, variable in enumerate(array):
                    result.append(("%s_%d" % (name, index), variable.tolist()))
            else:
                raise ValueError(
                    "%s record field %s must be one- or two-dimensional; shape=%s"
                    % (dataset, name, tuple(array.shape))
                )
    if not result:
        raise ValueError("%s record has no explicit sequence field; fields=%s" % (dataset, sorted(row)))
    return result


def _write_records(records, runtime_writer, dataset: str, subset: str,
                   max_series: float, seed: int,
                   source_path: Path | None = None,
                   preserve_invalid_values: bool = False):
    """Convert decoded source records directly into RuntimeDataset rows.

    @input: records yields ``(trajectory_id, field_name, values)`` tuples;
            runtime_writer is the RuntimeDataset destination; dataset and subset
            label the source; max_series and seed optionally bound deterministic
            Load-stage reservoir sampling;
            preserve_invalid_values explicitly retains NaN and infinity.
    @output: Number of rows appended to runtime_writer.
    @scenario: Shared bounded decoder for all extended Load adapters.
    """
    writer = runtime_writer
    reservoir = None if math.isinf(max_series) else SQLiteSeriesReservoir(
        runtime_writer.staging / (".%s_reservoir.sqlite3" % (source_path.stem if source_path else dataset)),
        int(max_series), int(seed)
    )
    rows = []
    skipped = []
    candidates = 0
    written = 0
    try:
        for trajectory_id, field_name, values in records:
            series_id = str(trajectory_id)
            if field_name:
                series_id += "__" + str(field_name)
            split_records, skipped_record = split_univariate_series(
                series_id, values, preserve_invalid_values
            )
            if skipped_record is not None:
                skipped.append(skipped_record)
                continue
            for split_id, series in split_records:
                if len(series) < 10:
                    continue
                candidates += 1
                row = {"series_id": split_id, "series": series,
                       "length": len(series), "dataset": dataset, "subset": subset}
                if reservoir is not None:
                    reservoir.consider(row, candidates)
                else:
                    rows.append(row)
                    if len(rows) >= BATCH_SIZE:
                        writer.write_rows(rows)
                        written += len(rows)
                        rows.clear()
        if reservoir is not None:
            for batch in reservoir.iter_batches(BATCH_SIZE):
                for row in batch:
                    row["dataset"] = dataset
                    row["subset"] = subset
                writer.write_rows(batch)
                written += len(batch)
        elif rows:
            writer.write_rows(rows)
            written += len(rows)
    finally:
        if reservoir is not None:
            reservoir.cleanup()
    log_skipped_series(log, "LOAD", skipped)
    log.info("Decoded %d %s series from %d candidates", written, dataset, candidates)
    return written


def load_adapter(input_dir, runtime_writer, dataset: str, subset: str, records,
                 max_series=float("inf"), seed=42,
                 preserve_invalid_values=False):
    """Load one source adapter selection directly into RuntimeDataset.

    @input: input_dir is the raw source root; runtime_writer is the destination;
            dataset is one extended source key; subset is the explicit selection;
            max_series and seed control deterministic Load sampling;
            preserve_invalid_values explicitly retains NaN and infinity.
    @output: Appended row count, or None when raw input is absent.
    @scenario: Converts heterogeneous corpora without changing Fetch files.
    """
    paths = _raw_files(Path(input_dir), subset)
    if paths is None:
        log.error("Completed extended raw input not found for %s/%s in %s", dataset, subset, input_dir)
        return None
    written = 0
    for path in paths:
        runtime_writer.record_input(path)
        # The source selection may contain multiple files; write each file's
        # records independently so source order remains deterministic.
        def file_records():
            yield from records(path)
        written += _write_records(
            file_records(), runtime_writer, dataset, subset, max_series, seed,
            path, preserve_invalid_values,
        )
    return written


def _generic_records(path: Path, dataset: str):
    """Yield generic Arrow or Parquet records using columnar list accessors."""
    import numpy as np
    import pyarrow as pa
    from .arrow_utils import ArrowColumnAccessor, is_sequence_column

    excluded = {
        "id", "item_id", "start", "freq", "date", "date_time",
        "timestamp", "time", "label", "labels", "target_class",
        "_original_filename",
    }
    ordinal = 0
    for batch in _iter_batches(path):
        names = batch.schema.names
        if "item_id" in names:
            id_values = batch.column(batch.schema.get_field_index("item_id")).to_pylist()
        elif "id" in names:
            id_values = batch.column(batch.schema.get_field_index("id")).to_pylist()
        elif "_original_filename" in names:
            id_values = batch.column(batch.schema.get_field_index("_original_filename")).to_pylist()
        else:
            id_values = None
        sequence_names = [
            field.name for field in batch.schema
            if field.name not in excluded and is_sequence_column(field)
        ]
        accessors = {
            name: ArrowColumnAccessor(batch.column(batch.schema.get_field_index(name)))
            for name in sequence_names
        }
        for index in range(batch.num_rows):
            trajectory_id = (
                id_values[index]
                if id_values is not None
                else "%s_%d" % (path.stem, ordinal)
            )
            ordinal += 1
            fields = []
            for name in sequence_names:
                value = accessors[name].get(index)
                if value is None:
                    continue
                array = np.asarray(value)
                if array.ndim == 1:
                    fields.append((name, value))
                elif array.ndim == 2:
                    fields.extend(
                        ("%s_%d" % (name, dimension), variable.tolist())
                        for dimension, variable in enumerate(array)
                    )
                else:
                    raise ValueError(
                        "%s record field %s must be one- or two-dimensional; shape=%s"
                        % (dataset, name, tuple(array.shape))
                    )
            if not fields:
                raise ValueError(
                    "%s record has no explicit sequence field; fields=%s"
                    % (dataset, sorted(names))
                )
            use_name = len(fields) > 1
            for name, values in fields:
                yield trajectory_id, name if use_name else "", values


def _iter_contiguous_groups(parquet, group_key: str):
    """Yield one bounded contiguous source trajectory at a time.

    @input: parquet is a pyarrow.parquet.ParquetFile; group_key is the source
            trajectory identifier column.
    @output: Iterator of row dictionaries for one complete contiguous group.
    @scenario: Streams Quito/RMISC Load without materializing a whole Parquet file.
               The upstream corpora store each source trajectory contiguously.
    """
    current_key = object()
    group = []
    for batch in parquet.iter_batches(batch_size=4096):
        for row in batch.to_pylist():
            key = row[group_key]
            if group and key != current_key:
                yield group
                group = []
            current_key = key
            group.append(row)
    if group:
        yield group
