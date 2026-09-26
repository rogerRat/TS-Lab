"""Dataset Load -- UTSD raw Arrow to RuntimeDataset conversion.

@input: Original UTSD Arrow shards produced by fetch.
@output: Bundle Load rows with one univariate series per row.
@scenario: Own UTSD Arrow decoding and sampling after fetch-only acquisition.
"""

import json
import logging
import math
from pathlib import Path

from assets.dataset_processing.series import BATCH_SIZE, SQLiteSeriesReservoir, log_skipped_series, split_univariate_series
from .arrow_utils import ArrowColumnAccessor

log = logging.getLogger("load")


def _raw_utsd_paths(input_dir, subset):
    """Return validated raw UTSD Arrow paths for one completed config.

    @input: input_dir is the UTSD source tree; subset is a UTSD config name.
    @output: List of source Arrow paths, or None.
    @scenario: Avoids converting incomplete raw fetch output.
    """
    raw_root = Path(input_dir)
    marker = raw_root / subset / ".complete"
    if not marker.is_file():
        return None
    try:
        filenames = json.loads(marker.read_text(encoding="ascii"))["files"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    paths = [raw_root / str(name) for name in filenames] if isinstance(filenames, list) else []
    return paths if paths and all(path.is_file() and path.stat().st_size > 0 for path in paths) else None


def _arrow_batches(path):
    """Yield record batches from one original Arrow stream or file.

    @input: path is one UTSD upstream Arrow shard.
    @output: Iterator of pyarrow.RecordBatch values.
    @scenario: Supports the Hub's native Arrow IPC serialization without conversion.
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc

    source = pa.memory_map(str(path), "r")
    try:
        try:
            reader = ipc.open_stream(source)
            for batch in reader:
                yield batch
        except pa.ArrowInvalid:
            source.seek(0)
            reader = ipc.open_file(source)
            for index in range(reader.num_record_batches):
                yield reader.get_batch(index)
    finally:
        source.close()


def load_utsd(input_dir, runtime_writer, subset, max_series=float("inf"),
              seed=42, preserve_invalid_values=False):
    """Convert each raw UTSD Arrow shard directly into RuntimeDataset rows.

    @input: input_dir contains one completed UTSD config tree; runtime_writer is
            the RuntimeDataset destination; max_series and seed control finite
            sampling per source shard; preserve_invalid_values explicitly retains
            NaN and infinity.
    @output: Written row count, or None when raw input is absent.
    @scenario: Preserve UTSD source ordering without a Load artifact.
    """
    input_dir = Path(input_dir)
    if not math.isinf(max_series) and int(max_series) <= 0:
        raise ValueError("max_series must be positive or infinity")
    source_paths = _raw_utsd_paths(input_dir, subset)
    if source_paths is None:
        log.error("Completed UTSD raw input not found for '%s' in %s", subset, input_dir)
        return None
    source_ordinal = 0
    seen_item_ids = set()
    duplicate_item_ids = 0
    for source_path in source_paths:
        runtime_writer.record_input(source_path)
        writer = runtime_writer
        reservoir = None if math.isinf(max_series) else SQLiteSeriesReservoir(
            runtime_writer.staging / (".%s_reservoir.sqlite3" % source_path.stem), int(max_series), int(seed)
        )
        rows = []
        skipped = []
        candidates = 0
        written = 0
        try:
            for batch in _arrow_batches(source_path):
                target_index = batch.schema.get_field_index("target")
                target_values = ArrowColumnAccessor(batch.column(target_index))
                id_index = batch.schema.get_field_index("item_id")
                id_values = (
                    batch.column(id_index).to_pylist() if id_index >= 0 else None
                )
                for index in range(batch.num_rows):
                    values = target_values.get(index)
                    if values is None:
                        raise ValueError("UTSD raw record missing target in %s" % source_path)
                    source_id = (
                        id_values[index]
                        if id_values is not None
                        else "%s_%d" % (subset, source_ordinal)
                    )
                    source_ordinal += 1
                    if id_values is not None and source_id is not None:
                        item_key = str(source_id)
                        if item_key in seen_item_ids:
                            duplicate_item_ids += 1
                            continue
                        seen_item_ids.add(item_key)
                    split_records, skipped_record = split_univariate_series(
                        source_id, values, preserve_invalid_values
                    )
                    if skipped_record is not None:
                        skipped.append(skipped_record)
                        continue
                    for series_id, series in split_records:
                        if len(series) < 10:
                            continue
                        candidates += 1
                        record = {"series_id": series_id, "series": series, "length": len(series), "dataset": "utsd", "subset": subset}
                        if reservoir is not None:
                            reservoir.consider(record, candidates)
                        else:
                            rows.append(record)
                            if len(rows) >= BATCH_SIZE:
                                writer.write_rows(rows)
                                written += len(rows)
                                rows.clear()
            if reservoir is not None:
                for selected_batch in reservoir.iter_batches(BATCH_SIZE):
                    for record in selected_batch:
                        record["dataset"] = "utsd"
                    writer.write_rows(selected_batch)
                    written += len(selected_batch)
            elif rows:
                writer.write_rows(rows)
                written += len(rows)
        finally:
            if reservoir is not None:
                reservoir.cleanup()
        log_skipped_series(log, "LOAD", skipped)
        log.info("  Decoded %d UTSD series from %d candidates", written, candidates)
    if duplicate_item_ids:
        log.warning(
            "LOAD skipped %d duplicate UTSD item_id records for %s",
            duplicate_item_ids, subset,
        )
    return runtime_writer.rows_written
