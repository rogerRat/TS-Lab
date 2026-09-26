"""Dataset Load -- Chronos raw Parquet to RuntimeDataset conversion.

@input: Raw HuggingFace Parquet files produced by assets.dataset_processing.fetch.src.chronos.
@output: Bundle Load rows through the supplied BundleLoadWriter.
@scenario: Own all Chronos schema adaptation after source-only acquisition.
"""

import json
import logging
import math
from pathlib import Path

from assets.dataset_processing.configs.chronos import CHRONOS_SEQUENCE_COLUMNS
from assets.dataset_processing.series import (
    BATCH_SIZE,
    BATCH_VALUE_LIMIT,
    SQLiteSeriesReservoir,
    log_skipped_series,
    split_univariate_series,
)

log = logging.getLogger("load")


def _raw_config_paths(input_dir, subset):
    """Return validated original files for one completed Chronos raw config.

    @input: input_dir is the Chronos source tree; subset is the selected config name.
    @output: list[Path] when the raw completion marker and every source file are valid;
             otherwise None.
    @scenario: Prevents conversion from consuming partial source downloads.
    """
    raw_root = Path(input_dir)
    marker = raw_root / subset / ".complete"
    if not marker.is_file():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="ascii"))
        filenames = payload["files"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(filenames, list) or not filenames:
        return None
    paths = [raw_root / str(filename) for filename in filenames]
    if any(not path.is_file() for path in paths):
        return None
    return paths


def load_chronos(input_dir, runtime_writer, subset, max_series=float("inf"),
                 seed=42, preserve_invalid_values=False):
    """Convert raw Chronos Parquet files directly into RuntimeDataset rows.

    @input: input_dir contains one completed config tree; runtime_writer is the
            destination RuntimeDataset writer; subset selects an explicit field
            mapping; max_series and seed control finite sampling per source;
            preserve_invalid_values explicitly retains NaN and infinity.
    @output: Written row count, or None when raw input is absent.
    @scenario: Preserve Chronos source ordering while avoiding Load-stage files.
    """
    import pyarrow.parquet as pq
    from .arrow_utils import ArrowColumnAccessor

    input_dir = Path(input_dir)
    if not math.isinf(max_series) and int(max_series) <= 0:
        raise ValueError("max_series must be positive or infinity")
    source_paths = _raw_config_paths(input_dir, subset)
    if source_paths is None:
        log.error("Completed Chronos raw input not found for '%s' in %s", subset, input_dir)
        return None
    fields = CHRONOS_SEQUENCE_COLUMNS.get(subset)
    if fields is None:
        raise ValueError("No explicit Chronos field mapping for subset: %s" % subset)

    source_ordinal = 0
    for source_path in source_paths:
        runtime_writer.record_input(source_path)
        writer = runtime_writer
        reservoir = None
        if not math.isinf(max_series):
            reservoir = SQLiteSeriesReservoir(
                runtime_writer.staging / (".%s_reservoir.sqlite3" % source_path.stem),
                int(max_series), int(seed),
            )
        rows = []
        skipped = []
        candidates = 0
        written = 0
        buffered_values = 0
        parquet = pq.ParquetFile(source_path)
        available = set(parquet.schema_arrow.names)
        missing = [name for name in fields if name not in available]
        if missing:
            raise ValueError(
                "Chronos subset %s is missing fields %s" % (subset, missing)
            )
        read_columns = ["id", *fields] if "id" in available else list(fields)
        try:
            for arrow_batch in parquet.iter_batches(
                batch_size=BATCH_SIZE, columns=read_columns
            ):
                id_values = None
                id_index = arrow_batch.schema.get_field_index("id")
                if id_index >= 0:
                    id_values = arrow_batch.column(id_index).to_pylist()
                field_values = {
                    field_name: ArrowColumnAccessor(
                        arrow_batch.column(arrow_batch.schema.get_field_index(field_name))
                    )
                    for field_name in fields
                }
                for index in range(arrow_batch.num_rows):
                    source_id = str(
                        id_values[index]
                        if id_values is not None
                        else "%s_%d" % (subset, source_ordinal)
                    )
                    source_ordinal += 1
                    for field_name in fields:
                        series_id = (
                            source_id
                            if len(fields) == 1
                            else "%s__%s" % (source_id, field_name)
                        )
                        split_records, skipped_record = split_univariate_series(
                            series_id, field_values[field_name].get(index),
                            preserve_invalid_values
                        )
                        if skipped_record is not None:
                            skipped.append(skipped_record)
                            continue
                        for split_id, values in split_records:
                            if len(values) < 10:
                                continue
                            candidates += 1
                            record = {
                                "series_id": split_id,
                                "series": values,
                                "length": len(values),
                                "dataset": "chronos",
                                "subset": subset,
                            }
                            if reservoir is not None:
                                reservoir.consider(record, candidates)
                            else:
                                rows.append(record)
                                buffered_values += int(record["length"])
                                if (
                                    len(rows) >= BATCH_SIZE
                                    or buffered_values >= BATCH_VALUE_LIMIT
                                ):
                                    writer.write_rows(rows)
                                    written += len(rows)
                                    rows.clear()
                                    buffered_values = 0
            if reservoir is not None:
                for batch in reservoir.iter_batches(BATCH_SIZE):
                    for record in batch:
                        record["dataset"] = "chronos"
                    writer.write_rows(batch)
                    written += len(batch)
            elif rows:
                writer.write_rows(rows)
                written += len(rows)
        finally:
            if reservoir is not None:
                reservoir.cleanup()
        log_skipped_series(log, "LOAD", skipped)
        log.info("  Decoded %d Chronos series from %d candidates", written, candidates)
    return runtime_writer.rows_written
