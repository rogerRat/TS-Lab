"""Explicit conversion of source values to RuntimeDataset series rows.

@scenario: Fetch and load adapters share one deterministic rule for converting
           source arrays into one-dimensional RuntimeDataset input rows.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SHARD_COMPLETE_SUFFIX = ".complete"


BATCH_SIZE = 1024

@dataclass(frozen=True)
class SkippedSeries:
    """Describe a source record excluded during explicit dimensional conversion.

    @input: series_id - source identifier converted to ASCII-safe text by caller.
    @input: shape - observed numeric array shape, or None when unavailable.
    @input: reason - stable machine-readable exclusion reason.
    @output: Immutable skipped-record audit value.
    @scenario: Lets adapters continue processing while reporting unsupported data.
    """

    series_id: str
    shape: tuple[int, ...] | None
    reason: str


def split_univariate_series(
    series_id: Any,
    values: Any,
    preserve_invalid_values: bool = False,
) -> tuple[list[tuple[str, np.ndarray]], SkippedSeries | None]:
    """Validate and return one explicit univariate series.

    @input: series_id - source record identifier; converted to str for output IDs.
    @input: values - numeric one-dimensional array-like data.
    @input: preserve_invalid_values - retain NaN and infinity when true.
    @output: (records, skipped), where records contains one (id, float64 array)
             pair; skipped is populated for non-numeric input.
    @scenario: Enforces that Dataset Adapters explicitly split multivariate data.
    """
    source_id = str(series_id)
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return [], SkippedSeries(source_id, None, "non_numeric_values")

    if array.ndim != 1:
        raise ValueError(
            "series_id %s must contain one-dimensional values; received shape %s"
            % (source_id, tuple(array.shape))
        )
    if not preserve_invalid_values and not np.isfinite(array).all():
        return [], SkippedSeries(
            source_id, tuple(array.shape), "non_finite_values"
        )
    # Keep the NumPy array until Arrow serialization. Converting a long series
    # to a Python float list here creates a second large allocation.
    return [(source_id, array)], None


class SQLiteSeriesReservoir:
    """Disk-backed deterministic reservoir for finite fetch sampling.

    @input: SQLite path, positive capacity, and integer random seed.
    @output: Ordered bounded series slots that can be streamed in batches.
    @scenario: Keep retained series values out of process memory for finite fetches.
    """

    def __init__(self, path: Any, capacity: int, seed: int) -> None:
        if int(capacity) <= 0:
            raise ValueError("reservoir capacity must be positive")
        from pathlib import Path
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.capacity = int(capacity)
        self._rng = np.random.default_rng(int(seed))
        self._connection = sqlite3.connect(str(self.path))
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS slots ("
            "slot INTEGER PRIMARY KEY, series_id TEXT NOT NULL, subset TEXT NOT NULL, "
            "length INTEGER NOT NULL, values_blob BLOB NOT NULL)"
        )
        self._connection.commit()
        self._slot_count = int(self._connection.execute("SELECT COUNT(*) FROM slots").fetchone()[0])

    @property
    def count(self) -> int:
        """Return the number of retained slots."""
        return self._slot_count

    def consider(self, row: dict[str, Any], draw_index: int) -> None:
        """Insert or replace one candidate using seeded reservoir sampling.

        @input: row has series_id, subset, length, and one-dimensional series;
                draw_index is the positive candidate draw number.
        @output: None; SQLite is updated only when the candidate is selected.
        @scenario: Preserve finite reservoir probability while bounding RAM.
        """
        if int(draw_index) <= 0:
            raise ValueError("draw_index must be positive")
        values = np.asarray(row["series"], dtype=np.float64)
        slot = None
        if self._slot_count < self.capacity:
            slot = self._slot_count
            self._slot_count += 1
        else:
            candidate = int(self._rng.integers(0, int(draw_index)))
            if candidate < self.capacity:
                slot = candidate
        if slot is None:
            return
        self._connection.execute(
            "INSERT OR REPLACE INTO slots(slot, series_id, subset, length, values_blob) "
            "VALUES (?, ?, ?, ?, ?)",
            (slot, str(row["series_id"]), str(row["subset"]), int(len(values)), values.tobytes()),
        )
        self._connection.commit()

    def iter_batches(self, batch_size: int = BATCH_SIZE):
        """Yield selected rows in slot order using bounded Python batches.

        @input: Positive batch_size.
        @output: Iterator of row dictionaries with NumPy series arrays.
        @scenario: Stream the final reservoir into the shared Bundle Load writer.
        """
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        rows = []
        cursor = self._connection.execute(
            "SELECT series_id, subset, length, values_blob FROM slots ORDER BY slot"
        )
        for series_id, subset, length, values_blob in cursor:
            rows.append({
                "series_id": series_id,
                "series": np.frombuffer(values_blob, dtype=np.float64).copy(),
                "length": int(length),
                "subset": subset,
            })
            if len(rows) >= int(batch_size):
                yield rows
                rows = []
        if rows:
            yield rows

    def close(self) -> None:
        """Commit and close the SQLite handle while retaining failure state."""
        self._connection.commit()
        self._connection.close()

    def cleanup(self) -> None:
        """Remove the reservoir database after successful final publication."""
        self.close()
        self.path.unlink(missing_ok=True)






# A row-count-only batch can still be too large for high-frequency datasets.
# This is a write threshold, not a data limit: rows are flushed earlier, never
# truncated or discarded.
BATCH_VALUE_LIMIT = 250_000

log = logging.getLogger("fetch")


def is_complete_parquet_shard(path: Any) -> bool:
    """Return whether a Parquet shard and its metadata are valid.

    @input: path - Path-like candidate Parquet shard.
    @output: bool - True only when the completion marker and metadata are valid.
    @scenario: Prevents resuming from a truncated shard.
    """
    from pathlib import Path
    import pyarrow.parquet as pq

    shard = Path(path)
    if not shard.is_file() or not Path(str(shard) + SHARD_COMPLETE_SUFFIX).is_file():
        return False
    try:
        pq.ParquetFile(str(shard)).metadata
    except Exception:
        return False
    return True


def is_valid_parquet_file(path: Any) -> bool:
    """Return whether a finalized Parquet file can be read safely.

    @input: path - Path-like candidate Parquet output.
    @output: bool - True when the file exists and Arrow can read its metadata.
    @scenario: Lets load resume from existing canonical outputs without treating
               a truncated Parquet file as complete.
    """
    from pathlib import Path
    import pyarrow.parquet as pq

    candidate = Path(path)
    if not candidate.is_file():
        return False
    try:
        pq.ParquetFile(str(candidate)).metadata
    except Exception:
        return False
    return True




def merge_parquet_shards(shard_paths: list[Any], output_path: Any) -> int:
    """Merge complete Parquet shards into one atomically replaceable file.

    @input: shard_paths - Ordered complete Parquet shard paths.
             output_path - Final Parquet destination.
    @output: int - Number of rows written to the final file.
    @scenario: Combines independent config outputs without concurrent file writes.
    """
    from pathlib import Path
    import pyarrow.parquet as pq

    paths = [Path(path) for path in shard_paths]
    if not paths or any(not is_complete_parquet_shard(path) for path in paths):
        raise ValueError("Cannot merge incomplete Parquet shards")
    destination = Path(output_path)
    partial = destination.with_suffix(destination.suffix + ".partial")
    writer = None
    row_count = 0
    try:
        for shard in paths:
            parquet_file = pq.ParquetFile(str(shard))
            if writer is None:
                writer = pq.ParquetWriter(
                    str(partial), parquet_file.schema_arrow, compression="zstd"
                )
            for batch in parquet_file.iter_batches(batch_size=8192):
                writer.write_batch(batch)
                row_count += batch.num_rows
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("No rows found in Parquet shards")
    partial.replace(destination)
    return row_count

def log_skipped_series(logger: Any, stage: str, skipped: list[SkippedSeries]) -> None:
    """Log a final summary for records dropped by dimensional conversion.

    @input: logger - standard logging-compatible logger.
    @input: stage - ASCII processing stage label, normally FETCH or LOAD.
    @input: skipped - skipped records accumulated by one adapter invocation.
    @output: None; emits warning records only.
    @scenario: Makes every dimensional exclusion visible without aborting a batch.
    """
    if not skipped:
        return
    logger.warning(
        "%s summary: discarded %d records during explicit univariate splitting.",
        stage,
        len(skipped),
    )
    for item in skipped:
        logger.warning(
            "%s discarded series_id=%s shape=%s reason=%s",
            stage,
            item.series_id,
            item.shape,
            item.reason,
        )


def normalize_load_dataframe(
    dataframe: pd.DataFrame,
    preserve_invalid_values: bool = False,
) -> tuple[pd.DataFrame, list[SkippedSeries]]:
    """Apply explicit univariate splitting to decoded load rows.

    @input: dataframe - table with series_id, series, dataset, and subset columns.
    @input: preserve_invalid_values - retain NaN and infinity when true.
    @output: (normalized_dataframe, skipped_records) with recalculated int32 length.
    @scenario: Lets every loader enforce the same univariate RuntimeDataset boundary.
    """
    records: list[dict[str, Any]] = []
    skipped: list[SkippedSeries] = []
    for row in dataframe.to_dict(orient="records"):
        split_records, skipped_record = split_univariate_series(
            row["series_id"], row["series"], preserve_invalid_values
        )
        if skipped_record is not None:
            skipped.append(skipped_record)
            continue
        for split_id, values in split_records:
            normalized = dict(row)
            normalized["series_id"] = split_id
            normalized["series"] = values
            normalized["length"] = len(values)
            records.append(normalized)
    return pd.DataFrame(records, columns=dataframe.columns), skipped
