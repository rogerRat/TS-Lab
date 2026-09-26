"""Dataset Load -- Monash dataset parser.

Reads Monash time-series data (either HuggingFace archives or legacy multi-file
archive) and writes RuntimeDataset rows.

@input:  Raw Monash data directory or parquet file.
@output: Bundle Load rows with series_id, values, dataset, and subset.
@scenario: Supports both legacy git clone and new HF streaming data sources.
"""

import json
import logging
import math
import zipfile
from pathlib import Path

# -- Project root for imports ------------------------------------------------
import sys
from configs.paths import PROJECT_ROOT as _PROJECT_ROOT
sys.path.insert(0, str(_PROJECT_ROOT))

from assets.dataset_processing.configs.monash import MONASH_NAME_MAP  # noqa: E402
from assets.dataset_processing.series import BATCH_SIZE, SQLiteSeriesReservoir, log_skipped_series, normalize_load_dataframe, split_univariate_series  # noqa: E402
from assets.dataset_processing.tsf_utils import parse_tsf_record, parse_tsf_values, is_index_like  # noqa: E402
from ts3.spec.load_dataframe import validate_series_rows  # noqa: E402

log = logging.getLogger("load")


# ===========================================================================
# Directory-Level Parser (Legacy git clone mode)
# ===========================================================================

def _read_all_series_from_dir(subset_dir, series_prefix=None):
    """Read all time series from a Monash subset directory.

    Handles .tsf (forecasting competition), .csv, .txt, and .arff formats.

    @input:  subset_dir (Path)        -- directory for a single Monash subset.
             series_prefix (str|None) -- optional naming prefix for series_id.
    @output: tuple (list[str], list[list[float]]) -- (ids, series_values)
    @scenario: Monash subsets use heterogeneous file layouts; this function
               dispatches by file extension and extracts individual series.
    """
    import numpy as np

    ids = []
    values = []

    # Sort for deterministic ordering
    files = sorted(subset_dir.iterdir())

    for fpath in files:
        if fpath.suffix.lower() == ".tsf":
            _load_tsf_file(fpath, ids, values, prefix=series_prefix)
        elif fpath.suffix.lower() == ".arff":
            _load_arff_file(fpath, ids, values, prefix=series_prefix)
        elif fpath.suffix.lower() in (".csv", ".txt"):
            _load_generic_table(fpath, ids, values, prefix=series_prefix)
        elif fpath.suffix.lower() == ".parquet":
            _load_parquet_file(fpath, ids, values, prefix=series_prefix)
        # Skip other file types silently (e.g., .info, .json metadata)

    log.info("  Loaded %d series from %d files in %s", len(ids), len(files), subset_dir.name)
    return ids, values


# ===========================================================================
# File-Level Parsers
# ===========================================================================

def _load_tsf_file(filepath, ids, values, prefix=None):
    """Parse a .tsf file (Monash forecasting competition format).

    Handles two .tsf data line formats:
      1. name:values              -- simple name + comma-separated values
      2. name:timestamp:freq:values -- with start timestamp and frequency prefix

    The key insight: in any .tsf data line, the FIRST colon-delimited segment
    is always the series name. The LAST segment that contains commas is always
    the values. Segments in between (if any) are metadata (timestamp, frequency).

    Also supports missing value markers (? and NaN) which are replaced by float('nan').

    @input:  filepath (Path)   -- path to the .tsf file.
             ids (list[str])   -- accumulator for series identifiers.
             values (list)     -- accumulator for series value lists.
             prefix (str|None)  -- optional naming prefix for series_id.
    @output: None (modifies ids and values in-place).
    @scenario: Monash subsets may contain .tsf files with timestamps and missing values.
    """
    import numpy as np

    with open(filepath, "r") as fh:
        lines = fh.readlines()

    prefix_tag = (prefix + "_") if prefix else ""

    for line in lines:
        line = line.strip()
        if not line or line.startswith("@") or line.startswith("#"):
            continue  # skip header / metadata lines

        if ":" in line:
            # Strategy: find the LAST segment containing a comma = values part.
            # The FIRST segment is always the series name.
            # Everything in between is metadata (timestamp, frequency) -- discarded.
            parts = line.split(":")
            vals_idx = None
            for i in range(len(parts) - 1, -1, -1):
                if "," in parts[i]:
                    vals_idx = i
                    break

            if vals_idx is not None:
                # series name = first segment; values = the segment at vals_idx
                series_name = parts[0].strip()
                vals_str = parts[vals_idx]
            else:
                # No segment has commas -- fallback to simple name:values split
                series_name, _, vals_str = line.partition(":")
                series_name = series_name.strip()

            series_name = prefix_tag + series_name
            vals = parse_tsf_values(vals_str)

            if vals:
                ids.append(series_name)
                values.append(vals)
        else:
            # Single series per file without explicit name
            vals = parse_tsf_values(line)
            if vals:
                ids.append(prefix_tag + filepath.stem)
                values.append(vals)


def _load_arff_file(filepath, ids, values, prefix=None):
    """Parse a .arff file -- extract numeric rows as individual series.

    .arff files have @attribute declarations then @data section.
    Each row after @data is one time point; we treat each attribute column
    (after the first, if it's an index) as an independent series.
    """
    import numpy as np

    with open(filepath, "r") as fh:
        lines = fh.readlines()

    in_data = False
    attrib_names = []
    data_rows = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        if line.lower().startswith("@attribute"):
            parts = line.split()
            if len(parts) >= 2:
                attrib_names.append(parts[1].strip("'\""))
        elif line.lower().startswith("@data"):
            in_data = True
        elif in_data:
            data_rows.append(line)

    if not data_rows:
        return

    # Build numeric matrix: rows = time, cols = variates
    matrix = []
    for row in data_rows:
        try:
            matrix.append([float(x) for x in row.split(",") if x.strip()])
        except ValueError:
            continue

    if not matrix:
        return

    matrix = np.array(matrix, dtype=np.float64)
    n_times, n_cols = matrix.shape

    prefix_tag = (prefix + "_") if prefix else ""
    # If first column looks like an index (all integers, no variance), skip it
    start_col = 1 if (n_cols > 1 and is_index_like(matrix[:, 0])) else 0

    for col_idx in range(start_col, n_cols):
        col_vals = matrix[:, col_idx].tolist()
        if len(col_vals) < 2:
            continue
        col_name = (
            attrib_names[col_idx]
            if col_idx < len(attrib_names)
            else f"col_{col_idx}"
        )
        ids.append(f"{prefix_tag}{filepath.stem}_{col_name}")
        values.append(col_vals)


def _load_generic_table(filepath, ids, values, prefix=None):
    """Parse a .csv or .txt file as a columnar table -- each column is a series.

    Assumes first column may be a date/time index column.
    """
    import numpy as np
    import pandas as pd

    try:
        df = pd.read_csv(filepath, header=None if filepath.suffix == ".txt" else "infer")
    except Exception as exc:
        log.warning("  Failed to read %s: %s", filepath.name, exc)
        return

    if df.empty or df.shape[1] < 1:
        return

    prefix_tag = (prefix + "_") if prefix else ""

    # If first column is string/date, treat as index; remaining cols are series
    start_col = 0
    if df.shape[1] > 1:
        first_col = df.iloc[:, 0]
        try:
            pd.to_numeric(first_col, errors="raise")
        except (ValueError, TypeError):
            start_col = 1  # non-numeric first col -> index, skip it

    for col_idx in range(start_col, df.shape[1]):
        col_vals = pd.to_numeric(df.iloc[:, col_idx], errors="coerce").dropna()
        if len(col_vals) < 2:
            continue
        col_name = str(df.columns[col_idx]) if hasattr(df, "columns") else f"col_{col_idx}"
        ids.append(f"{prefix_tag}{filepath.stem}_{col_name}")
        values.append(col_vals.tolist())


def _load_parquet_file(filepath, ids, values, prefix=None):
    """Parse a .parquet file -- each column is a series."""
    import numpy as np
    import pandas as pd

    try:
        df = pd.read_parquet(filepath)
    except Exception as exc:
        log.warning("  Failed to read %s: %s", filepath.name, exc)
        return

    if df.empty:
        return

    prefix_tag = (prefix + "_") if prefix else ""

    for col_name in df.columns:
        col_vals = df[col_name]
        try:
            numeric = pd.to_numeric(col_vals, errors="coerce").dropna()
        except Exception:
            continue
        if len(numeric) < 2:
            continue
        ids.append(f"{prefix_tag}{filepath.stem}_{col_name}")
        values.append(numeric.tolist())


# ===========================================================================
# Public API
# ===========================================================================

def load_monash(input_dir, subset, runtime_writer,
                 preserve_invalid_values=False):
    """Load a single legacy Monash subset into RuntimeDataset rows.

    @input:  input_dir (Path)  -- directory containing Monash data.
             subset (str)      -- sub-directory name (e.g. "weather") or HF config name.
             runtime_writer (BundleLoadWriter) -- shared Bundle Load destination.
             preserve_invalid_values (bool) -- retain NaN and infinity.
    @output: Appended row count, or None on error.
    @scenario: Supports historical Monash archive inputs without a Load artifact.
    """
    import pandas as pd

    input_dir = Path(input_dir)
    # Legacy git clone data remains supported for historical archive inputs.
    repo_dir = input_dir / "monash_ts_archive"
    if repo_dir.is_dir():
        log.warning(
            "Using legacy git clone data at %s -- this path is DEPRECATED. "
            "Consider re-fetching with: python -m assets.dataset_processing.fetch --source monash ...",
            repo_dir,
        )
        subset_dir = repo_dir / subset
        if not subset_dir.is_dir():
            log.error("Monash subset directory not found: %s", subset_dir)
            return None

        log.info("Loading Monash subset (legacy): %s", subset)
        ids, vals = _read_all_series_from_dir(subset_dir, series_prefix=subset)

        if not ids:
            log.warning("No series found in Monash subset '%s'.", subset)
            return None

        for path in sorted(subset_dir.iterdir()):
            if path.is_file():
                runtime_writer.record_input(path)
        df = pd.DataFrame({
            "series_id": ids, "series": vals, "length": [len(item) for item in vals],
            "dataset": "monash", "subset": subset,
        })
        df, skipped = normalize_load_dataframe(
            df, preserve_invalid_values
        )
        log_skipped_series(log, "LOAD", skipped)
        validate_series_rows(df, preserve_invalid_values)
        runtime_writer.write_rows(df.to_dict(orient="records"))
        log.info("  Decoded %d Monash series", len(df))
        return len(df)

    log.error(
        "No legacy Monash archive found in %s. Run fetch first.",
        input_dir,
    )
    return None


def _raw_monash_zip(input_dir, config):
    """Return one completed original Monash archive for a config.

    @input: input_dir is the Monash source tree; config is a HuggingFace config name.
    @output: Path to the source zip, or None.
    @scenario: Restricts conversion to validated raw fetch output.
    """
    raw_root = Path(input_dir)
    marker = raw_root / ".complete" / (config + ".json")
    if not marker.is_file():
        return None
    try:
        filename = json.loads(marker.read_text(encoding="ascii"))["file"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    path = raw_root / str(filename)
    return path if path.is_file() and zipfile.is_zipfile(path) else None


def load_monash_hf(input_dir, runtime_writer, subset="all",
                   max_series=float("inf"), seed=42,
                   preserve_invalid_values=False):
    """Convert raw Monash zip archives directly into RuntimeDataset rows.

    @input: input_dir contains raw source archives; runtime_writer is the
            RuntimeDataset destination; subset is one config or all; max_series
            and seed control deterministic sampling;
            preserve_invalid_values explicitly retains NaN and infinity.
    @output: Appended row count, or None when no raw archives exist.
    @scenario: Owns TSF parsing, splitting, filtering, and sampling after raw fetch.
    """
    from assets.dataset_processing.configs.monash import MONASH_HF_CONFIGS

    target = MONASH_NAME_MAP.get(subset, subset)
    configs = list(MONASH_HF_CONFIGS if subset == "all" else [target])
    raw_paths = {config: _raw_monash_zip(input_dir, config) for config in configs}
    raw_paths = {config: path for config, path in raw_paths.items() if path is not None}
    if not raw_paths:
        log.error("Completed Monash raw archives not found in %s", input_dir)
        return None
    if not math.isinf(max_series) and int(max_series) <= 0:
        raise ValueError("max_series must be positive or infinity")
    written_total = 0
    for config, zip_path in sorted(raw_paths.items()):
        runtime_writer.record_input(zip_path)
        writer = runtime_writer
        reservoir = None if math.isinf(max_series) else SQLiteSeriesReservoir(
            runtime_writer.staging / (".monash_%s_reservoir.sqlite3" % config),
            int(max_series), int(seed)
        )
        rows = []
        skipped = []
        candidates = 0
        written = 0
        source_ordinal = 0
        try:
            with zipfile.ZipFile(zip_path) as archive:
                tsf_names = sorted(name for name in archive.namelist() if name.lower().endswith(".tsf"))
                if not tsf_names:
                    raise ValueError("Monash raw archive has no TSF file: %s" % zip_path)
                for tsf_name in tsf_names:
                    with archive.open(tsf_name) as handle:
                        for row_index, raw_line in enumerate(handle):
                            line = raw_line.decode("utf-8", errors="replace").strip()
                            if not line or line.startswith("@") or line.startswith("#"):
                                continue
                            source_id, values = parse_tsf_record(line, config, source_ordinal)
                            source_ordinal += 1
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
                                record = {"series_id": series_id, "series": series, "length": len(series), "dataset": "monash", "subset": config}
                                if reservoir is not None:
                                    reservoir.consider(record, candidates)
                                else:
                                    rows.append(record)
                                    if len(rows) >= BATCH_SIZE:
                                        writer.write_rows(rows)
                                        written += len(rows)
                                        rows.clear()
            if reservoir is not None:
                for batch in reservoir.iter_batches(BATCH_SIZE):
                    for record in batch:
                        record["dataset"] = "monash"
                    writer.write_rows(batch)
                    written += len(batch)
            elif rows:
                writer.write_rows(rows)
                written += len(rows)
        finally:
            if reservoir is not None:
                reservoir.cleanup()
        log_skipped_series(log, "LOAD", skipped)
        log.info("  Decoded %d Monash series from %d candidates", written, candidates)
        written_total += written
    return written_total
