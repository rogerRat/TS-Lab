"""Explicit-only legacy ETT CSV decoder."""

import logging
from pathlib import Path

from assets.dataset_processing.series import log_skipped_series, normalize_load_dataframe
from ts3.spec.load_dataframe import validate_series_rows

log = logging.getLogger("load")


def load_ett(input_dir, subset, runtime_writer, preserve_invalid_values=False):
    """Load one explicit legacy ETT CSV into RuntimeDataset rows.

    @input: input_dir contains ``<subset>.csv``; runtime_writer receives rows;
            preserve_invalid_values controls numeric validation.
    @output: Written row count or None when the explicit legacy input is absent/invalid.
    @scenario: Retain access to existing ETT data without selecting it by default.
    @logic: Reads CSV, turns each non-date column into one series, normalizes and writes rows.
    """
    import pandas as pd

    csv_path = Path(input_dir) / (subset + ".csv")
    if not csv_path.is_file():
        log.error("ETT CSV not found: %s", csv_path)
        return None
    try:
        raw = pd.read_csv(csv_path)
    except Exception as exc:
        log.error("Failed to read %s: %s", csv_path, exc)
        return None
    columns = [name for name in raw.columns if name != "date"]
    records = []
    for name in columns:
        values = pd.to_numeric(raw[name], errors="coerce").tolist()
        if len(values) >= 2:
            records.append({"series_id": subset + "_" + name, "series": values,
                            "length": len(values), "dataset": "ett", "subset": subset})
    if not records:
        log.warning("No valid series found in ETT subset '%s'.", subset)
        return None
    frame, skipped = normalize_load_dataframe(pd.DataFrame(records), preserve_invalid_values)
    log_skipped_series(log, "LOAD", skipped)
    validate_series_rows(frame, preserve_invalid_values)
    runtime_writer.record_input(csv_path)
    runtime_writer.write_rows(frame.to_dict(orient="records"))
    return len(frame)
