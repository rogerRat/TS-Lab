"""Shared parsing helpers for Monash TSF source records.

@scenario: Let Fetch validate raw TSF records and Load convert them without
           either stage importing the other.
"""

import numpy as np


MISSING_TOKENS = frozenset({"?", "NaN", "nan", "NA", "na"})


def parse_tsf_values(vals_str):
    """Parse comma-separated TSF values with missing-value support.

    @input: vals_str is a string containing comma-separated source values.
    @output: list[float] with recognized missing tokens represented by NaN.
    @scenario: Decode TSF records consistently in Fetch validation and Load.
    """
    values = []
    for part in vals_str.split(","):
        part = part.strip()
        if not part:
            continue
        if part in MISSING_TOKENS:
            values.append(float("nan"))
            continue
        try:
            values.append(float(part))
        except ValueError:
            values.append(float("nan"))
    return values


def parse_tsf_record(line, config, row_index):
    """Convert one raw TSF data line to its source identifier and values.

    @input: line is a non-header TSF text line; config and row_index identify a fallback id.
    @output: Tuple of string series identifier and parsed float values.
    @scenario: Let Fetch and Load share raw-record parsing without stage coupling.
    """
    if ":" not in line:
        return "%s_%d" % (config, row_index), parse_tsf_values(line)
    parts = line.split(":")
    value_part = next((part for part in reversed(parts) if "," in part), parts[-1])
    return parts[0].strip(), parse_tsf_values(value_part)


def is_index_like(arr):
    """Return whether one integer array resembles a sequential index.

    @input: arr is a NumPy-compatible numeric one-dimensional array.
    @output: bool indicating a monotonic near-unit-step integer index.
    @scenario: Load excludes source table index columns from time-series values.
    """
    if np.issubdtype(arr.dtype, np.integer):
        differences = np.diff(arr)
        return bool(np.all(differences >= 0) and np.all(differences <= 2))
    return False
