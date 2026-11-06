"""Definition and validation of the temporary Load-stage DataFrame.

This module defines the in-memory tabular structure produced while converting
raw data into RuntimeDataset artifacts. It is not the RuntimeDataset schema;
it validates transient decoded series rows before publication.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd


SERIES_COLUMNS = ("series_id", "series", "length", "dataset", "subset")
FEATURE_INVALID_REASON_COLUMN = "invalid_reason"
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
JOIN_KEYS = ("dataset", "subset", "series_id") # Superkey for global search.


class DataContractError(ValueError):
    """Raised when a TSL canonical table violates its data contract.

    @input: Error message describing the invalid table or row.
    @output: ValueError-compatible exception instance.
    @scenario: Raised by public contract validation and join functions.
    """


def _require_dataframe(value: Any, name: str) -> pd.DataFrame:
    """Require a pandas DataFrame input.

    @input: value -- object expected to be a pandas DataFrame.
    @input: name -- ASCII label used in the error message.
    @output: The original DataFrame.
    @scenario: Shared input check for all public table operations.
    """
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    return value


def _require_columns(df: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    """Check that a table contains all required columns.

    @input: df -- pandas DataFrame to inspect.
    @input: required -- tuple of required column names.
    @input: name -- ASCII table label used in the error message.
    @output: None on success; raises DataContractError on failure.
    @scenario: Prevents later operations from relying on missing schema fields.
    """
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise DataContractError(f"{name} is missing required columns: {missing}")


def _validate_text_column(df: pd.DataFrame, column: str, name: str) -> None:
    """Validate a required non-empty string column.

    @input: df -- pandas DataFrame containing the column.
    @input: column -- column name to validate.
    @input: name -- ASCII table label used in the error message.
    @output: None on success; raises DataContractError on failure.
    @scenario: Validates identifiers and source scope fields.
    """
    for row_number, value in df[column].items():
        if not isinstance(value, str) or not value.strip():
            raise DataContractError(
                f"{name}.{column} row {row_number} must be a non-empty string"
            )


def _validate_series_values(
    df: pd.DataFrame, preserve_invalid_values: bool = False
) -> None:
    """Validate list-like series values and their declared lengths.

    @input: df -- canonical series DataFrame with series and length columns.
    @input: preserve_invalid_values -- allow NaN and infinity when true.
    @output: None on success; raises DataContractError on failure.
    @scenario: Ensures each row is one variable-length numeric time series.
    @logic: Preserve the generic iterable validation path, while using NumPy's
            vectorized finite-value check when a row is already a one-dimensional
            numeric ndarray supplied by the approved Parquet reader.
    """
    for row_number, (values, declared_length) in enumerate(
        zip(df["series"].tolist(), df["length"].tolist())
    ):
        if isinstance(declared_length, (bool, np.bool_)) or not isinstance(
            declared_length, (int, np.integer)
        ):
            raise DataContractError(
                f"series.length row {row_number} must be an integer"
            )
        if declared_length < 0:
            raise DataContractError(
                f"series.length row {row_number} must be non-negative"
            )
        if isinstance(values, (str, bytes, Mapping)) or not hasattr(values, "__iter__"):
            raise DataContractError(
                f"series.series row {row_number} must be a list-like numeric sequence"
            )
        if isinstance(values, np.ndarray) and values.ndim == 1 and values.dtype.kind in ("i", "u", "f"):
            if len(values) != declared_length:
                raise DataContractError(
                    f"series row {row_number} length mismatch: "
                    f"declared {declared_length}, actual {len(values)}"
                )
            if not preserve_invalid_values and not np.isfinite(values).all():
                raise DataContractError(
                    f"series.series row {row_number} contains non-finite values"
                )
            continue
        values_list = list(values)
        if len(values_list) != declared_length:
            raise DataContractError(
                f"series row {row_number} length mismatch: "
                f"declared {declared_length}, actual {len(values_list)}"
            )
        for value in values_list:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise DataContractError(
                    f"series.series row {row_number} contains a non-numeric value"
                )
            if not preserve_invalid_values and not np.isfinite(value):
                raise DataContractError(
                    f"series.series row {row_number} contains non-finite values"
                )


def _validate_unique_keys(df: pd.DataFrame, name: str) -> None:
    """Require one row per canonical composite key.

    @input: df -- table containing JOIN_KEYS.
    @input: name -- ASCII table label used in the error message.
    @output: None on success; raises DataContractError for duplicates.
    @scenario: Prevents many-to-many joins and ambiguous feature ownership.
    """
    duplicate_mask = df.duplicated(list(JOIN_KEYS), keep=False)
    if duplicate_mask.any():
        duplicate_keys = df.loc[duplicate_mask, list(JOIN_KEYS)].drop_duplicates()
        records = duplicate_keys.to_dict(orient="records")
        raise DataContractError(f"{name} contains duplicate composite keys: {records}")


def validate_series_rows(
    df: pd.DataFrame, preserve_invalid_values: bool = False
) -> None:
    """Validate the canonical series table schema and row invariants.

    @input: df -- pandas DataFrame with series_id, series, length, dataset,
             and subset columns.
    @input: preserve_invalid_values -- allow NaN and infinity when true.
    @output: None on success; raises DataContractError on any violation.
    @scenario: Validates decoded series rows before RuntimeDataset publication.
    """
    df = _require_dataframe(df, "series_df")
    _require_columns(df, SERIES_COLUMNS, "series_df")
    for column in ("series_id", "dataset", "subset"):
        _validate_text_column(df, column, "series_df")
    _validate_series_values(df, preserve_invalid_values)
    _validate_unique_keys(df, "series_df")
