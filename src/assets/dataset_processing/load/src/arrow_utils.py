"""Small zero-copy helpers for Arrow list columns used by Load adapters."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa


class ArrowColumnAccessor:
    """Expose Arrow rows without converting an entire RecordBatch to Python."""

    def __init__(self, column: pa.Array) -> None:
        self.column = column
        self.valid = column.is_valid().to_numpy(zero_copy_only=False)
        self.offsets = None
        self.values = None
        value_type = getattr(column.type, "value_type", None)
        nested = (
            value_type is not None
            and (
                pa.types.is_list(value_type)
                or pa.types.is_large_list(value_type)
                or pa.types.is_fixed_size_list(value_type)
            )
        )
        if not nested and (
            pa.types.is_list(column.type) or pa.types.is_large_list(column.type)
        ):
            self.offsets = column.offsets.to_numpy(zero_copy_only=False)
            self.values = column.values.to_numpy(zero_copy_only=False)
        elif not nested and pa.types.is_fixed_size_list(column.type):
            size = int(column.type.list_size)
            self.offsets = np.arange(len(column) + 1, dtype=np.int64) * size
            self.values = column.values.to_numpy(zero_copy_only=False)

    def get(self, index: int) -> Any:
        """Return one row, using a NumPy slice for primitive list columns."""
        if not bool(self.valid[index]):
            return None
        if self.offsets is not None:
            start = int(self.offsets[index])
            stop = int(self.offsets[index + 1])
            return self.values[start:stop]
        return self.column[index].as_py()


def is_sequence_column(field: pa.Field) -> bool:
    """Match the list/tuple fields accepted by the generic Load decoder."""
    type_ = field.type
    return (
        pa.types.is_list(type_)
        or pa.types.is_large_list(type_)
        or pa.types.is_fixed_size_list(type_)
    )
