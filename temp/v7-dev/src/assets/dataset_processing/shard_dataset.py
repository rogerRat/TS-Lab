"""Shared Bundle I/O used by Source and Reference pipelines.

Load owns the only value bytes.  Invalid values are retained and represented as
index decisions, letting later stages evolve without copying the data store.
"""

from __future__ import annotations

import mmap
import os
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import Any, Iterable

import numpy as np

from ts3.spec.files.shard_dataset import (
    DEFAULT_SHARD_BYTES, LOAD_INDEX_SCHEMA, META_SCHEMA, read_arrow, write_arrow,
)


class ShardDatasetWriter:
    """Write one Bundle Load stage with Source-safe invalid-value defaults."""

    def __init__(self, bundle_root: Path, shard_bytes: int = DEFAULT_SHARD_BYTES,
                 *, drop_non_finite: bool = True) -> None:
        if isinstance(shard_bytes, bool) or not isinstance(shard_bytes, int) or shard_bytes < 8:
            raise ValueError("shard_bytes must be an integer of at least eight")
        self.root, self.load = Path(bundle_root), Path(bundle_root) / "load"
        if self.root.exists(): raise FileExistsError("Bundle already exists: %s" % self.root)
        self.shards = self.load / "shards"; self.shards.mkdir(parents=True)
        self.shard_bytes, self.drop_non_finite, self.rows, self.meta = shard_bytes, bool(drop_non_finite), [], []
        self._stream = None; self._shard_id = -1; self._offset = 0

    def _open(self) -> None:
        if self._stream is not None: self._stream.flush(); os.fsync(self._stream.fileno()); self._stream.close()
        self._shard_id += 1; self._offset = 0
        self._stream = (self.shards / ("values_%06d.bin" % self._shard_id)).open("wb")

    def append(self, *, trajectory_id: str, scope: str, dataset: str, subset: str,
               series_id: str, values: Any) -> None:
        if any(not isinstance(item, str) or not item for item in (trajectory_id, scope, dataset, subset, series_id)):
            raise ValueError("trajectory metadata must be non-empty text")
        array = np.asarray(values, dtype="<f8")
        if array.ndim != 1 or len(array) == 0: raise ValueError("trajectory must be a non-empty vector")
        array = np.ascontiguousarray(array); size = array.nbytes
        if self._stream is None or (self._offset and self._offset + size > self.shard_bytes): self._open()
        offset = self._offset // 8; self._stream.write(memoryview(array).cast("B")); self._offset += size
        invalid = self.drop_non_finite and not bool(np.isfinite(array).all())
        self.rows.append({"trajectory_id": trajectory_id, "shard_id": self._shard_id,
            "value_offset": offset, "length": len(array), "dropped": invalid,
            "drop_stage": "load" if invalid else None, "drop_reason": "non_finite" if invalid else None})
        self.meta.append({"trajectory_id": trajectory_id, "scope": scope, "dataset": dataset,
                          "subset": subset, "series_id": series_id})

    def finish(self) -> Path:
        if self._stream is not None: self._stream.flush(); os.fsync(self._stream.fileno()); self._stream.close(); self._stream = None
        if not self.rows: raise ValueError("ShardDataset cannot be empty")
        write_arrow(self.load / "index.arrow", self.rows, LOAD_INDEX_SCHEMA)
        write_arrow(self.load / "meta.arrow", self.meta, META_SCHEMA)
        return self.root


class SourceShardDatasetWriter(ShardDatasetWriter):
    """Source Load writer: reject whole trajectories with non-finite values."""

    def __init__(self, bundle_root: Path, shard_bytes: int = DEFAULT_SHARD_BYTES) -> None:
        super().__init__(bundle_root, shard_bytes, drop_non_finite=True)


class ReferenceShardDatasetWriter(ShardDatasetWriter):
    """Reference Load writer: mark non-finite trajectories at Load as dropped."""

    def __init__(self, bundle_root: Path, shard_bytes: int = DEFAULT_SHARD_BYTES) -> None:
        super().__init__(bundle_root, shard_bytes, drop_non_finite=True)


class ShardDatasetReader:
    """Bounded mmap reader for a single Bundle Load shard pool."""

    def __init__(self, root: Path, capacity: int = 2) -> None:
        self.root, self.capacity, self.cache = Path(root), capacity, OrderedDict()
        self.index = {row["trajectory_id"]: row for row in read_arrow(self.root / "load/index.arrow", LOAD_INDEX_SCHEMA)}
        self._lock = RLock()

    def read_window_into(self, trajectory_id: str, start: int, output: np.ndarray) -> None:
        """Copy one locator-sized range directly into a supplied float64 row."""
        row = self.index[trajectory_id]
        if not isinstance(output, np.ndarray) or output.dtype != np.float64 or output.ndim != 1 or isinstance(start, bool) or not isinstance(start, int) or start < 0 or start + len(output) > int(row["length"]):
            raise ValueError("window bounds or output are invalid")
        key = int(row["shard_id"])
        with self._lock:
            if key not in self.cache:
                stream = (self.root / "load/shards" / ("values_%06d.bin" % key)).open("rb")
                self.cache[key] = [stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ), 0]
            entry = self.cache.pop(key); entry[2] += 1; self.cache[key] = entry
        try:
            mapped = entry[1]
            begin = (int(row["value_offset"]) + start) * 8
            end = begin + len(output) * 8
            np.copyto(output, np.frombuffer(memoryview(mapped)[begin:end], dtype="<f8"))
        finally:
            with self._lock:
                self.cache[key][2] -= 1
                while len(self.cache) > self.capacity:
                    evict_key, evicted = next(iter(self.cache.items()))
                    if evicted[2]: break
                    self.cache.pop(evict_key)
                    evicted[1].close(); evicted[0].close()

    def read(self, trajectory_id: str) -> np.ndarray:
        row = self.index[trajectory_id]
        result = np.empty(int(row["length"]), dtype="<f8")
        self.read_window_into(trajectory_id, 0, result)
        result.flags.writeable = False
        return result

    def close(self) -> None:
        with self._lock:
            if any(entry[2] for entry in self.cache.values()):
                raise RuntimeError("cannot close a reader with active mappings")
            for stream, mapped, _ in self.cache.values(): mapped.close(); stream.close()
            self.cache.clear()
