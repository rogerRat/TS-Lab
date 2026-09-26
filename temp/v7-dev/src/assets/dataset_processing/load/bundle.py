"""Adapters that feed existing decoders into a ShardDatasetBundle."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import quote

from assets.dataset_processing.load_reuse import write_load_metadata
from assets.dataset_processing.shard_dataset import ReferenceShardDatasetWriter, SourceShardDatasetWriter


log = logging.getLogger("load")


class BundleLoadWriter:
    """Single authoritative Load writer shared by Source and Reference decoders."""

    def __init__(self, bundle_root: Path, *, scope: str, shard_bytes: int,
                 load_metadata: Mapping[str, Any] | None = None) -> None:
        self.scope = str(scope)
        writer_class = {"source": SourceShardDatasetWriter, "reference": ReferenceShardDatasetWriter}.get(self.scope)
        if writer_class is None:
            raise ValueError("scope must be source or reference")
        self.writer = writer_class(Path(bundle_root), shard_bytes=shard_bytes)
        self.load_metadata = dict(load_metadata) if load_metadata is not None else None
        self.rows_written = 0
        self.duplicate_rows = 0
        self.source_inputs: list[dict[str, str]] = []
        self._seen: set[str] = set()

    def record_input(self, path: Path, role: str = "raw_source") -> None:
        # Kept for decoder reuse.  Fetch files remain the provenance source; no
        # derived manifest is persisted in the bundle.
        self.source_inputs.append({"path": str(Path(path).resolve()), "role": str(role)})

    def write_rows(self, rows: Iterable[Mapping[str, Any]], *, dataset: str | None = None,
                   subset: str | None = None) -> None:
        for row in rows:
            row_dataset = str(dataset if dataset is not None else row["dataset"])
            row_subset = str(subset if subset is not None else row["subset"])
            series_id = str(row["series_id"])
            trajectory_id = "/".join(
                quote(part, safe="-_.") for part in (self.scope, row_dataset, row_subset, series_id)
            )
            if trajectory_id in self._seen:
                self.duplicate_rows += 1
                continue
            self._seen.add(trajectory_id)
            values = row.get("values", row.get("series"))
            if values is None:
                raise ValueError("Load row is missing values")
            self.writer.append(
                trajectory_id=trajectory_id,
                scope=self.scope,
                dataset=row_dataset,
                subset=row_subset,
                series_id=series_id,
                values=values,
            )
            self.rows_written += 1

    def finalize(self) -> Path:
        path = self.writer.finish()
        if self.load_metadata is not None:
            metadata_path = write_load_metadata(path, self.load_metadata)
            log.info("LOAD identity metadata written: %s", metadata_path)
        else:
            log.warning(
                "LOAD identity metadata unavailable; bin-shard reuse is disabled for %s",
                path,
            )
        if self.duplicate_rows:
            log.warning(
                "LOAD skipped %d duplicate %s trajectory identities",
                self.duplicate_rows, self.scope,
            )
        return path
