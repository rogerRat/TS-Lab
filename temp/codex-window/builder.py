"""Build flat trajectory-level runtime datasets from direct Catalog V6 Bundle records."""

from __future__ import annotations

import logging

from ts3.spec.catalogs.models import Catalog, CatalogTrajectoryRecord, ReferenceCatalog, SourceCatalog
from ts3.spec.catalogs.reader import CatalogReader
from ts3.spec.models.data_model import RuntimeReferenceDataset, RuntimeSourceDataset, Trajectory, TrajectoryLocator, WindowSpecification
from ts3.core.runtime.windowing import legal_start_count

log = logging.getLogger("train")

TRAJECTORY_QUALITY_MODES = frozenset(("length", "count"))


def _quality_by_id(trajectories: list[Trajectory], trajectory_quality: str) -> dict[str, float]:
    if trajectory_quality not in TRAJECTORY_QUALITY_MODES:
        raise ValueError("trajectory_quality must be one of %s" % sorted(TRAJECTORY_QUALITY_MODES))
    return {
        item.catalog_trajectory_id: float(item.length if trajectory_quality == "length" else 1)
        for item in trajectories
    }


def _normalized_quality_mass(quality_by_id: dict[str, float]) -> dict[str, float]:
    total_quality = sum(quality_by_id.values())
    if total_quality <= 0.0:
        raise ValueError("trajectory quality must have positive total mass")
    return {trajectory_id: quality / total_quality for trajectory_id, quality in quality_by_id.items()}


class RuntimeDatasetBuilder:
    def __init__(self, catalog_reader: CatalogReader, catalog: Catalog) -> None:
        if not isinstance(catalog_reader, CatalogReader) or not isinstance(catalog, Catalog): raise TypeError("catalog_reader and catalog are required")
        self.catalog_reader, self.catalog = catalog_reader, catalog

    @staticmethod
    def _locator(record: CatalogTrajectoryRecord) -> TrajectoryLocator:
        return TrajectoryLocator(record.catalog_trajectory_id, record.bundle_directory, record.shard_id, record.value_offset, record.length, record.dataset, record.subset, record.series_id)


class RuntimeSourceDatasetBuilder(RuntimeDatasetBuilder):
    def __init__(self, catalog_reader: CatalogReader, catalog: SourceCatalog, window_spec: WindowSpecification, trajectory_quality: str) -> None:
        super().__init__(catalog_reader, catalog)
        if not isinstance(catalog, SourceCatalog) or not isinstance(window_spec, WindowSpecification): raise TypeError("SourceCatalog and WindowSpecification are required")
        if trajectory_quality not in TRAJECTORY_QUALITY_MODES: raise ValueError("trajectory_quality must be one of %s" % sorted(TRAJECTORY_QUALITY_MODES))
        self.window_spec = window_spec
        self.trajectory_quality = trajectory_quality

    def _is_length_eligible(self, length: int) -> bool:
        return legal_start_count(length, self.window_spec) > 0

    def build(self) -> RuntimeSourceDataset:
        total = len(self.catalog.records_data); trajectories: list[Trajectory] = []; audit = {"dropped_count": 0, "window_ineligible_count": 0, "eligible_trajectory_count": 0}; log.info("Source build started: records=%d", total)
        processed = 0
        for record in self.catalog_reader.records(self.catalog):
            processed += 1
            if record.dropped:
                audit["dropped_count"] += 1; continue
            if not self._is_length_eligible(record.length):
                audit["window_ineligible_count"] += 1; continue
            trajectories.append(Trajectory(record.catalog_trajectory_id, self._locator(record), record.length))
            audit["eligible_trajectory_count"] += 1
            if processed == 1 or processed % 10000 == 0 or processed == total:
                log.info("Source construction processed %d/%d records (%.1f%%); eligible %d, dropped %d, window-ineligible %d", processed, total, 100.0 * processed / total, audit["eligible_trajectory_count"], audit["dropped_count"], audit["window_ineligible_count"])
        if not trajectories: raise ValueError("Source has no eligible trajectories")
        quality_by_id = _quality_by_id(trajectories, self.trajectory_quality)
        base_mass_by_id = _normalized_quality_mass(quality_by_id)
        log.info("Source build completed: records=%d eligible=%d dropped=%d window_ineligible=%d trajectory_quality=%s", processed, len(trajectories), audit["dropped_count"], audit["window_ineligible_count"], self.trajectory_quality)
        return RuntimeSourceDataset(tuple(trajectories), {item.catalog_trajectory_id: item for item in trajectories}, quality_by_id, base_mass_by_id, {"dataset_kind": "source", "catalog_location": self.catalog.source_path, "catalog_kind": self.catalog.catalog_kind, "trajectory_quality": self.trajectory_quality, "audit": audit})


class RuntimeReferenceDatasetBuilder(RuntimeDatasetBuilder):
    def __init__(self, catalog_reader: CatalogReader, catalog: ReferenceCatalog, window_spec: WindowSpecification, trajectory_quality: str) -> None:
        super().__init__(catalog_reader, catalog)
        if not isinstance(catalog, ReferenceCatalog): raise TypeError("ReferenceCatalog is required")
        if not isinstance(window_spec, WindowSpecification): raise TypeError("window_spec must be WindowSpecification")
        if trajectory_quality not in TRAJECTORY_QUALITY_MODES: raise ValueError("trajectory_quality must be one of %s" % sorted(TRAJECTORY_QUALITY_MODES))
        self.window_spec = window_spec
        self.trajectory_quality = trajectory_quality

    def build(self) -> RuntimeReferenceDataset:
        total = len(self.catalog.records_data); trajectories: list[Trajectory] = []; audit = {"dropped_count": 0, "window_ineligible_count": 0}; reference_id = reference_id_of(self.catalog); log.info("Reference build started: reference_id=%s records=%d window_length=%d", reference_id, total, self.window_spec.window_length)
        processed = 0
        for record in self.catalog_reader.records(self.catalog):
            processed += 1
            if record.dropped:
                audit["dropped_count"] += 1; continue
            if legal_start_count(record.length, self.window_spec) == 0:
                audit["window_ineligible_count"] += 1; continue
            trajectories.append(Trajectory(record.catalog_trajectory_id, self._locator(record), record.length))
            if processed == 1 or processed % 10000 == 0 or processed == total:
                log.info("Reference construction for %s processed %d/%d records (%.1f%%); eligible %d, dropped %d, window-ineligible %d", reference_id, processed, total, 100.0 * processed / total, len(trajectories), audit["dropped_count"], audit["window_ineligible_count"])
        if not trajectories: raise ValueError("Reference %s has no eligible trajectories" % reference_id)
        quality_by_id = _quality_by_id(trajectories, self.trajectory_quality)
        raw_mass_by_id = _normalized_quality_mass(quality_by_id)
        log.info("Reference build completed: reference_id=%s records=%d eligible=%d dropped=%d window_ineligible=%d trajectory_quality=%s", reference_id, processed, len(trajectories), audit["dropped_count"], audit["window_ineligible_count"], self.trajectory_quality)
        return RuntimeReferenceDataset(reference_id, tuple(trajectories), {item.catalog_trajectory_id: item for item in trajectories}, quality_by_id, raw_mass_by_id, {"dataset_kind": "reference", "catalog_location": self.catalog.source_path, "catalog_kind": self.catalog.catalog_kind, "trajectory_quality": self.trajectory_quality, "audit": audit})


def reference_id_of(catalog: ReferenceCatalog) -> str:
    """Human-readable reference identity: the catalog file's parent directory name."""
    if not catalog.source_path: raise ValueError("ReferenceCatalog requires source_path to derive reference_id")
    from pathlib import Path
    return Path(catalog.source_path).resolve().parent.name


__all__ = ["TRAJECTORY_QUALITY_MODES", "RuntimeDatasetBuilder", "RuntimeSourceDatasetBuilder", "RuntimeReferenceDatasetBuilder", "reference_id_of"]
