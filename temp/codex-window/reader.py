"""Segment-level Bundle readers: the only readers of Feature Bundle assets.

Each reader is constructed from a validated Catalog and its Bundle locator,
reads feature metadata and Arrow data, validates ID/order/geometry/feature
validity, and builds compute parts with raw segment mass. Readers hold no
long-lived handles: Arrow files are read fully and closed. `close` exists
for the documented lifecycle and is a no-op. No metadata-only reader or
FeatureBundleReader exists; the Feature Bundle is the only persistent
source of analysis-segment features.

A complete training window, not an analysis segment, is the sole trajectory
eligibility criterion. Segments are proxies for transfer analysis, including
when one short segment covers a trajectory shorter than the analysis segment.
"""

from __future__ import annotations

import logging

import numpy as np
from pathlib import Path

from ts3.core.runtime.segment_model import AnalysisSegmentComputationPart, CompactRuntimeDatasetComputationPart, RuntimeDatasetComputationPart, TrajectoryComputationPart
from ts3.core.transfer.geometry import normalized_window_weight_masses
from ts3.core.runtime.windowing import legal_start_count
from ts3.features.config import ANALYSIS_SEGMENT_LENGTH, ANALYSIS_SEGMENT_STRIDE
from ts3.spec.catalogs.models import Catalog, ReferenceCatalog, SourceCatalog
from ts3.spec.files.shard_dataset import (
    FEATURE_METADATA_SCHEMA_VERSION,
    FEATURE_METADATA_SCHEMA_VERSION_V1,
    FEATURE_SCHEMA,
    FEATURE_SCHEMA_V1,
    iter_arrow_batches, iter_arrow_rows,
    read_feature_metadata,
)
from ts3.spec.load_dataframe import FEATURE_NAMES
from ts3.spec.models.data_model import TrajectoryLocator
from ts3.spec.models.training_protocol import WindowEligibilitySpecification
from ts3.core.bundle.bench import ReferenceBenchAdapter


log = logging.getLogger("train")


def _segment_feature_values(segment: dict) -> tuple[float, ...]:
    values = segment.get("feature_values")
    if not isinstance(values, dict):
        raise ValueError("segment feature_values are missing")
    return tuple(float(values[name]) for name in FEATURE_NAMES)



class _SegmentBundleReader:
    role = "source"

    def __init__(self, catalog: Catalog, window_spec: WindowEligibilitySpecification, bench_adapter: ReferenceBenchAdapter | None = None) -> None:
        if not isinstance(window_spec, WindowEligibilitySpecification):
            raise TypeError("window_spec must be WindowEligibilitySpecification")
        if window_spec.window_length > ANALYSIS_SEGMENT_LENGTH:
            log.info(
                "Training window spans analysis proxy segments; overlap-weighted sampling will be used: "
                "window_length=%d analysis_segment_length=%d",
                window_spec.window_length,
                ANALYSIS_SEGMENT_LENGTH,
            )
        self.catalog = catalog
        self.window_spec = window_spec
        self.bench_adapter = bench_adapter

    @property
    def bundle_root(self) -> Path:
        if not self.catalog.records_data:
            raise ValueError("catalog has no trajectory records")
        return Path(self.catalog.records_data[0].bundle_directory)

    def build_computation(self) -> RuntimeDatasetComputationPart:
        """Read and validate Feature assets, then build compute parts with raw mass."""
        metadata = read_feature_metadata(self.bundle_root)
        if metadata["schema_version"] not in {FEATURE_METADATA_SCHEMA_VERSION_V1, FEATURE_METADATA_SCHEMA_VERSION}:
            raise ValueError("feature asset schema_version mismatch: %s" % metadata["schema_version"])
        feature_schema = FEATURE_SCHEMA if metadata["schema_version"] == FEATURE_METADATA_SCHEMA_VERSION else FEATURE_SCHEMA_V1
        if metadata["analysis_segment_length"] != ANALYSIS_SEGMENT_LENGTH:
            raise ValueError(
                "feature assets require analysis_segment_length=%d but carry %d; "
                "recompute features with ANALYSIS_SEGMENT_LENGTH=%d"
                % (ANALYSIS_SEGMENT_LENGTH, metadata["analysis_segment_length"], ANALYSIS_SEGMENT_LENGTH)
            )
        if metadata["analysis_segment_stride"] != ANALYSIS_SEGMENT_STRIDE:
            raise ValueError(
                "feature assets require analysis_segment_stride=%d but carry %d"
                % (ANALYSIS_SEGMENT_STRIDE, metadata["analysis_segment_stride"])
            )
        record_by_id = {record.catalog_trajectory_id: record for record in self.catalog.records_data}
        feature_path = self.bundle_root / "feature" / "feature.arrow"
        if self.bench_adapter is None:
            return self._build_compact_computation(
                feature_path, record_by_id, metadata, feature_schema,
            )
        feature_ids = set()
        for row in iter_arrow_rows(feature_path, feature_schema, ("trajectory_id",)):
            trajectory_id = row["trajectory_id"]
            if trajectory_id in feature_ids:
                raise ValueError("feature rows must be unique per trajectory")
            feature_ids.add(trajectory_id)
        if feature_ids != set(record_by_id):
            raise ValueError("feature rows must match the Catalog trajectory ID set")

        candidate_ids = []
        processed_ids = set()
        for row in iter_arrow_rows(feature_path, feature_schema, ("trajectory_id", "dropped")):
            trajectory_id = row["trajectory_id"]
            if trajectory_id in processed_ids:
                raise ValueError("feature rows must be unique per trajectory")
            processed_ids.add(trajectory_id)
            record = record_by_id[trajectory_id]
            if record.dropped:
                continue
            if row["dropped"]:
                raise ValueError(
                    "Catalog record is valid but its feature row is dropped; regenerate the Catalog: %s"
                    % trajectory_id
                )
            # The shared minimum-window contract applies before any
            # benchmark-specific segment adaptation. No benchmark may create
            # or pad a sample that the DataLoader cannot legally read.
            if legal_start_count(record.length, self.window_spec) == 0:
                continue
            if self.bench_adapter is not None and not self.bench_adapter.include(record, self.window_spec):
                continue
            candidate_ids.append(trajectory_id)

        if not candidate_ids:
            raise ValueError("%s has no eligible trajectories" % (self.reference_id or self.role))
        total_trajectories = len(candidate_ids)
        window_length = self.window_spec.window_length
        stride = self.window_spec.window_stride

        trajectories = []
        candidate_set = set(candidate_ids)
        processed_ids.clear()
        feature_store = np.empty((4096, len(FEATURE_NAMES)), dtype=np.float64)
        feature_store_index = 0
        for row in iter_arrow_rows(feature_path, feature_schema):
            trajectory_id = row["trajectory_id"]
            if trajectory_id not in candidate_set:
                continue
            if trajectory_id in processed_ids:
                raise ValueError("feature rows must be unique per trajectory")
            processed_ids.add(trajectory_id)
            record = record_by_id[trajectory_id]
            segments = row["segments"]
            locator = TrajectoryLocator(record.catalog_trajectory_id, record.bundle_directory, record.shard_id, record.value_offset, record.length, record.dataset, record.subset, record.series_id)
            legal_count = legal_start_count(record.length, self.window_spec)
            if self.bench_adapter is not None:
                segments = self.bench_adapter.segments(record, tuple(segments), self.window_spec)
            overlap_masses = None
            overlap_mass_scale = None
            if self.bench_adapter is None:
                overlap_masses = normalized_window_weight_masses(
                    record.length,
                    tuple((int(segment["start"]), int(segment["length"])) for segment in segments),
                    window_length,
                    stride,
                )
                overlap_mass_scale = legal_count / sum(overlap_masses)
            parts = []
            for segment_index, segment in enumerate(segments):
                if self.bench_adapter is None:
                    overlap_mass = overlap_masses[segment_index] * overlap_mass_scale
                    raw_mass = overlap_mass / (total_trajectories * legal_count)
                else:
                    raw_mass = self.bench_adapter.raw_mass(record, segment, self.window_spec, total_trajectories)
                if feature_store_index == len(feature_store):
                    feature_store = np.empty((4096, len(FEATURE_NAMES)), dtype=np.float64)
                    feature_store_index = 0
                feature_store[feature_store_index] = _segment_feature_values(segment)
                feature_values = feature_store[feature_store_index]
                feature_store_index += 1
                transition_identity = None
                parts.append(AnalysisSegmentComputationPart(
                    locator, segment["segment_index"], segment["start"], segment["length"],
                    feature_values, raw_mass,
                ))
            if self.bench_adapter is not None:
                actual_mass = sum(part.raw_mass for part in parts)
                expected_mass = 1.0 / total_trajectories
                scale = expected_mass / actual_mass
                parts = [
                    AnalysisSegmentComputationPart(
                        part.trajectory_locator, part.segment_index, part.start, part.length,
                        part.feature_values, part.raw_mass * scale,
                    )
                    for part in parts
                ]
            computation_legal_count = legal_count
            trajectories.append(TrajectoryComputationPart(locator, record.length, computation_legal_count, tuple(parts)))

        return RuntimeDatasetComputationPart(
            self.role,
            self.reference_id,
            tuple(trajectories),
            {
                "feature_asset_schema_version": metadata["schema_version"],
                "analysis_segment_length": metadata["analysis_segment_length"],
                "analysis_segment_stride": metadata["analysis_segment_stride"],
            },
        )

    def _build_compact_computation(self, feature_path, record_by_id, metadata, feature_schema) -> CompactRuntimeDatasetComputationPart:
        """Build the common source path in one Arrow-batch pass.

        Arrow nested values stay columnar until they enter the final contiguous
        feature matrix.  This avoids a second file scan and millions of
        intermediate dictionaries, tuples, and segment dataclasses.
        """
        feature_ids: set[str] = set()
        locators = []
        trajectory_lengths = []
        legal_counts = []
        offsets = [0]
        starts_parts, lengths_parts, feature_parts, raw_parts = [], [], [], []
        total_batches = 0
        for total_batches, batch in enumerate(iter_arrow_batches(feature_path, feature_schema), 1):
            ids = batch.column("trajectory_id")
            dropped = batch.column("dropped")
            segments = batch.column("segments")
            segment_values = segments.values
            feature_struct = segment_values.field("feature_values")
            batch_starts = segment_values.field("start").to_numpy(zero_copy_only=False)
            batch_lengths = segment_values.field("length").to_numpy(zero_copy_only=False)
            batch_features = np.column_stack([
                feature_struct.field(name).to_numpy(zero_copy_only=False)
                for name in FEATURE_NAMES
            ])
            batch_offsets = segments.offsets.to_numpy(zero_copy_only=False)
            for row_index in range(batch.num_rows):
                trajectory_id = ids[row_index].as_py()
                if trajectory_id in feature_ids:
                    raise ValueError("feature rows must be unique per trajectory")
                feature_ids.add(trajectory_id)
                record = record_by_id.get(trajectory_id)
                if record is None:
                    raise ValueError("feature rows must match the Catalog trajectory ID set")
                if record.dropped or dropped[row_index].as_py() or legal_start_count(record.length, self.window_spec) == 0:
                    if not record.dropped and dropped[row_index].as_py():
                        raise ValueError("Catalog record is valid but its feature row is dropped; regenerate the Catalog: %s" % trajectory_id)
                    continue
                start, stop = int(batch_offsets[row_index]), int(batch_offsets[row_index + 1])
                if start == stop:
                    raise ValueError("valid feature row requires segments: %s" % trajectory_id)
                segment_starts = batch_starts[start:stop].astype(np.int64, copy=True)
                segment_lengths = batch_lengths[start:stop].astype(np.int64, copy=True)
                legal_count = legal_start_count(record.length, self.window_spec)
                overlap_masses = normalized_window_weight_masses(record.length, tuple(zip(segment_starts.tolist(), segment_lengths.tolist())), self.window_spec.window_length, self.window_spec.window_stride)
                raw = np.asarray(overlap_masses, dtype=np.float64) * legal_count / sum(overlap_masses)
                locators.append(TrajectoryLocator(record.catalog_trajectory_id, record.bundle_directory, record.shard_id, record.value_offset, record.length, record.dataset, record.subset, record.series_id))
                trajectory_lengths.append(record.length)
                legal_counts.append(legal_count)
                starts_parts.append(segment_starts)
                lengths_parts.append(segment_lengths)
                feature_parts.append(np.asarray(batch_features[start:stop], dtype=np.float64))
                raw_parts.append(raw / legal_count)
                offsets.append(offsets[-1] + len(segment_starts))
            if total_batches == 1 or total_batches % 10 == 0:
                log.info("%s computation built %d batches, %d trajectories, and %d segments", self.role.capitalize(), total_batches, len(locators), offsets[-1])
        if feature_ids != set(record_by_id):
            raise ValueError("feature rows must match the Catalog trajectory ID set")
        if not locators:
            raise ValueError("%s has no eligible trajectories" % self.role)
        total_trajectories = len(locators)
        raw_masses = np.concatenate(raw_parts) / total_trajectories
        return CompactRuntimeDatasetComputationPart(
            self.role, self.reference_id, tuple(locators), np.asarray(trajectory_lengths, dtype=np.int64),
            np.asarray(legal_counts, dtype=np.int64), np.asarray(offsets, dtype=np.int64),
            np.concatenate(starts_parts), np.concatenate(lengths_parts), np.concatenate(feature_parts), raw_masses,
            {"feature_asset_schema_version": metadata["schema_version"], "analysis_segment_length": metadata["analysis_segment_length"], "analysis_segment_stride": metadata["analysis_segment_stride"], "feature_bundle_transition": metadata["schema_version"] == FEATURE_METADATA_SCHEMA_VERSION}
        )

    def close(self) -> None:
        """Release any resources; readers hold no long-lived handles."""

    @property
    def reference_id(self) -> str | None:
        return None


class SourceBundleReader(_SegmentBundleReader):
    """Feature Bundle reader for the validated Source Catalog."""

    role = "source"

    def __init__(self, catalog: SourceCatalog, window_spec: WindowEligibilitySpecification) -> None:
        if not isinstance(catalog, SourceCatalog):
            raise TypeError("SourceBundleReader requires SourceCatalog")
        super().__init__(catalog, window_spec, )


class ReferenceBundleReader(_SegmentBundleReader):
    """Feature Bundle reader for one validated Reference Catalog."""

    role = "reference"

    def __init__(self, catalog: ReferenceCatalog, window_spec: WindowEligibilitySpecification, bench_adapter: ReferenceBenchAdapter | None = None) -> None:
        if not isinstance(catalog, ReferenceCatalog):
            raise TypeError("ReferenceBundleReader requires ReferenceCatalog")
        super().__init__(catalog, window_spec, bench_adapter, )

    @property
    def reference_id(self) -> str:
        from ts3.core.runtime.builder import reference_id_of
        return reference_id_of(self.catalog)


__all__ = ["SourceBundleReader", "ReferenceBundleReader"]
