"""Benchmark-specific reference Bundle compatibility."""

from __future__ import annotations

from typing import Any, Mapping

from ts3.core.runtime.windowing import legal_start_count
from ts3.core.transfer.geometry import compute_segment_start_ranges
from ts3.spec.models.training_protocol import WindowEligibilitySpecification


class ReferenceBenchAdapter:
    """Small hook for benchmark-specific reference trajectory rules."""

    def include(self, record: Any, window_spec: WindowEligibilitySpecification) -> bool:
        raise NotImplementedError

    def segments(self, record: Any, values: tuple[Mapping[str, Any], ...], window_spec: WindowEligibilitySpecification) -> tuple[Mapping[str, Any], ...]:
        raise NotImplementedError

    def raw_mass(self, record: Any, segment: Mapping[str, Any], window_spec: WindowEligibilitySpecification, trajectory_count: int) -> float:
        raise NotImplementedError


class _SafePrefixReferenceBenchAdapter(ReferenceBenchAdapter):
    """Reference rules for a benchmark-provided safe-history prefix."""

    def include(self, record: Any, window_spec: WindowEligibilitySpecification) -> bool:
        # Benchmark identity may customize segment selection, but it cannot
        # weaken the shared minimum training-window contract.
        return legal_start_count(record.length, window_spec) > 0

    def segments(self, record: Any, values: tuple[Mapping[str, Any], ...], window_spec: WindowEligibilitySpecification) -> tuple[Mapping[str, Any], ...]:
        if not values:
            return values
        return values

    def raw_mass(self, record: Any, segment: Mapping[str, Any], window_spec: WindowEligibilitySpecification, trajectory_count: int) -> float:
        ranges = compute_segment_start_ranges(record.length, int(segment["start"]), int(segment["length"]), window_spec.window_length, window_spec.window_stride)
        return ranges.total_overlap() / (trajectory_count * legal_start_count(record.length, window_spec))


class FEVReferenceBenchAdapter(_SafePrefixReferenceBenchAdapter):
    """FEV reference exception for trajectories shorter than a train window."""


class TimeReferenceBenchAdapter(_SafePrefixReferenceBenchAdapter):
    """TIME/timesBench reference rules for safe-history trajectories."""


# Keep the upstream benchmark name visible to callers that use its acronym.
TIMEReferenceBenchAdapter = TimeReferenceBenchAdapter
TimesBenchReferenceBenchAdapter = TimeReferenceBenchAdapter


def reference_bench_adapter(benchmark: str | None) -> ReferenceBenchAdapter:
    if benchmark == "fev":
        return FEVReferenceBenchAdapter()
    if benchmark == "time":
        return TimeReferenceBenchAdapter()
    raise NotImplementedError("reference benchmark adapter is not implemented: %s" % benchmark)


__all__ = [
    "ReferenceBenchAdapter",
    "FEVReferenceBenchAdapter",
    "TimeReferenceBenchAdapter",
    "TIMEReferenceBenchAdapter",
    "TimesBenchReferenceBenchAdapter",
    "reference_bench_adapter",
]
