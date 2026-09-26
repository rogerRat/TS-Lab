"""Shared cgroup memory admission and runtime guard.

This module deliberately reads the active cgroup rather than host memory. It
is project runtime infrastructure, not part of the TS3 domain kernel.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Mapping
import logging
from pathlib import Path
from typing import Iterator, Optional
import math


DEFAULT_PLANNING_FRACTION = 0.70
DEFAULT_STOP_FRACTION = 0.80
_V1_UNLIMITED = 1 << 60


class MemoryBudgetError(RuntimeError):
    """Raised when a finite cgroup budget cannot safely admit work.

    @input: ASCII diagnostic text.
    @output: RuntimeError-compatible exception.
    @scenario: Stop long-running stages before cgroup OOM termination.
    """


@dataclass(frozen=True)
class MemorySnapshot:
    """One serializable observation of the shared memory contract.

    @input: Resolved thresholds and current usage.
    @output: Immutable snapshot with JSON-compatible as_dict output.
    @scenario: Put common memory fields into stage logs and audit metadata.
    """

    stage: str
    limit_bytes: int
    planning_budget_bytes: int
    stop_threshold_bytes: int
    current_bytes: int
    effective_current_bytes: Optional[int] = None
    reclaimable_cache_bytes: Optional[int] = None
    batch_rows: Optional[int] = None
    batch_values: Optional[int] = None
    worker_count: Optional[int] = None
    peak_bytes: Optional[int] = None

    def as_dict(self) -> dict[str, object]:
        """Return the snapshot as a JSON-compatible mapping."""
        return {
            "stage": self.stage,
            "limit_bytes": self.limit_bytes,
            "planning_budget_bytes": self.planning_budget_bytes,
            "stop_threshold_bytes": self.stop_threshold_bytes,
            "current_bytes": self.current_bytes,
            "effective_current_bytes": self.effective_current_bytes,
            "reclaimable_cache_bytes": self.reclaimable_cache_bytes,
            "batch_rows": self.batch_rows,
            "batch_values": self.batch_values,
            "worker_count": self.worker_count,
            "peak_bytes": self.peak_bytes,
        }


def _validate_fractions(planning_fraction: float, stop_fraction: float) -> tuple[float, float]:
    """Validate the planning and stop fractions.

    @input: Two finite fractions in (0, 1).
    @output: Normalized float pair satisfying planning < stop.
    @scenario: Reject contradictory memory thresholds at startup.
    """
    planning = float(planning_fraction)
    stop = float(stop_fraction)
    if not math.isfinite(planning) or not math.isfinite(stop):
        raise ValueError("memory fractions must be finite")
    if not 0.0 < planning < stop < 1.0:
        raise ValueError("memory fractions must satisfy 0 < planning < stop < 1")
    return planning, stop


def resolve_memory_fractions(config: Optional[Mapping[str, object]] = None) -> tuple[float, float]:
    """Resolve memory fractions from one project configuration mapping.

    @input: Optional root configuration mapping with a memory mapping.
    @output: Validated (planning_fraction, stop_fraction) float pair.
    @scenario: Keep CLI stages on the same explicit cgroup memory contract.
    @param config: Parsed YAML configuration, or None to use policy defaults.
    """
    if config is None:
        return DEFAULT_PLANNING_FRACTION, DEFAULT_STOP_FRACTION
    memory = config.get("memory", {})
    if memory is None:
        memory = {}
    if not isinstance(memory, Mapping):
        raise ValueError("memory must be a mapping")
    planning = memory.get("planning_fraction", DEFAULT_PLANNING_FRACTION)
    stop = memory.get("stop_fraction", DEFAULT_STOP_FRACTION)
    return _validate_fractions(float(planning), float(stop))


def load_memory_fractions(config_path: Optional[Path] = None) -> tuple[float, float]:
    """Load and validate memory fractions from an optional YAML file.

    @input: Optional YAML configuration path.
    @output: Validated (planning_fraction, stop_fraction) float pair.
    @scenario: Give standalone data stages the same config behavior as RQ3.
    @param config_path: YAML path containing an optional memory mapping.
    """
    if config_path is None:
        return resolve_memory_fractions()
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load memory configuration") from exc
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError("memory config not found: %s" % path)
    value = yaml.safe_load(path.read_text(encoding="ascii"))
    if not isinstance(value, Mapping):
        raise ValueError("memory config must be a mapping: %s" % path)
    return resolve_memory_fractions(value)


def _read_limit_value(path: Path) -> Optional[int]:
    """Read one cgroup memory limit value.

    @input: Path containing an ASCII byte count or v2 max.
    @output: Positive byte count, or None for unlimited.
    @scenario: Normalize cgroup v1/v2 limits.
    """
    try:
        value = path.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise MemoryBudgetError("cannot read cgroup memory file: %s" % path) from exc
    if value == "max":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise MemoryBudgetError("malformed cgroup memory value in %s" % path) from exc
    if parsed <= 0 or parsed >= _V1_UNLIMITED:
        return None
    return parsed


def _read_current_value(path: Path) -> int:
    """Read current cgroup memory usage.

    @input: Path containing a non-negative ASCII byte count.
    @output: Current usage in bytes.
    @scenario: Apply the runtime stop threshold before new work.
    """
    try:
        parsed = int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError) as exc:
        raise MemoryBudgetError("cannot read cgroup current usage: %s" % path) from exc
    if parsed < 0:
        raise MemoryBudgetError("negative cgroup current usage: %s" % path)
    return parsed


def _read_memory_stat(path: Path) -> dict[str, int]:
    """Read cgroup memory.stat counters needed for reclaimable cache credit."""
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except OSError as exc:
        raise MemoryBudgetError("cannot read cgroup memory stat file: %s" % path) from exc
    result: dict[str, int] = {}
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            value = int(parts[1])
        except ValueError:
            continue
        if value >= 0:
            result[parts[0]] = value
    return result


def resolve_memory_paths(cgroup_root: Path = Path("/sys/fs/cgroup")) -> tuple[Path, Path, str]:
    """Resolve active cgroup limit and current-usage files.

    @input: cgroup_root is an active cgroup root or test fixture root.
    @output: (limit_path, current_path, version) for v2 or v1.
    @scenario: Discover cgroup memory without host fallbacks.
    """
    root = Path(cgroup_root)
    v2_limit = root / "memory.max"
    v2_current = root / "memory.current"
    if v2_limit.is_file() and v2_current.is_file():
        return v2_limit, v2_current, "v2"
    for v1_root in (root / "memory", root / "memory cgroup", Path("/sys/fs/cgroup/memory")):
        limit = v1_root / "memory.limit_in_bytes"
        current = v1_root / "memory.usage_in_bytes"
        if limit.is_file() and current.is_file():
            return limit, current, "v1"
    raise MemoryBudgetError("active cgroup memory limit files were not found under %s" % root)


def resolve_memory_limit(cgroup_root: Path = Path("/sys/fs/cgroup")) -> int:
    """Resolve a finite active cgroup memory limit.

    @input: cgroup_root is an active cgroup root or test fixture root.
    @output: Positive finite limit in bytes.
    @scenario: Fail a full-data stage instead of using host free memory.
    """
    limit_path, _current_path, _version = resolve_memory_paths(cgroup_root)
    limit = _read_limit_value(limit_path)
    if limit is None:
        raise MemoryBudgetError("cgroup memory limit is unlimited or unavailable: %s" % limit_path)
    return limit


@dataclass(frozen=True)
class MemoryBudget:
    """Resolved planning and runtime thresholds.

    @input: Finite cgroup limit, fractions, and cgroup file locations.
    @output: Immutable guard for admission checks and snapshots.
    @scenario: Share one 70/80 percent contract across long-running stages.
    """

    limit_bytes: int
    planning_fraction: float = DEFAULT_PLANNING_FRACTION
    stop_fraction: float = DEFAULT_STOP_FRACTION
    limit_path: Path = Path("/sys/fs/cgroup/memory.max")
    current_path: Path = Path("/sys/fs/cgroup/memory.current")
    cgroup_version: str = "v2"

    def __post_init__(self) -> None:
        """Validate the resolved byte limit and fractions."""
        if int(self.limit_bytes) <= 0:
            raise ValueError("limit_bytes must be positive")
        planning, stop = _validate_fractions(self.planning_fraction, self.stop_fraction)
        object.__setattr__(self, "limit_bytes", int(self.limit_bytes))
        object.__setattr__(self, "planning_fraction", planning)
        object.__setattr__(self, "stop_fraction", stop)

    @property
    def planning_budget_bytes(self) -> int:
        """Return the floor of the planning budget in bytes."""
        return int(self.limit_bytes * self.planning_fraction)

    @property
    def stop_threshold_bytes(self) -> int:
        """Return the floor of the runtime stop threshold in bytes."""
        return int(self.limit_bytes * self.stop_fraction)

    @classmethod
    def resolve(
        cls,
        planning_fraction: float = DEFAULT_PLANNING_FRACTION,
        stop_fraction: float = DEFAULT_STOP_FRACTION,
        cgroup_root: Path = Path("/sys/fs/cgroup"),
    ) -> "MemoryBudget":
        """Resolve a finite budget from the active cgroup.

        @input: Fractions with 0 < planning < stop < 1; optional cgroup root.
        @output: A fully resolved MemoryBudget.
        @scenario: Initialize a full-data stage with visible thresholds.
        """
        limit_path, current_path, version = resolve_memory_paths(cgroup_root)
        limit = _read_limit_value(limit_path)
        if limit is None:
            raise MemoryBudgetError("cgroup memory limit is unlimited: %s" % limit_path)
        return cls(limit, planning_fraction, stop_fraction, limit_path, current_path, version)

    @classmethod
    def from_limit(
        cls,
        limit_bytes: int,
        planning_fraction: float = DEFAULT_PLANNING_FRACTION,
        stop_fraction: float = DEFAULT_STOP_FRACTION,
    ) -> "MemoryBudget":
        """Build a testable budget from a known finite limit.

        @input: Positive finite limit and fractions.
        @output: MemoryBudget using the default cgroup current path.
        @scenario: Unit-test threshold arithmetic without host files.
        """
        return cls(int(limit_bytes), planning_fraction, stop_fraction)

    def current_bytes(self) -> int:
        """Return current usage from the resolved cgroup current file."""
        return _read_current_value(self.current_path)

    def effective_current(self) -> tuple[int, int]:
        """Return usage after crediting clean reclaimable file cache."""
        current = self.current_bytes()
        stats = _read_memory_stat(self.current_path.parent / "memory.stat")
        reclaimable = max(
            0,
            stats.get("file", 0)
            - stats.get("shmem", 0)
            - stats.get("file_dirty", 0)
            - stats.get("file_writeback", 0),
        )
        return max(0, current - reclaimable), reclaimable

    def snapshot(
        self,
        stage: str,
        batch_rows: Optional[int] = None,
        batch_values: Optional[int] = None,
        worker_count: Optional[int] = None,
        peak_bytes: Optional[int] = None,
    ) -> MemorySnapshot:
        """Capture one structured current-usage observation.

        @input: Stage label and optional batch and worker diagnostics.
        @output: MemorySnapshot containing thresholds and usage.
        @scenario: Log memory state at stage startup and batch boundaries.
        """
        effective, reclaimable = self.effective_current()
        return MemorySnapshot(
            stage=str(stage),
            limit_bytes=self.limit_bytes,
            planning_budget_bytes=self.planning_budget_bytes,
            stop_threshold_bytes=self.stop_threshold_bytes,
            current_bytes=_read_current_value(self.current_path),
            effective_current_bytes=effective,
            reclaimable_cache_bytes=reclaimable,
            batch_rows=batch_rows,
            batch_values=batch_values,
            worker_count=worker_count,
            peak_bytes=peak_bytes,
        )

    def check(
        self,
        stage: str,
        additional_bytes: int = 0,
        batch_rows: Optional[int] = None,
        batch_values: Optional[int] = None,
        worker_count: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ) -> MemorySnapshot:
        """Check current and projected usage before admitting work.

        @input: Non-negative projected bytes and batch diagnostics.
        @output: Passing MemorySnapshot; raises MemoryBudgetError otherwise.
        @scenario: Stop new work at 80 percent and reject planning overflow.
        """
        if int(additional_bytes) < 0:
            raise ValueError("additional_bytes must be non-negative")
        snapshot = self.snapshot(stage, batch_rows, batch_values, worker_count)
        effective = snapshot.effective_current_bytes
        if effective is None:
            effective = snapshot.current_bytes
        projected = effective + int(additional_bytes)
        if effective >= self.stop_threshold_bytes:
            message = (
                "memory stop threshold reached during %s: effective=%d raw=%d stop=%d limit=%d "
                "batch_rows=%s batch_values=%s workers=%s"
                % (stage, effective, snapshot.current_bytes, self.stop_threshold_bytes, self.limit_bytes,
                   batch_rows, batch_values, worker_count)
            )
            if logger is not None:
                logger.error(message)
            raise MemoryBudgetError(message)
        if projected > self.planning_budget_bytes:
            message = (
                "memory planning budget exceeded during %s: projected effective=%d planning=%d "
                "available=%d required_additional=%d batch_rows=%s batch_values=%s workers=%s"
                % (stage, projected, self.planning_budget_bytes,
                   max(0, self.planning_budget_bytes - effective),
                   additional_bytes, batch_rows, batch_values, worker_count)
            )
            if logger is not None:
                logger.error(message)
            raise MemoryBudgetError(message)
        if logger is not None:
            logger.debug("memory admission: %s", snapshot.as_dict())
        return snapshot


@contextmanager
def memory_guard(
    stage: str,
    budget: Optional[MemoryBudget] = None,
    planning_fraction: float = DEFAULT_PLANNING_FRACTION,
    stop_fraction: float = DEFAULT_STOP_FRACTION,
    logger: Optional[logging.Logger] = None,
) -> Iterator[MemoryBudget]:
    """Resolve and log one stage budget for a guarded operation.

    @input: Stage name, optional budget, fractions, and logger.
    @output: Context manager yielding MemoryBudget and preserving exceptions.
    @scenario: Make each long stage report common thresholds at startup.
    """
    active = budget or MemoryBudget.resolve(planning_fraction, stop_fraction)
    if logger is not None:
        logger.info(
            "memory contract for %s: cgroup=%s limit=%d planning=%d stop=%d workers=unset",
            stage, active.cgroup_version, active.limit_bytes,
            active.planning_budget_bytes, active.stop_threshold_bytes,
        )
    yield active
