"""Core data structures for RAGE."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass(frozen=True)
class RAGEConfig:
    """Configuration for RAGE algorithm."""

    # Time series protocol
    context_length: int = 512
    prediction_length: int = 64
    min_fragment_length: int = 576  # context + prediction + padding

    # Adaptive quadrature parameters
    direction_tolerance: float = 0.1  # cosine residual threshold
    magnitude_tolerance: float = 0.15  # magnitude residual threshold
    min_interval_size: int = 3  # minimum window starts in an interval
    use_simpson: bool = True  # use Simpson quadrature if True, trapezoidal if False

    # Reference perturbation parameters
    block_length: int = 16  # moving block size for bootstrap
    num_reference_versions: int = 50  # R in paper
    perturbation_seed: int = 42

    # Selection parameters
    selection_budget: int = 1000  # B in paper
    ranking_budget: int = 1000  # k in paper (usually same as selection_budget)
    stability_threshold: float = 0.5  # π₀ in paper
    lower_quantile: float = 0.25  # α in paper

    # Scoring parameters
    score_epsilon: float = 1e-8  # ε for numerical stability in cosine

    # Computational parameters
    gradient_device: str = "auto"
    projection_dimension: int = 4096
    projection_seed: int = 42

    def __post_init__(self) -> None:
        # Validate parameters
        if self.context_length <= 0 or self.prediction_length <= 0:
            raise ValueError("context and prediction lengths must be positive")
        if self.min_fragment_length < (self.context_length + self.prediction_length):
            raise ValueError("min_fragment_length must allow at least one valid window")
        if not (0 < self.direction_tolerance < 1):
            raise ValueError("direction_tolerance must be in (0, 1)")
        if self.magnitude_tolerance <= 0:
            raise ValueError("magnitude_tolerance must be positive")
        if self.min_interval_size < 2:
            raise ValueError("min_interval_size must be at least 2 for interpolation")
        if self.block_length < 1:
            raise ValueError("block_length must be positive")
        if self.num_reference_versions < 1:
            raise ValueError("num_reference_versions must be positive")
        if self.selection_budget < 1:
            raise ValueError("selection_budget must be positive")
        if self.ranking_budget < 1:
            raise ValueError("ranking_budget must be positive")
        if not (0 <= self.stability_threshold <= 1):
            raise ValueError("stability_threshold must be in [0, 1]")
        if not (0 < self.lower_quantile <= 0.5):
            raise ValueError("lower_quantile must be in (0, 0.5]")

    @property
    def window_length(self) -> int:
        """Total window span (context + prediction)."""
        return self.context_length + self.prediction_length


@dataclass
class WindowStartPoint:
    """Represents a single window starting position within a fragment.

    In RAGE, sampling windows is equivalent to sampling valid start points.
    Each start point corresponds to exactly one (context, prediction) pair.
    """
    fragment_index: int  # Which fragment this belongs to
    global_index: int  # Global index across all fragments
    position: int  # Position within fragment (1-indexed for paper consistency)
    context_length: int
    prediction_length: int

    @property
    def window_length(self) -> int:
        return self.context_length + self.prediction_length

    def __repr__(self) -> str:
        return f"WindowStart(fragment={self.fragment_index}, pos={self.position})"


@dataclass
class TimeSeriesFragment:
    """A time series fragment with entity boundaries and state preservation.

    Fragments organize data by entity boundaries, known state boundaries,
    or non-overlapping time blocks. Windows never cross fragment boundaries.
    """
    data: np.ndarray  # Shape: [length], the time series values
    entity_id: str | None = None  # Optional entity identifier
    state_weight: float = 1.0  # Weight for this fragment's states
    fragment_id: int = 0  # Unique identifier

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float32)
        if self.data.ndim != 1:
            raise ValueError("Fragment data must be 1-dimensional")
        if not np.isfinite(self.data).all():
            raise ValueError("Fragment data must be finite")
        if self.state_weight <= 0:
            raise ValueError("state_weight must be positive")

    @property
    def length(self) -> int:
        return len(self.data)

    def valid_starts(
        self, context_length: int, prediction_length: int, window_stride: int = 1
    ) -> np.ndarray:
        """Return valid 1-indexed context starts under the ``(C, H, D)`` protocol."""
        if window_stride <= 0:
            raise ValueError("window_stride must be positive")
        min_length = context_length + prediction_length
        if self.length < min_length:
            return np.array([], dtype=np.int32)
        max_start = self.length - min_length + 1
        return np.arange(1, max_start + 1, window_stride, dtype=np.int32)

    def extract_window(self, start: int, context_length: int, prediction_length: int) -> np.ndarray:
        """Extract a (context, prediction) window given start position (1-indexed)."""
        if start < 1:
            raise ValueError(f"Start position must be ≥ 1, got {start}")
        total = context_length + prediction_length
        end = start - 1 + total
        if end > self.length:
            raise ValueError(f"Window [{start}, {end}) exceeds fragment length {self.length}")
        # Convert to 0-indexed for numpy
        return self.data[start-1:end].copy()

    def __len__(self) -> int:
        return len(self.data)


@dataclass
class AdaptiveInterval:
    """An interval of window start points after adaptive subdivision.

    Represents a contiguous range of start positions that passed the
    residual test (no subdivision needed). Used for quadrature approximation.
    """
    fragment_index: int
    start_position: int  # Start of interval (1-indexed)
    end_position: int  # End of interval (1-indexed, inclusive)
    computed_anchors: dict[int, np.ndarray]  # position -> gradient projection
    interval_weight: float = 1.0  # Total weight of starts in this interval

    @property
    def size(self) -> int:
        """Number of start positions in this interval."""
        return self.end_position - self.start_position + 1

    @property
    def positions(self) -> np.ndarray:
        """All positions in this interval."""
        return np.arange(self.start_position, self.end_position + 1, dtype=np.int32)

    def contains(self, position: int) -> bool:
        """Check if position is in this interval."""
        return self.start_position <= position <= self.end_position

    def __repr__(self) -> str:
        return f"Interval(fragment={self.fragment_index}, [{self.start_position}, {self.end_position}], size={self.size})"


@dataclass
class GradientTrajectory:
    """Gradient values along start positions within a fragment.

    Represents the high-dimensional trajectory g_{j,t} as start position t varies.
    Adjacent positions have highly overlapping windows and thus correlated gradients.
    """
    fragment_index: int
    positions: np.ndarray  # Sorted positions where gradients are computed
    gradients: np.ndarray  # Shape: [len(positions), projection_dimension]
    weights: np.ndarray  # Shape: [len(positions)], weight for each position

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=np.int32)
        self.gradients = np.asarray(self.gradients, dtype=np.float32)
        self.weights = np.asarray(self.weights, dtype=np.float32)
        if len(self.positions) != len(self.gradients):
            raise ValueError("positions and gradients must have same length")
        if len(self.weights) != len(self.positions):
            raise ValueError("weights must match positions length")
        if not np.isfinite(self.gradients).all():
            raise ValueError("gradients must be finite")
        if (self.weights < 0).any():
            raise ValueError("weights must be non-negative")

    @property
    def num_points(self) -> int:
        return len(self.positions)

    @property
    def dimension(self) -> int:
        return self.gradients.shape[1]

    def gradient_at(self, position: int) -> np.ndarray | None:
        """Get gradient at exact position if computed, otherwise None."""
        idx = np.searchsorted(self.positions, position)
        if idx < len(self.positions) and self.positions[idx] == position:
            return self.gradients[idx]
        return None

    def interpolate_at(self, position: int, left_pos: int, right_pos: int) -> np.ndarray:
        """Linear interpolation between two computed anchor points."""
        if left_pos >= right_pos:
            raise ValueError("left_pos must be less than right_pos")
        left_grad = self.gradient_at(left_pos)
        right_grad = self.gradient_at(right_pos)
        if left_grad is None or right_grad is None:
            raise ValueError("Both anchor positions must have computed gradients")

        # Linear interpolation: (right - pos) / (right - left) * left + (pos - left) / (right - left) * right
        alpha = (right_pos - position) / (right_pos - left_pos)
        beta = (position - left_pos) / (right_pos - left_pos)
        return alpha * left_grad + beta * right_grad

    def total_weight(self) -> float:
        """Sum of all weights."""
        return float(self.weights.sum())

    def weighted_average(self) -> np.ndarray:
        """Weighted average of gradients."""
        total = self.total_weight()
        if total == 0:
            return np.zeros_like(self.gradients[0])
        return (self.weights[:, None] * self.gradients).sum(axis=0) / total

    def __len__(self) -> int:
        return len(self.positions)


@dataclass
class CandidateWindow:
    """A candidate window from source data pool."""
    window_data: np.ndarray  # Shape: [window_length]
    fragment_id: int  # Which source fragment
    position: int  # Start position (1-indexed)
    candidate_id: int  # Global unique identifier

    def __post_init__(self) -> None:
        self.window_data = np.asarray(self.window_data, dtype=np.float32)
        if self.window_data.ndim != 1:
            raise ValueError("Window data must be 1-dimensional")

    @property
    def length(self) -> int:
        return len(self.window_data)

    def __repr__(self) -> str:
        return f"Candidate(id={self.candidate_id}, fragment={self.fragment_id}, pos={self.position})"


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray, epsilon: float = 1e-8) -> float:
    """Compute cosine similarity between two vectors.

    The epsilon guards only the zero-norm case: identical non-zero vectors
    return exactly 1.0, so residual tolerances can be set arbitrarily tight.
    """
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm < epsilon or b_norm < epsilon:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm + epsilon))


def compute_inner_product(a: np.ndarray, b: np.ndarray) -> float:
    """Compute inner product between two vectors."""
    return float(np.dot(a, b))


__all__ = [
    "RAGEConfig",
    "TimeSeriesFragment",
    "WindowStartPoint",
    "AdaptiveInterval",
    "GradientTrajectory",
    "CandidateWindow",
    "compute_cosine_similarity",
    "compute_inner_product",
]
