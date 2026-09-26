"""Adaptive gradient trajectory quadrature for RAGS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from collections import deque
import numpy as np

from .core import (
    RAGSConfig,
    TimeSeriesFragment,
    AdaptiveInterval,
    GradientTrajectory,
    compute_cosine_similarity,
)


def _lagrange_weights(t: float, nodes: list[float]) -> list[float]:
    """Lagrange interpolation weights: interp(t) = Σ w_i · f(nodes_i)."""
    weights = []
    for i, node_i in enumerate(nodes):
        weight = 1.0
        for j, node_j in enumerate(nodes):
            if i != j:
                weight *= (t - node_j) / (node_i - node_j)
        weights.append(weight)
    return weights


@dataclass
class QuadratureResult:
    """Result of adaptive quadrature on a single fragment."""
    fragment_index: int
    intervals: list[AdaptiveInterval]
    computed_gradients: int  # Total number of gradient computations performed
    total_positions: int  # Total valid start positions in fragment

    @property
    def compression_ratio(self) -> float:
        """Ratio of positions to computed gradients."""
        if self.computed_gradients == 0:
            return float('inf')
        return self.total_positions / self.computed_gradients

    @property
    def coverage_ratio(self) -> float:
        """Fraction of total positions covered by intervals."""
        covered = sum(interval.size for interval in self.intervals)
        return covered / self.total_positions if self.total_positions > 0 else 0.0


@dataclass
class MidpointResidualComputer:
    """Computes midpoint interpolation residuals for adaptive subdivision."""

    config: RAGSConfig

    def direction_residual(self, gradient_mid: np.ndarray, gradient_interp: np.ndarray) -> float:
        """Compute cosine residual: 1 - cos(g_mid, g_interp)."""
        cosine = compute_cosine_similarity(gradient_mid, gradient_interp, self.config.score_epsilon)
        return 1.0 - cosine

    def magnitude_residual(self, gradient_mid: np.ndarray, gradient_interp: np.ndarray) -> float:
        """Compute magnitude residual: ||g_mid - g_interp|| / (||g_mid|| + ε)."""
        diff = gradient_mid - gradient_interp
        norm_diff = np.linalg.norm(diff)
        norm_mid = np.linalg.norm(gradient_mid)
        return float(norm_diff / (norm_mid + self.config.score_epsilon))

    def should_subdivide(self, gradient_mid: np.ndarray, gradient_interp: np.ndarray) -> bool:
        """Check if residuals exceed tolerances."""
        dir_res = self.direction_residual(gradient_mid, gradient_interp)
        mag_res = self.magnitude_residual(gradient_mid, gradient_interp)
        return (dir_res > self.config.direction_tolerance or
                mag_res > self.config.magnitude_tolerance)

    def compute_residuals(self, gradient_mid: np.ndarray, gradient_interp: np.ndarray) -> dict[str, float]:
        """Compute both residuals for diagnostics."""
        return {
            "direction": self.direction_residual(gradient_mid, gradient_interp),
            "magnitude": self.magnitude_residual(gradient_mid, gradient_interp),
        }


@dataclass
class AdaptiveQuadratureEngine:
    """Adaptive quadrature engine for gradient trajectory approximation.

    Implements the core algorithm from Section 4.2 of the RAGS paper:
    1. Start with endpoint anchors of each fragment
    2. Test midpoint interpolation residuals
    3. Subdivide if residuals exceed tolerance
    4. Apply quadrature to accepted intervals

    ``anchor_cache`` optionally maps (fragment_id, start_position) to a
    precomputed gradient; cached anchors skip the gradient computer entirely,
    so a batched precomputation pass can serve the adaptive subdivision. Both
    stages of the pipeline are meant to share one such anchor set.
    """

    config: RAGSConfig
    gradient_computer: Callable[[np.ndarray, int, int], np.ndarray]  # window -> projected gradient
    anchor_cache: dict[tuple[int, int], np.ndarray] | None = None
    residual_computer: MidpointResidualComputer = field(init=False)

    def __post_init__(self) -> None:
        self.residual_computer = MidpointResidualComputer(self.config)

    def process_fragment(
        self,
        fragment: TimeSeriesFragment,
        fragment_index: int,
    ) -> QuadratureResult:
        """Process a single fragment with adaptive quadrature."""
        valid_starts = fragment.valid_starts(
            self.config.context_length,
            self.config.prediction_length
        )

        if len(valid_starts) == 0:
            return QuadratureResult(
                fragment_index=fragment_index,
                intervals=[],
                computed_gradients=0,
                total_positions=0,
            )

        # Initialize with endpoints
        first_pos = int(valid_starts[0])
        last_pos = int(valid_starts[-1])

        # Storage for computed gradients
        computed_grads: dict[int, np.ndarray] = {}

        # Compute endpoint gradients
        computed_grads[first_pos] = self._compute_gradient(fragment, first_pos)
        computed_grads[last_pos] = self._compute_gradient(fragment, last_pos)

        # Queue of intervals to test: (start, end)
        test_queue = deque([(first_pos, last_pos)])
        accepted_intervals: list[AdaptiveInterval] = []

        while test_queue:
            start, end = test_queue.popleft()

            # Test midpoint
            mid_pos = (start + end) // 2
            if (end - start + 1) < self.config.min_interval_size or not start < mid_pos < end:
                # Interval too small to subdivide further: the midpoint
                # coincides with an endpoint, so accept without testing.
                interval = self._create_interval(fragment_index, start, end, computed_grads, valid_starts)
                accepted_intervals.append(interval)
                continue

            if mid_pos not in computed_grads:
                computed_grads[mid_pos] = self._compute_gradient(fragment, mid_pos)

            # Linear interpolation at midpoint
            grad_start = computed_grads[start]
            grad_end = computed_grads[end]
            alpha = (end - mid_pos) / (end - start)
            beta = (mid_pos - start) / (end - start)
            grad_interp = alpha * grad_start + beta * grad_end

            # Check residuals
            grad_mid = computed_grads[mid_pos]
            if self.residual_computer.should_subdivide(grad_mid, grad_interp):
                # Subdivide: add both sub-intervals to queue
                test_queue.append((start, mid_pos))
                test_queue.append((mid_pos, end))
            else:
                # Accept interval
                interval = self._create_interval(fragment_index, start, end, computed_grads, valid_starts)
                accepted_intervals.append(interval)

        return QuadratureResult(
            fragment_index=fragment_index,
            intervals=accepted_intervals,
            computed_gradients=len(computed_grads),
            total_positions=len(valid_starts),
        )

    def _compute_gradient(self, fragment: TimeSeriesFragment, position: int) -> np.ndarray:
        """Projected gradient for the window at ``position``, cache first."""
        cached = (self.anchor_cache or {}).get((fragment.fragment_id, position))
        if cached is not None:
            return np.asarray(cached, dtype=np.float32)
        window = fragment.extract_window(
            position,
            self.config.context_length,
            self.config.prediction_length
        )
        gradient = self.gradient_computer(window, self.config.context_length, self.config.prediction_length)
        return np.asarray(gradient, dtype=np.float32)

    def _create_interval(
        self,
        fragment_index: int,
        start: int,
        end: int,
        computed_grads: dict[int, np.ndarray],
        valid_starts: np.ndarray,
    ) -> AdaptiveInterval:
        """Create an AdaptiveInterval with computed anchors."""
        # Get anchors that fall within this interval
        anchors = {
            pos: grad for pos, grad in computed_grads.items()
            if start <= pos <= end
        }

        # Calculate interval weight (number of valid positions)
        mask = (valid_starts >= start) & (valid_starts <= end)
        weight = float(np.sum(mask))

        return AdaptiveInterval(
            fragment_index=fragment_index,
            start_position=start,
            end_position=end,
            computed_anchors=anchors,
            interval_weight=weight,
        )

    def compute_interval_contributions(
        self,
        interval: AdaptiveInterval,
        valid_starts: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Compute gradient and weight contributions for an interval using quadrature.

        Approximates Q_{j,q} ≈ Σ_{t∈I_{j,q}} ν_{j,t} g_{j,t} by interpolating
        the gradient at each valid start from the interval's anchors, and
        A_{j,q} = Σ ν_{j,t}. Interpolation is piecewise quadratic (Simpson,
        three-anchor groups) or piecewise linear (trapezoid).

        Returns:
            (Q_jq, A_jq) where Q is the weighted gradient mass and A the
            start-point mass of the interval.
        """
        anchors = sorted(interval.computed_anchors.items())
        if len(anchors) < 2:
            raise ValueError(f"Interval {interval} has fewer than 2 anchors")

        in_interval = (valid_starts >= interval.start_position) & (
            valid_starts <= interval.end_position
        )
        positions = valid_starts[in_interval]

        groups = self._anchor_groups(anchors)
        # Per-anchor interpolation coefficient: Q = Σ_i coeff_i · g_i.
        # Each start belongs to exactly one group (half-open spans, the last
        # group closed), so shared boundary anchors are counted once.
        coefficients = np.zeros(len(anchors), dtype=np.float64)
        for group_index, (node_indices, span) in enumerate(groups):
            left, right = span
            is_last = group_index == len(groups) - 1
            in_span = (positions >= left) & (positions < right if not is_last else positions <= right)
            node_positions = [anchors[i][0] for i in node_indices]
            for t in positions[in_span]:
                weights = _lagrange_weights(float(t), [float(p) for p in node_positions])
                for local_index, weight in zip(node_indices, weights):
                    coefficients[local_index] += weight

        anchor_gradients = np.stack([gradient for _position, gradient in anchors]).astype(np.float64)
        Q = coefficients @ anchor_gradients
        A = float(len(positions))
        return Q.astype(np.float32), A

    def _anchor_groups(self, anchors: list[tuple[int, np.ndarray]]) -> list[tuple[list[int], tuple[int, int]]]:
        """Split anchors into interpolation groups.

        With Simpson enabled, consecutive triples form quadratic groups
        (positions [a_i, a_{i+2}]); a trailing pair degrades to linear.
        Otherwise every consecutive pair is a linear group. Groups share
        their boundary anchor, and the shared start is counted once in the
        final sum because interior positions are assigned to exactly one
        group by the coverage mask in the caller.
        """
        count = len(anchors)
        groups: list[tuple[list[int], tuple[int, int]]] = []
        if self.config.use_simpson:
            index = 0
            while index + 2 < count:
                groups.append(([index, index + 1, index + 2], (anchors[index][0], anchors[index + 2][0])))
                index += 2
            if index + 1 < count:
                groups.append(([index, index + 1], (anchors[index][0], anchors[index + 1][0])))
        else:
            for index in range(count - 1):
                groups.append(([index, index + 1], (anchors[index][0], anchors[index + 1][0])))
        return groups

    def compute_full_reference_gradient(
        self,
        fragment_results: list[QuadratureResult],
        fragments: list[TimeSeriesFragment],
    ) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """Aggregate interval contributions to compute full reference gradient.

        Returns:
            (G, fragment_gradients) where G is the full reference gradient
            and fragment_gradients maps fragment_index to its contribution
        """
        all_Q = []  # Gradient contributions
        all_A = []  # Weight contributions

        fragment_contributions: dict[int, np.ndarray] = {}

        for result, fragment in zip(fragment_results, fragments):
            valid_starts = fragment.valid_starts(
                self.config.context_length,
                self.config.prediction_length
            )

            fragment_Q = np.zeros(self.config.projection_dimension)
            fragment_A = 0.0

            for interval in result.intervals:
                Q, A = self.compute_interval_contributions(interval, valid_starts)
                fragment_Q += Q
                fragment_A += A

            all_Q.append(fragment_Q)
            all_A.append(fragment_A)
            fragment_contributions[result.fragment_index] = fragment_Q / (fragment_A + 1e-8)

        # Aggregate across fragments
        total_Q = sum(all_Q)
        total_A = sum(all_A)

        if total_A == 0:
            return np.zeros(self.config.projection_dimension), fragment_contributions

        G = total_Q / total_A
        return G.astype(np.float32), fragment_contributions


__all__ = [
    "QuadratureResult",
    "MidpointResidualComputer",
    "AdaptiveQuadratureEngine",
]