"""Dependency-preserving reference perturbations for RAGE."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from .core import (
    RAGEConfig,
    AdaptiveInterval,
)
from .quadrature import QuadratureResult


@dataclass
class ReferenceVersion:
    """One perturbed version of the reference gradient.

    Each version represents a different reference construction that still
    preserves time dependencies. Used to measure ranking stability.
    """
    version_id: int
    reference_gradient: np.ndarray  # The perturbed reference gradient
    interval_weights: dict[int, dict[int, float]]  # fragment -> interval -> weight
    total_weight: float

    def __post_init__(self) -> None:
        self.reference_gradient = np.asarray(self.reference_gradient, dtype=np.float32)


@dataclass
class MovingBlockBootstrap:
    """Moving block bootstrap for dependency-preserving resampling.

    Implements the algorithm from Section 4.3 of the RAGE paper:
    - Sample continuous blocks of window starts
    - Preserve order within blocks
    - Maintain time dependencies
    """

    config: RAGEConfig
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.config.perturbation_seed)

    def resample_positions(
        self,
        positions: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resample positions using moving block bootstrap.

        Args:
            positions: Sorted array of valid positions
            weights: Optional weights for each position

        Returns:
            (resampled_positions, resampled_weights)
        """
        if len(positions) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        if weights is None:
            weights = np.ones(len(positions), dtype=np.float32)

        block_length = min(self.config.block_length, len(positions))

        # Create continuous blocks
        num_blocks = int(np.ceil(len(positions) / block_length))
        resampled_pos: list[int] = []
        resampled_weights: list[float] = []

        for i in range(num_blocks):
            # Random block start
            if len(positions) <= block_length:
                # Not enough positions for random sampling, use all
                block_indices = np.arange(len(positions))
            else:
                start_idx = self.rng.integers(0, len(positions) - block_length + 1)
                block_indices = np.arange(start_idx, min(start_idx + block_length, len(positions)))

            block_pos = positions[block_indices]
            block_w = weights[block_indices]

            resampled_pos.extend(block_pos.tolist())
            resampled_weights.extend(block_w.tolist())

        return np.array(resampled_pos, dtype=np.int32), np.array(resampled_weights, dtype=np.float32)

    def block_delete_positions(
        self,
        positions: np.ndarray,
        weights: np.ndarray | None = None,
        delete_fraction: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Delete some continuous blocks to test sensitivity.

        Args:
            positions: Sorted array of valid positions
            weights: Optional weights for each position
            delete_fraction: Fraction of blocks to delete

        Returns:
            (remaining_positions, remaining_weights)
        """
        if len(positions) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        if weights is None:
            weights = np.ones(len(positions), dtype=np.float32)

        block_length = min(self.config.block_length, len(positions))
        num_blocks = int(np.ceil(len(positions) / block_length))

        # Decide which blocks to delete
        keep_prob = 1.0 - delete_fraction
        blocks_to_keep = []
        for i in range(num_blocks):
            if self.rng.random() < keep_prob:
                blocks_to_keep.append(i)

        if not blocks_to_keep:
            # Keep at least one block
            blocks_to_keep = [0]

        # Collect positions from kept blocks
        kept_pos: list[int] = []
        kept_weights: list[float] = []

        for block_idx in blocks_to_keep:
            start_idx = block_idx * block_length
            end_idx = min(start_idx + block_length, len(positions))
            block_indices = np.arange(start_idx, end_idx)

            block_pos = positions[block_indices]
            block_w = weights[block_indices]

            kept_pos.extend(block_pos.tolist())
            kept_weights.extend(block_w.tolist())

        return np.array(kept_pos, dtype=np.int32), np.array(kept_weights, dtype=np.float32)


@dataclass
class ReferencePerturbationEngine:
    """Generate multiple reference versions via dependency-preserving perturbations.

    Implements Section 4.3 of the RAGE paper:
    1. Use moving block bootstrap on continuous window blocks
    2. Handle block boundaries that fall inside adaptive intervals
    3. Reuse interval contributions, just re-weight them
    4. Preserve rare/important states via fragment weights
    """

    config: RAGEConfig
    bootstrap: MovingBlockBootstrap = field(init=False)

    def __post_init__(self) -> None:
        self.bootstrap = MovingBlockBootstrap(self.config)

    def generate_reference_versions(
        self,
        quadrature_results: list[QuadratureResult],
        fragments: list[Any],  # TimeSeriesFragment objects
        valid_starts_list: list[np.ndarray],
    ) -> list[ReferenceVersion]:
        """Generate R perturbed reference versions.

        Args:
            quadrature_results: Results from adaptive quadrature for each fragment
            fragments: Original fragment objects
            valid_starts_list: Valid start positions for each fragment

        Returns:
            List of ReferenceVersion objects
        """
        versions: list[ReferenceVersion] = []

        for r in range(self.config.num_reference_versions):
            # Reset RNG for reproducibility across versions
            version_seed = self.config.perturbation_seed + r
            self.bootstrap.rng = np.random.default_rng(version_seed)

            # Collect interval contributions with re-weighting
            all_Q = []
            all_A = []
            interval_weights: dict[int, dict[int, float]] = {}

            for frag_idx, (result, fragment, valid_starts) in enumerate(
                zip(quadrature_results, fragments, valid_starts_list)
            ):
                Q, A, interval_w = self._perturb_fragment(
                    result, fragment, valid_starts, frag_idx
                )
                all_Q.append(Q)
                all_A.append(A)
                interval_weights[frag_idx] = interval_w

            # Aggregate across fragments
            total_Q = sum(all_Q)
            total_A = sum(all_A)

            if total_A > 0:
                G_r = total_Q / total_A
            else:
                G_r = np.zeros(self.config.projection_dimension)

            version = ReferenceVersion(
                version_id=r,
                reference_gradient=G_r.astype(np.float32),
                interval_weights=interval_weights,
                total_weight=float(total_A),
            )
            versions.append(version)

        return versions

    def _perturb_fragment(
        self,
        result: QuadratureResult,
        fragment: Any,
        valid_starts: np.ndarray,
        fragment_index: int,
    ) -> tuple[np.ndarray, float, dict[int, float]]:
        """Apply perturbation to a single fragment.

        Returns:
            (Q_fragment, A_fragment, interval_weights)
        """
        fragment_Q = np.zeros(self.config.projection_dimension)
        fragment_A = 0.0
        interval_weights: dict[int, float] = {}

        if len(valid_starts) == 0:
            return fragment_Q, fragment_A, interval_weights

        # Resample positions within this fragment
        resampled_pos, resampled_weights = self.bootstrap.resample_positions(
            valid_starts,
        )

        if len(resampled_pos) == 0:
            return fragment_Q, fragment_A, interval_weights

        # Apply resampled weights to intervals
        for interval in result.intervals:
            interval_Q, interval_A, interval_w = self._perturb_interval(
                interval, resampled_pos, resampled_weights, valid_starts, fragment.state_weight
            )
            fragment_Q += interval_Q
            fragment_A += interval_A
            interval_weights[id(interval)] = interval_w

        return fragment_Q, fragment_A, interval_weights

    def _perturb_interval(
        self,
        interval: AdaptiveInterval,
        resampled_pos: np.ndarray,
        resampled_weights: np.ndarray,
        original_valid_starts: np.ndarray,
        state_weight: float,
    ) -> tuple[np.ndarray, float, float]:
        """Perturb a single interval using resampled positions.

        Handles block boundaries that fall inside intervals by splitting
        and using interpolation.

        Returns:
            (Q_interval, A_interval, effective_weight)
        """
        # Find positions from resampled set that fall in this interval
        mask = (resampled_pos >= interval.start_position) & (resampled_pos <= interval.end_position)
        interval_pos = resampled_pos[mask]
        interval_weights = resampled_weights[mask]

        if len(interval_pos) == 0:
            return np.zeros(self.config.projection_dimension), 0.0, 0.0

        # For each resampled position, compute contribution using interpolation
        anchors = sorted(interval.computed_anchors.items())
        if len(anchors) < 2:
            return np.zeros(self.config.projection_dimension), 0.0, 0.0

        Q = np.zeros(self.config.projection_dimension)
        A = 0.0

        for pos, weight in zip(interval_pos, interval_weights):
            # Find enclosing anchors for interpolation
            grad = self._interpolate_gradient(pos, anchors)

            # Apply state weight
            Q += state_weight * weight * grad
            A += state_weight * weight

        return Q, A, float(A)

    def _interpolate_gradient(self, position: int, anchors: list[tuple[int, np.ndarray]]) -> np.ndarray:
        """Linear interpolation between nearest anchors."""
        # Find surrounding anchors
        for i in range(len(anchors) - 1):
            pos_left, grad_left = anchors[i]
            pos_right, grad_right = anchors[i + 1]

            if pos_left <= position <= pos_right:
                if position == pos_left:
                    return grad_left
                elif position == pos_right:
                    return grad_right
                else:
                    # Linear interpolation
                    alpha = (pos_right - position) / (pos_right - pos_left)
                    beta = (position - pos_left) / (pos_right - pos_left)
                    return alpha * grad_left + beta * grad_right

        # If we get here, position is outside anchor range
        # Extrapolate using nearest anchors
        if position < anchors[0][0]:
            return anchors[0][1]
        else:
            return anchors[-1][1]

    def generate_boundary_perturbations(
        self,
        quadrature_results: list[QuadratureResult],
        fragments: list[Any],
        valid_starts_list: list[np.ndarray],
    ) -> list[ReferenceVersion]:
        """Generate perturbations by changing window boundaries.

        Tests sensitivity to window protocol variations.
        """
        versions: list[ReferenceVersion] = []

        # Shift boundaries by ±1, ±2 positions
        for shift in [-2, -1, 1, 2]:
            version_seed = self.config.perturbation_seed + abs(shift) * 100
            self.bootstrap.rng = np.random.default_rng(version_seed)

            shifted_valid_list = []
            for valid_starts in valid_starts_list:
                # Shift valid positions
                shifted = valid_starts + shift
                # Filter to keep valid positions
                shifted = shifted[shifted >= 1]
                shifted_valid_list.append(shifted)

            # Generate version with shifted boundaries
            all_Q = []
            all_A = []

            for result, fragment, shifted_valids in zip(quadrature_results, fragments, shifted_valid_list):
                if len(shifted_valids) == 0:
                    all_Q.append(np.zeros(self.config.projection_dimension))
                    all_A.append(0.0)
                    continue

                Q = np.zeros(self.config.projection_dimension)
                A = 0.0

                for interval in result.intervals:
                    mask = (shifted_valids >= interval.start_position) & (shifted_valids <= interval.end_position)
                    interval_pos = shifted_valids[mask]
                    if len(interval_pos) > 0:
                        A += len(interval_pos)
                        # Simple approximation: use uniform gradient
                        anchors = list(interval.computed_anchors.values())
                        if anchors:
                            avg_grad = np.mean(anchors, axis=0)
                            Q += len(interval_pos) * avg_grad

                all_Q.append(Q)
                all_A.append(A)

            total_Q = sum(all_Q)
            total_A = sum(all_A)

            if total_A > 0:
                G_shifted = total_Q / total_A
            else:
                G_shifted = np.zeros(self.config.projection_dimension)

            version = ReferenceVersion(
                version_id=len(versions),
                reference_gradient=G_shifted.astype(np.float32),
                interval_weights={},
                total_weight=float(total_A),
            )
            versions.append(version)

        return versions


__all__ = [
    "ReferenceVersion",
    "MovingBlockBootstrap",
    "ReferencePerturbationEngine",
]