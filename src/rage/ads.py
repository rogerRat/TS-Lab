"""Adaptive Density Sampling over real source and reference gradients."""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from bisect import bisect_left, bisect_right
import math
import time
from typing import Callable, Mapping, Sequence

import numpy as np

from .core import TimeSeriesFragment


BatchGradientComputer = Callable[[TimeSeriesFragment, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ADSConfig:
    """Configuration for adaptive density sampling.

    Exactly one of top_k, top_ratio and target_fine_density is active.
    Target density is the desired fraction of all legal source and reference
    positions to compute as new fine gradients in each round.
    """

    eta: float
    q: float = 16.0
    top_k: int | None = None
    top_ratio: float | None = None
    target_fine_density: float | None = None
    rounds: int = 1
    decay_distance: int = 4
    aggregation: str = "max"

    def __post_init__(self) -> None:
        for name in ("q", "eta"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError("%s must be finite and positive" % name)
        active = sum(value is not None for value in (
            self.top_k, self.top_ratio, self.target_fine_density))
        if active == 0:
            object.__setattr__(self, "top_ratio", 0.2)
        elif active != 1:
            raise ValueError("exactly one of top_k, top_ratio, and target_fine_density is required")
        if self.top_k is not None and (
                isinstance(self.top_k, bool) or int(self.top_k) != self.top_k
                or self.top_k <= 0):
            raise ValueError("top_k must be a positive integer")
        if self.top_ratio is not None and (
                not np.isfinite(self.top_ratio) or not 0 < self.top_ratio <= 1):
            raise ValueError("top_ratio must be in (0, 1]")
        if self.target_fine_density is not None and (
                not np.isfinite(self.target_fine_density)
                or not 0 < self.target_fine_density <= 1):
            raise ValueError("target_fine_density must be in (0, 1]")
        if isinstance(self.rounds, bool) or int(self.rounds) != self.rounds or self.rounds < 0:
            raise ValueError("rounds must be a non-negative integer")
        if (isinstance(self.decay_distance, bool)
                or int(self.decay_distance) != self.decay_distance
                or self.decay_distance <= 0):
            raise ValueError("decay_distance must be a positive integer")
        if self.aggregation not in ("linear", "max", "exp", "relu"):
            raise ValueError("aggregation must be linear, max, exp, or relu")

    @property
    def source_spacing(self) -> float:
        return self.q * (1.0 + self.eta)

    @property
    def reference_spacing(self) -> float:
        return self.q * (1.0 + self.eta) / self.eta


@dataclass(frozen=True)
class ADSMatch:
    reference_fragment_id: int
    reference_start: int
    source_fragment_id: int
    source_start: int
    similarity: float


@dataclass(frozen=True)
class ADSMatchSummary:
    pair_count: int
    reference_anchor_hits: int
    source_anchor_hits: int
    threshold: float

    def __len__(self) -> int:
        return self.pair_count


@dataclass
class ADSTrajectoryResult:
    fragment_id: int
    valid_starts: np.ndarray
    anchors: dict[int, np.ndarray]
    anchor_scores: dict[int, float] = field(default_factory=dict)
    matched_starts: list[int] = field(default_factory=list)
    candidate_regions: list[tuple[int, int]] = field(default_factory=list)
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    probabilities: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    coarse_gradients: int = 0
    fine_gradients: int = 0


@dataclass
class ADSRunReport:
    reference_results: list[ADSTrajectoryResult]
    source_results: list[ADSTrajectoryResult]
    matches_by_round: list[ADSMatchSummary]
    score_matrix_summaries: list[dict]
    actual_gradient_counts: dict[str, dict[str, int]]
    nominal_gradient_counts: dict
    cached_gradient_counts: dict[str, int]
    zero_norm_anchors: dict[str, int]
    zero_norm_pairs: int
    probability_fallback: bool
    target_fine_density: float | None = None
    achieved_fine_density: float | None = None
    selected_top_k: list[int] = field(default_factory=list)
    decay_distance: int = 4

    @property
    def source_probabilities(self) -> dict[int, np.ndarray]:
        return {result.fragment_id: result.probabilities for result in self.source_results}

    @property
    def source_weights(self) -> dict[int, np.ndarray]:
        return {result.fragment_id: result.weights for result in self.source_results}


@dataclass
class _TrajectoryState:
    fragment: TimeSeriesFragment
    valid_starts: np.ndarray
    anchors: dict[int, np.ndarray] = field(default_factory=dict)
    coarse_gradients: int = 0
    fine_gradients: int = 0
    matched: set[int] = field(default_factory=set)
    regions: list[tuple[int, int]] = field(default_factory=list)
    anchor_scores: dict[int, float] = field(default_factory=dict)
    weights: np.ndarray | None = None
    probabilities: np.ndarray | None = None

    @property
    def fragment_id(self) -> int:
        return int(self.fragment.fragment_id)


def _identifier_key(identifier: object) -> tuple:
    if isinstance(identifier, (int, np.integer)):
        return (0, int(identifier))
    return (1, str(identifier))


def uniform_anchor_positions(length: int, target_spacing: float) -> list[int]:
    """Return evenly spread zero-based anchors, always including both endpoints."""
    if length <= 0:
        return []
    if not np.isfinite(target_spacing) or target_spacing <= 0:
        raise ValueError("target_spacing must be finite and positive")
    if length == 1:
        return [0]
    interior_count = length - 2
    wanted = min(interior_count, int(math.ceil(interior_count / target_spacing)))
    if wanted == 0:
        return [0, length - 1]
    chosen: list[int] = []
    if wanted == 1:
        numerator, denominator = length - 1, 2
        position, remainder = divmod(numerator, denominator)
        if 2 * remainder > denominator:
            position += 1
        chosen.append(position)
    else:
        denominator = wanted - 1
        for index in range(wanted):
            numerator = denominator + index * (interior_count - 1)
            position, remainder = divmod(numerator, denominator)
            if 2 * remainder > denominator:
                position += 1
            chosen.append(position)
    return sorted({0, length - 1, *chosen})


def cosine_similarity_matrix(reference_gradients: np.ndarray,
                              source_gradients: np.ndarray) -> tuple[np.ndarray, int]:
    """Cosine matrix with zero-vector pairs set to zero."""
    reference = np.asarray(reference_gradients, dtype=np.float32)
    source = np.asarray(source_gradients, dtype=np.float32)
    if reference.ndim != 2 or source.ndim != 2:
        raise ValueError("gradient matrices must be two dimensional")
    if reference.shape[1] == 0 or source.shape[1] == 0:
        raise ValueError("gradient dimensions must be positive")
    if reference.shape[1] != source.shape[1]:
        raise ValueError("reference and source gradient dimensions must match")
    if not np.isfinite(reference).all() or not np.isfinite(source).all():
        raise ValueError("gradient matrices must be finite")
    reference_norms = np.linalg.norm(reference, axis=1)
    source_norms = np.linalg.norm(source, axis=1)
    reference_nonzero = reference_norms > 0
    source_nonzero = source_norms > 0
    zero_pairs = int((~reference_nonzero).sum() * len(source)
                     + reference_nonzero.sum() * (~source_nonzero).sum())
    pair_count = int(len(reference) * len(source))
    use_cuda = pair_count >= 10_000_000
    if use_cuda:
        try:
            import torch
            use_cuda = torch.cuda.is_available()
        except ImportError:
            use_cuda = False
    if use_cuda:
        import torch
        matrix = np.empty((len(reference), len(source)), dtype=np.float32)
        device = torch.device("cuda")
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            with torch.inference_mode():
                source_tensor = torch.as_tensor(source, device=device)
                source_norm = torch.linalg.vector_norm(source_tensor, dim=1, keepdim=True)
                source_tensor = torch.where(
                    source_norm > 0, source_tensor / source_norm.clamp_min(1e-30),
                    torch.zeros_like(source_tensor))
                row_chunk = 128
                for begin in range(0, len(reference), row_chunk):
                    end = min(begin + row_chunk, len(reference))
                    reference_tensor = torch.as_tensor(reference[begin:end], device=device)
                    reference_norm = torch.linalg.vector_norm(reference_tensor, dim=1, keepdim=True)
                    reference_tensor = torch.where(
                        reference_norm > 0,
                        reference_tensor / reference_norm.clamp_min(1e-30),
                        torch.zeros_like(reference_tensor))
                    scores = torch.matmul(reference_tensor, source_tensor.T)
                    matrix[begin:end] = scores.cpu().numpy()
                    del reference_tensor, reference_norm, scores
                del source_tensor, source_norm
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            torch.cuda.empty_cache()
    else:
        normalized_reference = np.zeros_like(reference)
        normalized_source = np.zeros_like(source)
        normalized_reference[reference_nonzero] = (
            reference[reference_nonzero] / reference_norms[reference_nonzero, None])
        normalized_source[source_nonzero] = (
            source[source_nonzero] / source_norms[source_nonzero, None])
        matrix = normalized_reference @ normalized_source.T
    return matrix.astype(np.float32, copy=False), zero_pairs


def global_top_k(similarities: np.ndarray, top_k: int, *,
                 reference_keys: Sequence[tuple] | None = None,
                 source_keys: Sequence[tuple] | None = None) -> list[tuple[int, int, float]]:
    """Select a deterministic global Top-K; score ties use the supplied keys."""
    matrix = np.asarray(similarities)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("similarity matrix must be nonempty and two dimensional")
    total = int(matrix.size)
    if top_k < 1 or top_k > total:
        raise ValueError("top_k must be between 1 and the number of pairs")
    ref_keys = list(reference_keys or [(row,) for row in range(matrix.shape[0])])
    src_keys = list(source_keys or [(column,) for column in range(matrix.shape[1])])
    if len(ref_keys) != matrix.shape[0] or len(src_keys) != matrix.shape[1]:
        raise ValueError("tie-break keys must match the similarity matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("similarity matrix must be finite")
    flat = matrix.reshape(-1)
    threshold = np.partition(flat, total - top_k)[total - top_k]
    greater = np.flatnonzero(flat > threshold)
    needed_equal = top_k - len(greater)
    greater_pairs = [
        (int(index // matrix.shape[1]), int(index % matrix.shape[1]),
         float(flat[index])) for index in greater
    ]
    greater_pairs.sort(
        key=lambda item: (-item[2], ref_keys[item[0]], src_keys[item[1]]))

    reference_order = sorted(range(matrix.shape[0]), key=ref_keys.__getitem__)
    source_rank = np.empty(matrix.shape[1], dtype=np.int64)
    for rank, column in enumerate(sorted(range(matrix.shape[1]), key=src_keys.__getitem__)):
        source_rank[column] = rank
    equal_grid = (matrix == threshold)
    equal_pairs: list[tuple[int, int, float]] = []
    for row in reference_order:
        columns = np.flatnonzero(equal_grid[row])
        if len(columns):
            columns = columns[np.argsort(source_rank[columns], kind="stable")]
            take = min(needed_equal, len(columns))
            equal_pairs.extend((row, int(column), float(threshold))
                               for column in columns[:take])
            needed_equal -= take
            if needed_equal == 0:
                break
    return greater_pairs + equal_pairs


def global_top_k_hit_masks(similarities: np.ndarray, top_k: int, *,
                           reference_keys: Sequence[tuple] | None = None,
                           source_keys: Sequence[tuple] | None = None
                           ) -> tuple[np.ndarray, np.ndarray, float]:
    """Return endpoints touched by deterministic global Top-K pairs."""
    matrix = np.asarray(similarities)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("similarity matrix must be nonempty and two dimensional")
    total = int(matrix.size)
    if top_k < 1 or top_k > total:
        raise ValueError("top_k must be between 1 and the number of pairs")
    ref_keys = list(reference_keys or [(row,) for row in range(matrix.shape[0])])
    src_keys = list(source_keys or [(column,) for column in range(matrix.shape[1])])
    if len(ref_keys) != matrix.shape[0] or len(src_keys) != matrix.shape[1]:
        raise ValueError("tie-break keys must match the similarity matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("similarity matrix must be finite")
    flat = matrix.reshape(-1)
    threshold = float(np.partition(flat, total - top_k)[total - top_k])
    needed_equal = top_k - int(np.count_nonzero(flat > threshold))
    reference_hits = np.zeros(matrix.shape[0], dtype=bool)
    source_hits = np.zeros(matrix.shape[1], dtype=bool)
    for row in range(matrix.shape[0]):
        columns = np.flatnonzero(matrix[row] > threshold)
        if len(columns):
            reference_hits[row] = True
            source_hits[columns] = True
    if needed_equal:
        reference_order = sorted(range(matrix.shape[0]), key=ref_keys.__getitem__)
        source_rank = np.empty(matrix.shape[1], dtype=np.int64)
        for rank, column in enumerate(sorted(range(matrix.shape[1]), key=src_keys.__getitem__)):
            source_rank[column] = rank
        for row in reference_order:
            columns = np.flatnonzero(matrix[row] == threshold)
            if not len(columns):
                continue
            columns = columns[np.argsort(source_rank[columns], kind="stable")]
            take = min(needed_equal, len(columns))
            source_hits[columns[:take]] = True
            reference_hits[row] = True
            needed_equal -= take
            if needed_equal == 0:
                break
    if needed_equal != 0:
        raise RuntimeError("Top-K tie selection did not reach its requested pair count")
    return reference_hits, source_hits, threshold


def summarize_similarity_matrix(similarities: np.ndarray, top_k: int, *,
                                threshold: float, sample_size: int = 2_000_000,
                                seed: int = 42) -> dict:
    """Summarize exact sparsity and sampled score quantiles for a dense matrix."""
    matrix = np.asarray(similarities, dtype=np.float32)
    if matrix.ndim != 2 or matrix.size == 0 or not np.isfinite(matrix).all():
        raise ValueError("similarity matrix must be finite, nonempty, and two dimensional")
    flat = matrix.reshape(-1)
    total = int(flat.size)
    zero_count = near_1_count = near_3_count = positive_count = negative_count = 0
    total_sum = total_sq = 0.0
    minimum, maximum = float("inf"), -float("inf")
    scan_chunk = 5_000_000
    for begin in range(0, total, scan_chunk):
        block = flat[begin:begin + scan_chunk]
        absolute = np.abs(block)
        zero_count += int(np.count_nonzero(block == 0.0))
        near_1_count += int(np.count_nonzero(absolute <= 1e-8))
        near_3_count += int(np.count_nonzero(absolute <= 1e-3))
        positive_count += int(np.count_nonzero(block > 0.0))
        negative_count += int(np.count_nonzero(block < 0.0))
        total_sum += float(np.sum(block, dtype=np.float64))
        total_sq += float(np.sum(block * block, dtype=np.float64))
        minimum = min(minimum, float(block.min()))
        maximum = max(maximum, float(block.max()))
    mean = total_sum / total
    variance = max(total_sq / total - mean * mean, 0.0)
    rng = np.random.default_rng(seed)
    take = min(total, int(sample_size))
    sample_indices = (np.arange(total, dtype=np.int64) if take == total else
                      rng.integers(0, total, size=take, dtype=np.int64))
    sample = flat[sample_indices]
    levels = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    quantiles = np.quantile(sample, levels)
    return {
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "pair_count": total,
        "raw_exact_zero_count": zero_count,
        "raw_exact_zero_fraction": float(zero_count / total),
        "raw_abs_le_1e-8_count": near_1_count,
        "raw_abs_le_1e-8_fraction": float(near_1_count / total),
        "raw_abs_le_1e-3_count": near_3_count,
        "raw_abs_le_1e-3_fraction": float(near_3_count / total),
        "raw_positive_fraction": float(positive_count / total),
        "raw_negative_fraction": float(negative_count / total),
        "raw_mean": mean,
        "raw_std": float(math.sqrt(variance)),
        "raw_min": minimum,
        "raw_max": maximum,
        "quantile_sample_count": take,
        "quantile_sampling": "uniform_with_replacement",
        "quantiles": {"p%02d" % int(level * 100): float(value)
                      for level, value in zip(levels, quantiles)},
        "top_k_pairs": int(top_k),
        "top_k_threshold": float(threshold),
        "top_k_fraction": float(top_k / total),
        "matching_mask_zero_fraction": float(1.0 - top_k / total),
        "quantile_seed": int(seed),
    }


def aggregate_source_scores(similarities: np.ndarray, mode: str = "max") -> np.ndarray:
    """Aggregate reference rows for each source column."""
    matrix = np.asarray(similarities, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("similarities must be a nonempty matrix")
    if mode == "linear":
        return matrix.sum(axis=0)
    if mode == "max":
        return matrix.max(axis=0)
    if mode == "exp":
        return np.exp(matrix).sum(axis=0)
    if mode == "relu":
        return np.maximum(matrix, 0).sum(axis=0)
    raise ValueError("aggregation must be linear, max, exp, or relu")


def spread_anchor_scores(length: int, anchor_scores: dict[int, float],
                         decay_distance: int) -> np.ndarray:
    """Apply finite linear decay and max-merge overlapping anchor neighborhoods."""
    if length < 0 or decay_distance <= 0:
        raise ValueError("length must be non-negative and decay_distance positive")
    weights = np.zeros(length, dtype=np.float64)
    for anchor_position, score in anchor_scores.items():
        if anchor_position < 0 or anchor_position >= length:
            raise ValueError("anchor position outside trajectory")
        if not np.isfinite(score):
            raise ValueError("anchor scores must be finite")
        anchor_weight = max(0.0, float(score))
        lower = max(0, anchor_position - decay_distance + 1)
        upper = min(length, anchor_position + decay_distance)
        positions = np.arange(lower, upper, dtype=np.int64)
        contribution = anchor_weight * (
            1.0 - np.abs(positions - anchor_position) / decay_distance)
        weights[positions] = np.maximum(weights[positions], contribution)
    return weights


def normalize_sampling_probabilities(weights: np.ndarray) -> tuple[np.ndarray, bool]:
    """Normalize nonnegative weights, falling back to a uniform distribution."""
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("weights must be a nonempty vector")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("weights must be finite and non-negative")
    total = float(values.sum())
    fallback = total <= 0
    if fallback:
        return np.full(len(values), 1.0 / len(values), dtype=np.float64), True
    return values / total, False


def _merge_regions(regions: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not regions:
        return []
    merged: list[list[int]] = []
    for left, right in sorted(regions):
        if merged and left <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return [(left, right) for left, right in merged]


def _regions_for_hits(state: _TrajectoryState, matched: set[int]) -> list[tuple[int, int]]:
    if not matched:
        return []
    hit_positions = sorted(matched)
    runs: list[tuple[int, int]] = []
    left = right = hit_positions[0]
    for position in hit_positions[1:]:
        if position == right + 1:
            right = position
        else:
            runs.append((left, right))
            left = right = position
    runs.append((left, right))

    regions: list[tuple[int, int]] = []
    unmatched = sorted(set(state.anchors) - matched)
    last = len(state.valid_starts) - 1
    for left, right in runs:
        left_index = bisect_left(unmatched, left)
        right_index = bisect_right(unmatched, right)
        left_miss = unmatched[left_index - 1] if left_index else None
        right_miss = unmatched[right_index] if right_index < len(unmatched) else None
        expanded_left = (0 if left_miss is None
                         else left - (left - left_miss) // 2)
        expanded_right = (last if right_miss is None
                          else right + (right_miss - right) // 2)
        regions.append((max(0, expanded_left), min(last, expanded_right)))
    return _merge_regions(regions)


def _fine_anchor_plan(states: list[_TrajectoryState],
                      matched_by_state: dict[int, set[int]],
                      target_spacing: float) -> dict[int, set[int]]:
    """Plan new fine anchors using mandatory endpoints and longest-gap bisection."""
    additions: dict[int, set[int]] = {id(state): set() for state in states}
    goals: dict[int, int] = {}
    by_identity = {id(state): state for state in states}
    heap: list[tuple] = []

    def push_gap(state: _TrajectoryState, left: int, right: int,
                 region: tuple[int, int]) -> None:
        width = right - left
        if width <= 1:
            return
        midpoint = (left + right) // 2
        heapq.heappush(heap, (
            -width, int(state.valid_starts[midpoint]),
            _identifier_key(state.fragment_id), left, right, id(state), region,
        ))

    for state in states:
        state_key = id(state)
        regions = _regions_for_hits(state, matched_by_state.get(state_key, set()))
        state.regions = regions
        candidate_count = sum(right - left + 1 for left, right in regions)
        if not regions:
            goals[state_key] = 0
            continue
        nominal_target = int(math.ceil(candidate_count / target_spacing))
        for left, right in regions:
            for endpoint in {left, right}:
                if endpoint not in state.anchors:
                    additions[state_key].add(endpoint)
        goals[state_key] = max(nominal_target, len(additions[state_key]))
        for left, right in regions:
            positions = sorted(
                {position for position in state.anchors if left <= position <= right}
                | {position for position in additions[state_key]
                   if left <= position <= right}
                | {left, right})
            for gap_left, gap_right in zip(positions, positions[1:]):
                push_gap(state, gap_left, gap_right, (left, right))

    while heap:
        _neg_width, _time, _id_order, left, right, state_key, region = heapq.heappop(heap)
        state = by_identity[state_key]
        if len(additions[state_key]) >= goals[state_key]:
            continue
        midpoint = (left + right) // 2
        additions[state_key].add(midpoint)
        push_gap(state, left, midpoint, region)
        push_gap(state, midpoint, right, region)
    return additions


def _matrix_inputs(states: list[_TrajectoryState]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    gradients: list[np.ndarray] = []
    locations: list[tuple[int, int]] = []
    for state_index, state in enumerate(states):
        for position in sorted(state.anchors):
            gradients.append(state.anchors[position])
            locations.append((state_index, position))
    if not gradients:
        raise ValueError("ADS requires at least one computed gradient per side")
    dimensions = {len(gradient) for gradient in gradients}
    if len(dimensions) != 1:
        raise ValueError("all gradients on one side must have the same dimension")
    return np.stack(gradients).astype(np.float32, copy=False), locations


class ADSPipeline:
    """Compute ADS anchors and probabilities.

    Optional cache keys are (side, fragment_id, physical_start). The cache is
    read-only and may include off-grid rows for later fine-anchor requests.
    Supplied coarse positions map (side, fragment_id) to zero-based legal-start
    indices and must include both endpoints when a trajectory has multiple starts.
    """

    def __init__(self, config: ADSConfig, gradient_computer: BatchGradientComputer, *,
                 context_length: int = 1, prediction_length: int = 1,
                 window_stride: int = 1, gradient_batch_size: int = 128,
                 progress_callback: Callable[[dict], None] | None = None,
                 heartbeat_seconds: float = 30.0,
                 gradient_cache: Mapping[tuple[str, int, int], np.ndarray] | None = None,
                 coarse_anchor_positions: Mapping[tuple[str, int], Sequence[int]] | None = None):
        if context_length <= 0 or prediction_length <= 0 or window_stride <= 0:
            raise ValueError("window lengths and stride must be positive")
        if gradient_batch_size <= 0:
            raise ValueError("gradient_batch_size must be positive")
        if not np.isfinite(heartbeat_seconds) or heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be finite and positive")
        self.config = config
        self.gradient_computer = gradient_computer
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_stride = window_stride
        self.gradient_batch_size = gradient_batch_size
        self.progress_callback = progress_callback
        self.heartbeat_seconds = heartbeat_seconds
        self._last_heartbeat = 0.0
        self.gradient_cache = gradient_cache
        self.coarse_anchor_positions = coarse_anchor_positions
        self.cached_gradient_counts = {"reference": 0, "source": 0}

    def _planned_coarse_positions(self, state: _TrajectoryState, side: str) -> set[int]:
        key = (side, state.fragment_id)
        supplied = None if self.coarse_anchor_positions is None else self.coarse_anchor_positions.get(key)
        if supplied is None:
            spacing = (self.config.reference_spacing if side == "reference"
                       else self.config.source_spacing)
            positions = set(uniform_anchor_positions(len(state.valid_starts), spacing))
        else:
            positions = {int(position) for position in supplied}
        if not positions or min(positions) < 0 or max(positions) >= len(state.valid_starts):
            raise ValueError("coarse anchor positions must be nonempty and within the legal range")
        if len(state.valid_starts) > 1 and not {0, len(state.valid_starts) - 1}.issubset(positions):
            raise ValueError("coarse anchor positions must include both endpoints")
        return positions

    def _emit(self, stage: str, event: str, **details) -> None:
        if self.progress_callback is None:
            return
        now = time.monotonic()
        if event in ("start", "end") or now - self._last_heartbeat >= self.heartbeat_seconds:
            self.progress_callback({"stage": stage, "event": event, **details})
            self._last_heartbeat = now

    def _states(self, fragments: list[TimeSeriesFragment],
                side: str) -> list[_TrajectoryState]:
        identifiers = [int(fragment.fragment_id) for fragment in fragments]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("fragment_id values must be unique within a side")
        states = []
        for fragment in fragments:
            starts = fragment.valid_starts(
                self.context_length, self.prediction_length,
                self.window_stride).astype(np.int64, copy=False)
            state = _TrajectoryState(fragment=fragment, valid_starts=starts)
            if self.gradient_cache is not None and len(starts):
                for position in self._planned_coarse_positions(state, side):
                    start = int(starts[position])
                    key = (side, int(fragment.fragment_id), start)
                    if key not in self.gradient_cache:
                        continue
                    gradient = np.asarray(self.gradient_cache[key], dtype=np.float32)
                    if gradient.ndim != 1 or len(gradient) == 0 or not np.isfinite(gradient).all():
                        raise ValueError("cached gradients must be nonempty finite vectors")
                    state.anchors[int(position)] = gradient.copy()
                    self.cached_gradient_counts[side] += 1
            states.append(state)
        states = [state for state in states if len(state.valid_starts)]
        if not states:
            raise ValueError("each side must contain at least one legal position")
        return states

    def _compute(self, states: list[_TrajectoryState], positions_by_state: dict[int, set[int]],
                 *, coarse: bool, side: str) -> int:
        computed = 0
        phase = "coarse" if coarse else "fine"
        requested_by_state = {
            id(state): sorted(positions_by_state.get(id(state), set()) - set(state.anchors))
            for state in states
        }
        total_requested = sum(len(values) for values in requested_by_state.values())
        completed = 0
        started = time.monotonic()
        for state in states:
            requested = requested_by_state[id(state)]
            compute_positions = []
            for position in requested:
                cache_key = (side, state.fragment_id, int(state.valid_starts[position]))
                if self.gradient_cache is None or cache_key not in self.gradient_cache:
                    compute_positions.append(position)
                    continue
                gradient = np.asarray(self.gradient_cache[cache_key], dtype=np.float32)
                if gradient.ndim != 1 or len(gradient) == 0 or not np.isfinite(gradient).all():
                    raise ValueError("cached gradients must be nonempty finite vectors")
                dimensions = {len(value) for value in state.anchors.values()}
                if dimensions and len(gradient) not in dimensions:
                    raise ValueError("cached gradient dimensions changed within a trajectory")
                state.anchors[position] = gradient.copy()
                self.cached_gradient_counts[side] += 1
                completed += 1
            for offset in range(0, len(compute_positions), self.gradient_batch_size):
                chunk = compute_positions[offset:offset + self.gradient_batch_size]
                starts = state.valid_starts[np.asarray(chunk, dtype=np.int64)]
                gradients = np.asarray(self.gradient_computer(state.fragment, starts),
                                       dtype=np.float32)
                if (gradients.ndim != 2 or gradients.shape[0] != len(chunk)
                        or gradients.shape[1] == 0):
                    raise ValueError("gradient computer returned a matrix with the wrong shape")
                if not np.isfinite(gradients).all():
                    raise ValueError("gradient computer returned nonfinite values")
                dimensions = {len(gradient) for gradient in state.anchors.values()}
                if dimensions and gradients.shape[1] not in dimensions:
                    raise ValueError("gradient dimensions changed within a trajectory")
                for position, gradient in zip(chunk, gradients):
                    state.anchors[int(position)] = gradient.copy()
                count = len(chunk)
                if coarse:
                    state.coarse_gradients += count
                else:
                    state.fine_gradients += count
                computed += count
                completed += count
                elapsed = time.monotonic() - started
                eta_seconds = (elapsed * max(total_requested - completed, 0) / completed
                               if completed else None)
                self._emit(
                    phase, "progress", side=side, completed=completed,
                    total=total_requested, fragment_id=state.fragment_id,
                    eta_seconds=eta_seconds)
        return computed

    def _similarities(self, reference: list[_TrajectoryState],
                      source: list[_TrajectoryState]):
        reference_sorted = sorted(reference, key=lambda item: _identifier_key(item.fragment_id))
        source_sorted = sorted(source, key=lambda item: _identifier_key(item.fragment_id))
        reference_gradients, reference_locations = _matrix_inputs(reference_sorted)
        source_gradients, source_locations = _matrix_inputs(source_sorted)
        if reference_gradients.shape[1] != source_gradients.shape[1]:
            raise ValueError("reference and source gradient dimensions must match")
        matrix, zero_pairs = cosine_similarity_matrix(reference_gradients, source_gradients)
        reference_keys = [
            (_identifier_key(reference_sorted[state_index].fragment_id),
             int(reference_sorted[state_index].valid_starts[position]))
            for state_index, position in reference_locations
        ]
        source_keys = [
            (_identifier_key(source_sorted[state_index].fragment_id),
             int(source_sorted[state_index].valid_starts[position]))
            for state_index, position in source_locations
        ]
        return (matrix, reference_sorted, source_sorted, reference_locations,
                source_locations, reference_keys, source_keys, zero_pairs)

    def _target_top_k(self, matrix: np.ndarray, reference: list[_TrajectoryState],
                      source: list[_TrajectoryState], reference_locations,
                      source_locations, target_density: float) -> tuple[int, int]:
        flat = matrix.reshape(-1)
        order = np.argsort(-flat, kind="stable")
        total_positions = sum(len(state.valid_starts) for state in reference + source)
        reference_hits = {id(state): set() for state in reference}
        source_hits = {id(state): set() for state in source}
        seen_reference: set[int] = set()
        seen_source: set[int] = set()
        best_k = 1
        best_count = 0
        best_difference = float("inf")
        target_count = target_density * total_positions
        for rank, flat_index in enumerate(order, start=1):
            row = int(flat_index // matrix.shape[1])
            column = int(flat_index % matrix.shape[1])
            reference_state_index, reference_position = reference_locations[row]
            source_state_index, source_position = source_locations[column]
            changed = False
            if row not in seen_reference:
                seen_reference.add(row)
                reference_hits[id(reference[reference_state_index])].add(reference_position)
                changed = True
            if column not in seen_source:
                seen_source.add(column)
                source_hits[id(source[source_state_index])].add(source_position)
                changed = True
            if not changed and rank != 1:
                continue
            reference_plan = _fine_anchor_plan(
                reference, reference_hits, self.config.reference_spacing)
            source_plan = _fine_anchor_plan(source, source_hits, self.config.source_spacing)
            count = sum(map(len, reference_plan.values())) + sum(map(len, source_plan.values()))
            difference = abs(count - target_count)
            if difference < best_difference:
                best_k, best_count, best_difference = rank, count, difference
            if len(seen_reference) == matrix.shape[0] and len(seen_source) == matrix.shape[1]:
                break
        return best_k, best_count

    @staticmethod
    def _mark_top_k_matches(reference: list[_TrajectoryState], source: list[_TrajectoryState],
                            reference_locations, source_locations, matrix: np.ndarray,
                            top_k: int, reference_keys, source_keys) -> ADSMatchSummary:
        reference_mask, source_mask, threshold = global_top_k_hit_masks(
            matrix, top_k, reference_keys=reference_keys, source_keys=source_keys)
        reference_hits = {id(state): set() for state in reference}
        source_hits = {id(state): set() for state in source}
        for row, hit in enumerate(reference_mask):
            if hit:
                state_index, position = reference_locations[row]
                reference_hits[id(reference[state_index])].add(position)
        for column, hit in enumerate(source_mask):
            if hit:
                state_index, position = source_locations[column]
                source_hits[id(source[state_index])].add(position)
        for state in reference:
            state.matched = reference_hits[id(state)]
        for state in source:
            state.matched = source_hits[id(state)]
        return ADSMatchSummary(
            pair_count=int(top_k),
            reference_anchor_hits=int(reference_mask.sum()),
            source_anchor_hits=int(source_mask.sum()),
            threshold=threshold,
        )

    def run(self, reference_fragments: list[TimeSeriesFragment],
            source_fragments: list[TimeSeriesFragment]) -> ADSRunReport:
        self.cached_gradient_counts = {"reference": 0, "source": 0}
        reference = self._states(reference_fragments, "reference")
        source = self._states(source_fragments, "source")
        coarse_reference = {
            id(state): self._planned_coarse_positions(state, "reference")
            for state in reference
        }
        coarse_source = {
            id(state): self._planned_coarse_positions(state, "source")
            for state in source
        }
        coarse_nominal_internal = {
            "reference": sum(
                min(max(len(state.valid_starts) - 2, 0),
                    int(math.ceil(max(len(state.valid_starts) - 2, 0)
                                  / (self.config.reference_spacing))))
                for state in reference),
            "source": sum(
                min(max(len(state.valid_starts) - 2, 0),
                    int(math.ceil(max(len(state.valid_starts) - 2, 0)
                                  / self.config.source_spacing)))
                for state in source),
        }
        coarse_target_total = {
            "reference": sum(len(positions) for positions in coarse_reference.values()),
            "source": sum(len(positions) for positions in coarse_source.values()),
        }
        fine_nominal_by_round: list[dict[str, int]] = []
        for side, states, positions in (
            ("reference", reference, coarse_reference),
            ("source", source, coarse_source),
        ):
            self._emit("coarse", "start", side=side, trajectories=len(states))
            self._compute(states, positions, coarse=True, side=side)
            self._emit("coarse", "end", side=side)

        matches_by_round: list[ADSMatchSummary] = []
        score_matrix_summaries: list[dict] = []
        selected_top_k: list[int] = []
        achieved_density = None
        for _round in range(self.config.rounds):
            self._emit("matching", "start", round=_round + 1)
            (matrix, reference_sorted, source_sorted, reference_locations,
             source_locations, reference_keys, source_keys, _zero_pairs) = (
                self._similarities(reference, source))
            if self.config.target_fine_density is not None:
                top_k, planned_count = self._target_top_k(
                    matrix, reference_sorted, source_sorted,
                    reference_locations, source_locations,
                    self.config.target_fine_density)
                achieved_density = planned_count / sum(
                    len(state.valid_starts) for state in reference + source)
            elif self.config.top_k is not None:
                top_k = self.config.top_k
            else:
                top_k = int(math.ceil(float(matrix.size) * float(self.config.top_ratio)))
            match = self._mark_top_k_matches(
                reference_sorted, source_sorted, reference_locations, source_locations,
                matrix, top_k, reference_keys, source_keys)
            score_matrix_summaries.append(dict(
                stage="coarse_round_%d" % (_round + 1),
                **summarize_similarity_matrix(matrix, top_k, threshold=match.threshold)))
            selected_top_k.append(top_k)
            matches_by_round.append(match)
            reference_hits = {id(state): state.matched for state in reference_sorted}
            source_hits = {id(state): state.matched for state in source_sorted}
            fine_reference = _fine_anchor_plan(
                reference_sorted, reference_hits, self.config.reference_spacing)
            fine_source = _fine_anchor_plan(
                source_sorted, source_hits, self.config.source_spacing)
            self._emit("matching", "end", round=len(matches_by_round),
                       top_k=top_k, pairs=len(match),
                       reference_anchor_hits=match.reference_anchor_hits,
                       source_anchor_hits=match.source_anchor_hits)
            fine_nominal_by_round.append({
                "reference": sum(
                    int(math.ceil(sum(right - left + 1 for left, right in state.regions)
                                  / (self.config.reference_spacing)))
                    for state in reference_sorted if state.regions),
                "source": sum(
                    int(math.ceil(sum(right - left + 1 for left, right in state.regions)
                                  / self.config.source_spacing))
                    for state in source_sorted if state.regions),
            })
            self._emit("fine", "start", round=len(matches_by_round))
            new_reference = self._compute(
                reference_sorted, fine_reference, coarse=False, side="reference")
            new_source = self._compute(
                source_sorted, fine_source, coarse=False, side="source")
            self._emit("fine", "end", round=len(matches_by_round),
                       reference=new_reference, source=new_source)
            if new_reference + new_source == 0:
                break

        self._emit("matching", "start", round=len(matches_by_round) + 1)
        (matrix, reference_sorted, source_sorted, reference_locations,
         source_locations, reference_keys, source_keys, zero_pairs) = (
            self._similarities(reference, source))
        if self.config.target_fine_density is not None:
            final_top_k, final_planned_count = self._target_top_k(
                matrix, reference_sorted, source_sorted,
                reference_locations, source_locations,
                self.config.target_fine_density)
            achieved_density = final_planned_count / sum(
                len(state.valid_starts) for state in reference + source)
        elif self.config.top_k is not None:
            final_top_k = self.config.top_k
        else:
            final_top_k = int(math.ceil(
                float(matrix.size) * float(self.config.top_ratio)))
        final_match = self._mark_top_k_matches(
            reference_sorted, source_sorted, reference_locations, source_locations,
            matrix, final_top_k, reference_keys, source_keys)
        score_matrix_summaries.append(dict(
            stage="final", **summarize_similarity_matrix(
                matrix, final_top_k, threshold=final_match.threshold)))
        selected_top_k.append(final_top_k)
        matches_by_round.append(final_match)
        self._emit("matching", "end", round=len(matches_by_round),
                   top_k=final_top_k, pairs=len(final_match),
                   reference_anchor_hits=final_match.reference_anchor_hits,
                   source_anchor_hits=final_match.source_anchor_hits)
        self._emit("scoring", "start", source_anchors=len(source_locations))
        source_scores = aggregate_source_scores(matrix, self.config.aggregation)
        for state in reference + source:
            state.anchor_scores.clear()
        for column, (state_index, position) in enumerate(source_locations):
            state = source_sorted[state_index]
            state.anchor_scores[position] = float(source_scores[column])
        for state in source:
            state.weights = spread_anchor_scores(
                len(state.valid_starts), state.anchor_scores,
                self.config.decay_distance)

        all_source_weights = np.concatenate([state.weights for state in source])
        all_probabilities, probability_fallback = normalize_sampling_probabilities(
            all_source_weights)
        offset = 0
        for state in source:
            stop = offset + len(state.valid_starts)
            state.probabilities = all_probabilities[offset:stop].copy()
            offset = stop
        self._emit("scoring", "end", probability_fallback=probability_fallback)

        zero_reference = sum(
            int(np.linalg.norm(gradient) == 0)
            for state in reference for gradient in state.anchors.values())
        zero_source = sum(
            int(np.linalg.norm(gradient) == 0)
            for state in source for gradient in state.anchors.values())
        actual_counts = {
            "coarse": {
                "reference": sum(state.coarse_gradients for state in reference),
                "source": sum(state.coarse_gradients for state in source),
            },
            "fine": {
                "reference": sum(state.fine_gradients for state in reference),
                "source": sum(state.fine_gradients for state in source),
            },
        }
        for phase in actual_counts.values():
            phase["total"] = phase["reference"] + phase["source"]
        reference_results = [
            ADSTrajectoryResult(
                fragment_id=state.fragment_id,
                valid_starts=state.valid_starts.copy(),
                anchors={int(state.valid_starts[position]): gradient.copy()
                         for position, gradient in sorted(state.anchors.items())},
                matched_starts=[int(state.valid_starts[position])
                                for position in sorted(state.matched)],
                candidate_regions=[
                    (int(state.valid_starts[left]), int(state.valid_starts[right]))
                    for left, right in state.regions
                ],
                coarse_gradients=state.coarse_gradients,
                fine_gradients=state.fine_gradients,
            )
            for state in reference
        ]
        source_results = [
            ADSTrajectoryResult(
                fragment_id=state.fragment_id,
                valid_starts=state.valid_starts.copy(),
                anchors={int(state.valid_starts[position]): gradient.copy()
                         for position, gradient in sorted(state.anchors.items())},
                anchor_scores={int(state.valid_starts[position]): float(score)
                               for position, score in state.anchor_scores.items()},
                matched_starts=[int(state.valid_starts[position])
                                for position in sorted(state.matched)],
                candidate_regions=[
                    (int(state.valid_starts[left]), int(state.valid_starts[right]))
                    for left, right in state.regions
                ],
                weights=state.weights.copy(),
                probabilities=state.probabilities.copy(),
                coarse_gradients=state.coarse_gradients,
                fine_gradients=state.fine_gradients,
            )
            for state in source
        ]
        nominal_counts = {
            "coarse_non_endpoint": dict(
                coarse_nominal_internal,
                total=coarse_nominal_internal["reference"]
                     + coarse_nominal_internal["source"]),
            "coarse_target_including_endpoints": dict(
                coarse_target_total,
                total=coarse_target_total["reference"] + coarse_target_total["source"]),
            "fine_targets_by_round": [
                dict(values, total=values["reference"] + values["source"])
                for values in fine_nominal_by_round
            ],
        }
        return ADSRunReport(
            reference_results=reference_results,
            source_results=source_results,
            matches_by_round=matches_by_round,
            score_matrix_summaries=score_matrix_summaries,
            actual_gradient_counts=actual_counts,
            nominal_gradient_counts=nominal_counts,
            cached_gradient_counts=dict(self.cached_gradient_counts),
            zero_norm_anchors={"reference": zero_reference, "source": zero_source},
            zero_norm_pairs=zero_pairs,
            probability_fallback=probability_fallback,
            target_fine_density=self.config.target_fine_density,
            achieved_fine_density=achieved_density,
            selected_top_k=selected_top_k,
            decay_distance=self.config.decay_distance,
        )


def source_probability_descriptor(report: ADSRunReport) -> dict:
    """Return a compact, JSON-safe source distribution based on real anchors."""
    trajectories = []
    total_weight = float(sum(result.weights.sum() for result in report.source_results))
    for result in report.source_results:
        starts = np.asarray(result.valid_starts, dtype=np.int64)
        if len(starts) == 0:
            raise ValueError("source probability descriptor requires legal starts")
        stride = int(starts[1] - starts[0]) if len(starts) > 1 else 1
        if stride <= 0 or not np.array_equal(
                starts, starts[0] + np.arange(len(starts), dtype=np.int64) * stride):
            raise ValueError("source candidate starts must form a regular stride")
        trajectories.append({
            "fragment_id": int(result.fragment_id),
            "first_start": int(starts[0]),
            "stride": stride,
            "candidate_count": int(len(starts)),
            "anchor_positions": [
                int((position - int(starts[0])) // stride)
                for position in result.anchor_scores
            ],
            "anchor_scores": [float(score) for score in result.anchor_scores.values()],
        })
    return {
        "schema": 1,
        "mode": "uniform" if report.probability_fallback else "anchor_neighbor_max_linear",
        "decay_distance": int(report.decay_distance),
        "normalization_mass": total_weight,
        "population": int(sum(item["candidate_count"] for item in trajectories)),
        "probability_fallback": bool(report.probability_fallback),
        "trajectories": trajectories,
    }


class SourceProbabilitySampler:
    """Precompute a compact descriptor's positive-support sampling table."""

    def __init__(self, descriptor: dict):
        trajectories = descriptor.get("trajectories", [])
        if not trajectories:
            raise ValueError("probability descriptor has no trajectories")
        self.counts = np.asarray(
            [int(item["candidate_count"]) for item in trajectories], dtype=np.int64)
        if np.any(self.counts <= 0):
            raise ValueError("probability descriptor has an empty trajectory")
        self.offsets = np.cumsum(np.r_[0, self.counts])
        self.fragment_ids = np.asarray(
            [int(item["fragment_id"]) for item in trajectories], dtype=np.int64)
        self.first_starts = np.asarray(
            [int(item["first_start"]) for item in trajectories], dtype=np.int64)
        self.strides = np.asarray(
            [int(item["stride"]) for item in trajectories], dtype=np.int64)
        self.mode = descriptor.get("mode")
        self.support_rows = None
        self.cumulative_mass = None
        if self.mode == "uniform":
            return
        if self.mode != "anchor_neighbor_max_linear":
            raise ValueError("unsupported source probability descriptor mode")
        decay = int(descriptor["decay_distance"])
        if decay <= 0:
            raise ValueError("decay_distance must be positive")
        support_rows = []
        support_weights = []
        for owner, item in enumerate(trajectories):
            positions = item["anchor_positions"]
            scores = item["anchor_scores"]
            if len(positions) != len(scores):
                raise ValueError("anchor positions and scores must have matching lengths")
            weights = np.zeros(self.counts[owner], dtype=np.float64)
            for position, score in zip(positions, scores):
                position = int(position)
                contribution = max(0.0, float(score))
                if position < 0 or position >= len(weights):
                    raise ValueError("anchor position outside candidate range")
                lo = max(0, position - decay + 1)
                hi = min(len(weights), position + decay)
                neighbors = np.arange(lo, hi, dtype=np.int64)
                values = contribution * (1.0 - np.abs(neighbors - position) / decay)
                weights[neighbors] = np.maximum(weights[neighbors], values)
            positive = np.flatnonzero(weights > 0)
            support_rows.append(self.offsets[owner] + positive)
            support_weights.append(weights[positive])
        rows = np.concatenate(support_rows) if support_rows else np.zeros(0, dtype=np.int64)
        values = (np.concatenate(support_weights) if support_weights
                  else np.zeros(0, dtype=np.float64))
        mass = float(values.sum())
        if mass <= 0:
            raise ValueError("zero-mass descriptor must use uniform mode")
        self.support_rows = rows
        self.cumulative_mass = np.cumsum(values / mass)
        self.cumulative_mass[-1] = 1.0

    def sample(self, rng: np.random.Generator,
               n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample legal (fragment_id, physical_start) pairs with replacement."""
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if self.mode == "uniform":
            rows = rng.integers(0, int(self.offsets[-1]), size=n_samples)
        else:
            selected = np.searchsorted(
                self.cumulative_mass, rng.random(n_samples), side="right")
            rows = self.support_rows[selected]
        owners = np.searchsorted(self.offsets[1:], rows, side="right")
        local_positions = rows - self.offsets[owners]
        return (self.fragment_ids[owners],
                self.first_starts[owners] + local_positions * self.strides[owners])


def sample_source_probability_descriptor(descriptor: dict, rng: np.random.Generator,
                                         n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample legal (fragment_id, physical_start) pairs from a descriptor."""
    return SourceProbabilitySampler(descriptor).sample(rng, n_samples)


__all__ = [
    "ADSConfig",
    "ADSMatch",
    "ADSMatchSummary",
    "ADSTrajectoryResult",
    "ADSRunReport",
    "ADSPipeline",
    "source_probability_descriptor",
    "sample_source_probability_descriptor",
    "SourceProbabilitySampler",
    "uniform_anchor_positions",
    "cosine_similarity_matrix",
    "global_top_k",
    "global_top_k_hit_masks",
    "summarize_similarity_matrix",
    "aggregate_source_scores",
    "spread_anchor_scores",
    "normalize_sampling_probabilities",
]
