"""Adaptive Density Sampling over real source and reference gradients."""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from bisect import bisect_left, bisect_right
import math
import time
from typing import Callable, MutableMapping, Sequence

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

    q_coarse: float = 32.0
    q_fine: float = 8.0
    eta: float = 1.0
    top_k: int | None = 64
    top_ratio: float | None = None
    target_fine_density: float | None = None
    rounds: int = 1
    decay_distance: int = 4
    aggregation: str = "max"

    def __post_init__(self) -> None:
        for name in ("q_coarse", "q_fine", "eta"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError("%s must be finite and positive" % name)
        active = sum(value is not None for value in (
            self.top_k, self.top_ratio, self.target_fine_density))
        if active != 1:
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
        if isinstance(self.rounds, bool) or int(self.rounds) != self.rounds or self.rounds <= 0:
            raise ValueError("rounds must be a positive integer")
        if (isinstance(self.decay_distance, bool)
                or int(self.decay_distance) != self.decay_distance
                or self.decay_distance <= 0):
            raise ValueError("decay_distance must be a positive integer")
        if self.aggregation not in ("linear", "max", "exp", "relu"):
            raise ValueError("aggregation must be linear, max, exp, or relu")


@dataclass(frozen=True)
class ADSMatch:
    reference_fragment_id: int
    reference_start: int
    source_fragment_id: int
    source_start: int
    similarity: float


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
    matches_by_round: list[list[ADSMatch]]
    actual_gradient_counts: dict[str, dict[str, int]]
    nominal_gradient_counts: dict
    cached_gradient_counts: dict[str, int]
    zero_norm_anchors: dict[str, int]
    zero_norm_pairs: int
    probability_fallback: bool
    target_fine_density: float | None = None
    achieved_fine_density: float | None = None
    selected_top_k: list[int] = field(default_factory=list)

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
    candidates = list(range(1, length - 1))
    if wanted == 1:
        targets = [(length - 1) / 2.0]
    else:
        targets = [
            1.0 + index * (interior_count - 1) / (wanted - 1)
            for index in range(wanted)
        ]
    chosen: list[int] = []
    available = set(candidates)
    for target in targets:
        position = min(available, key=lambda candidate: (abs(candidate - target), candidate))
        chosen.append(position)
        available.remove(position)
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
    normalized_reference = np.zeros_like(reference)
    normalized_source = np.zeros_like(source)
    normalized_reference[reference_nonzero] = (
        reference[reference_nonzero] / reference_norms[reference_nonzero, None])
    normalized_source[source_nonzero] = (
        source[source_nonzero] / source_norms[source_nonzero, None])
    matrix = normalized_reference @ normalized_source.T
    zero_pairs = int((~reference_nonzero).sum() * len(source)
                     + reference_nonzero.sum() * (~source_nonzero).sum())
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
    """Apply finite one-sided linear decay and max-overlap aggregation."""
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

    Optional cache keys are (side, fragment_id, physical_start). The caller
    must only reuse vectors from the same model and projection coordinates.
    """

    def __init__(self, config: ADSConfig, gradient_computer: BatchGradientComputer, *,
                 context_length: int = 1, prediction_length: int = 1,
                 window_stride: int = 1, gradient_batch_size: int = 128,
                 progress_callback: Callable[[dict], None] | None = None,
                 heartbeat_seconds: float = 30.0,
                 gradient_cache: MutableMapping[tuple[str, int, int], np.ndarray] | None = None):
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
        self.cached_gradient_counts = {"reference": 0, "source": 0}

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
            if self.gradient_cache is not None:
                for position, start in enumerate(starts):
                    key = (side, int(fragment.fragment_id), int(start))
                    if key not in self.gradient_cache:
                        continue
                    gradient = np.asarray(self.gradient_cache[key], dtype=np.float32)
                    if gradient.ndim != 1 or len(gradient) == 0 or not np.isfinite(gradient).all():
                        raise ValueError("cached gradients must be nonempty finite vectors")
                    state.anchors[position] = gradient.copy()
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
        total_requested = sum(
            len(positions_by_state.get(id(state), set()) - set(state.anchors))
            for state in states)
        completed = 0
        started = time.monotonic()
        for state in states:
            requested = sorted(positions_by_state.get(id(state), set()) - set(state.anchors))
            for offset in range(0, len(requested), self.gradient_batch_size):
                chunk = requested[offset:offset + self.gradient_batch_size]
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
                    if self.gradient_cache is not None:
                        cache_key = (
                            side, state.fragment_id,
                            int(state.valid_starts[int(position)]))
                        self.gradient_cache[cache_key] = gradient.copy()
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
                reference, reference_hits, self.config.q_fine / self.config.eta)
            source_plan = _fine_anchor_plan(source, source_hits, self.config.q_fine)
            count = sum(map(len, reference_plan.values())) + sum(map(len, source_plan.values()))
            difference = abs(count - target_count)
            if difference < best_difference:
                best_k, best_count, best_difference = rank, count, difference
            if len(seen_reference) == matrix.shape[0] and len(seen_source) == matrix.shape[1]:
                break
        return best_k, best_count

    @staticmethod
    def _mark_matches(reference: list[_TrajectoryState], source: list[_TrajectoryState],
                      reference_locations, source_locations,
                      pairs: list[tuple[int, int, float]]) -> list[ADSMatch]:
        reference_hits = {id(state): set() for state in reference}
        source_hits = {id(state): set() for state in source}
        matches: list[ADSMatch] = []
        for row, column, score in pairs:
            reference_state_index, reference_position = reference_locations[row]
            source_state_index, source_position = source_locations[column]
            reference_state = reference[reference_state_index]
            source_state = source[source_state_index]
            reference_hits[id(reference_state)].add(reference_position)
            source_hits[id(source_state)].add(source_position)
            matches.append(ADSMatch(
                reference_fragment_id=reference_state.fragment_id,
                reference_start=int(reference_state.valid_starts[reference_position]),
                source_fragment_id=source_state.fragment_id,
                source_start=int(source_state.valid_starts[source_position]),
                similarity=float(score),
            ))
        for state in reference:
            state.matched = reference_hits[id(state)]
        for state in source:
            state.matched = source_hits[id(state)]
        return matches

    def run(self, reference_fragments: list[TimeSeriesFragment],
            source_fragments: list[TimeSeriesFragment]) -> ADSRunReport:
        self.cached_gradient_counts = {"reference": 0, "source": 0}
        reference = self._states(reference_fragments, "reference")
        source = self._states(source_fragments, "source")
        coarse_reference = {
            id(state): set(uniform_anchor_positions(
                len(state.valid_starts), self.config.q_coarse / self.config.eta))
            for state in reference
        }
        coarse_source = {
            id(state): set(uniform_anchor_positions(
                len(state.valid_starts), self.config.q_coarse))
            for state in source
        }
        coarse_nominal_internal = {
            "reference": sum(
                min(max(len(state.valid_starts) - 2, 0),
                    int(math.ceil(max(len(state.valid_starts) - 2, 0)
                                  / (self.config.q_coarse / self.config.eta))))
                for state in reference),
            "source": sum(
                min(max(len(state.valid_starts) - 2, 0),
                    int(math.ceil(max(len(state.valid_starts) - 2, 0)
                                  / self.config.q_coarse)))
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

        matches_by_round: list[list[ADSMatch]] = []
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
            pairs = global_top_k(
                matrix, top_k, reference_keys=reference_keys, source_keys=source_keys)
            selected_top_k.append(top_k)
            matches_by_round.append(self._mark_matches(
                reference_sorted, source_sorted, reference_locations, source_locations, pairs))
            reference_hits = {id(state): state.matched for state in reference_sorted}
            source_hits = {id(state): state.matched for state in source_sorted}
            fine_reference = _fine_anchor_plan(
                reference_sorted, reference_hits, self.config.q_fine / self.config.eta)
            fine_source = _fine_anchor_plan(
                source_sorted, source_hits, self.config.q_fine)
            self._emit("matching", "end", round=len(matches_by_round),
                       top_k=top_k, pairs=len(pairs))
            fine_nominal_by_round.append({
                "reference": sum(
                    int(math.ceil(sum(right - left + 1 for left, right in state.regions)
                                  / (self.config.q_fine / self.config.eta)))
                    for state in reference_sorted if state.regions),
                "source": sum(
                    int(math.ceil(sum(right - left + 1 for left, right in state.regions)
                                  / self.config.q_fine))
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
        final_pairs = global_top_k(
            matrix, final_top_k, reference_keys=reference_keys,
            source_keys=source_keys)
        selected_top_k.append(final_top_k)
        matches_by_round.append(self._mark_matches(
            reference_sorted, source_sorted, reference_locations,
            source_locations, final_pairs))
        self._emit("matching", "end", round=len(matches_by_round),
                   top_k=final_top_k, pairs=len(final_pairs))
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
            actual_gradient_counts=actual_counts,
            nominal_gradient_counts=nominal_counts,
            cached_gradient_counts=dict(self.cached_gradient_counts),
            zero_norm_anchors={"reference": zero_reference, "source": zero_source},
            zero_norm_pairs=zero_pairs,
            probability_fallback=probability_fallback,
            target_fine_density=self.config.target_fine_density,
            achieved_fine_density=achieved_density,
            selected_top_k=selected_top_k,
        )


__all__ = [
    "ADSConfig",
    "ADSMatch",
    "ADSTrajectoryResult",
    "ADSRunReport",
    "ADSPipeline",
    "uniform_anchor_positions",
    "cosine_similarity_matrix",
    "global_top_k",
    "aggregate_source_scores",
    "spread_anchor_scores",
    "normalize_sampling_probabilities",
]
