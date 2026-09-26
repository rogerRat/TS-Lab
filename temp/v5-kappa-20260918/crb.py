"""Fixed-budget CRB gradient reconstruction and contiguous block retrieval.

This module implements the five-stage protocol in ``_method-0903.md``.  It is
deliberately independent of the older adaptive-quadrature/perturbation code so
experiments produced by that implementation remain reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import heapq
from typing import Callable, Iterable

import numpy as np

from .core import TimeSeriesFragment


BatchGradientComputer = Callable[[TimeSeriesFragment, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class CRBConfig:
    """Configuration for the fixed ``(C, H, D)`` protocol and three budgets."""

    context_length: int = 512
    prediction_length: int = 64
    window_stride: int = 1
    cover_density: float = 0.01
    error_density: float = 0.005
    split_density: float = 0.005
    curvature_cap: float = 1.0
    mean_epsilon: float = 1e-8
    endpoint_epsilon: float = 1e-8
    score_epsilon: float = 1e-8  # Legacy serialized option; not a floor for cosine normalization.
    initial_placement: str = "longest"  # "longest" | "random" (ablation)
    error_placement: str = "risk"       # "risk" | "uniform" (ablation)
    detection: str = "standard"  # "standard" | "none" | "drop_only" (ablation)
    placement_seed: int = 0
    priority_tie_policy: str = "left"  # V5 approved: "right"; legacy: "left"; "random" is historical.
    split_policy: str = "closest_zero_v2"
    epsilon_policy: str = "fixed"  # Legacy thresholds; min_kappa_v5 explicitly uses none.

    def __post_init__(self) -> None:
        if self.split_policy not in ("closest_zero_v2", "closest_zero_v3", "closest_zero_v4", "closest_zero_v5", "min_kappa_v5"):
            raise ValueError("split_policy must be closest_zero_v2, closest_zero_v3, closest_zero_v4, closest_zero_v5, or min_kappa_v5")
        if self.priority_tie_policy not in ("left", "random", "right"):
            raise ValueError("priority_tie_policy must be left, random, or right")
        if self.epsilon_policy not in ("fixed", "trajectory_p10", "none"):
            raise ValueError("epsilon_policy must be fixed, trajectory_p10, or none")
        if self.split_policy == "min_kappa_v5":
            if self.epsilon_policy != "none" or self.priority_tie_policy != "right":
                raise ValueError("min_kappa_v5 requires epsilon_policy=none and priority_tie_policy=right")
            if self.initial_placement != "longest":
                raise ValueError("min_kappa_v5 requires deterministic longest coverage")
        elif self.epsilon_policy == "none":
            raise ValueError("epsilon_policy=none requires min_kappa_v5")
        if self.context_length <= 0 or self.prediction_length <= 0 or self.window_stride <= 0:
            raise ValueError("C, H, and D must be positive")
        for name in ("cover_density", "error_density", "split_density"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError("%s must be finite and non-negative" % name)
        if self.curvature_cap <= 0:
            raise ValueError("curvature_cap must be positive")
        if self.mean_epsilon < 0 or self.endpoint_epsilon < 0 or self.score_epsilon <= 0:
            raise ValueError("epsilon values must be non-negative and score_epsilon positive")
        if self.initial_placement not in ("longest", "random"):
            raise ValueError("initial_placement must be 'longest' or 'random'")
        if self.error_placement not in ("risk", "uniform"):
            raise ValueError("error_placement must be 'risk' or 'uniform'")
        if self.detection not in ("standard", "none", "drop_only"):
            raise ValueError("detection must be 'standard', 'none', or 'drop_only'")

    def budgets(self, positions: int) -> tuple[int, int, int]:
        """Rounded per-trajectory budgets, with endpoints charged to coverage."""
        if positions <= 0:
            return 0, 0, 0
        cover = min(positions, max(min(2, positions), int(round(self.cover_density * positions))))
        error = min(positions - cover, int(round(self.error_density * positions)))
        split = min(positions - cover - error, int(round(self.split_density * positions)))
        return cover, error, split


@dataclass
class _Interval:
    left: int
    right: int
    curvature: float | None = None

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def midpoint(self) -> int:
        return (self.left + self.right) // 2

    @property
    def splittable(self) -> bool:
        return self.left < self.midpoint < self.right


@dataclass(frozen=True)
class GradientBlock:
    """One retained minimum anchor interval, represented by its mean direction."""

    fragment_id: int
    left_index: int
    right_index: int
    start_positions: np.ndarray
    mean_gradient: np.ndarray
    direction: np.ndarray

    @property
    def size(self) -> int:
        return int(len(self.start_positions))


@dataclass
class CRBResult:
    fragment_id: int
    valid_starts: np.ndarray
    anchors: dict[int, np.ndarray]
    blocks: list[GradientBlock]
    discarded_no_signal: int
    discarded_antagonistic: int
    banned_budget: int
    cover_computations: int
    error_computations: int
    split_computations: int
    epsilon_audit: dict | None = None
    split_audit: dict | None = None

    @property
    def computed_gradients(self) -> int:
        return self.cover_computations + self.error_computations + self.split_computations


@dataclass(frozen=True)
class BlockMatch:
    reference_block: int
    candidate_block: int
    score: float


@dataclass
class CRBRunReport:
    reference_results: list[CRBResult]
    candidate_results: list[CRBResult]
    reference_blocks: list[GradientBlock]
    candidate_blocks: list[GradientBlock]
    matches_by_reference: list[list[BlockMatch]]

    @property
    def reference_backward_count(self) -> int:
        return sum(result.computed_gradients for result in self.reference_results)

    @property
    def candidate_backward_count(self) -> int:
        return sum(result.computed_gradients for result in self.candidate_results)


def _crb_pipeline(fragment: TimeSeriesFragment, config: CRBConfig):
    """The five-stage CRB protocol as a coroutine (single source of truth).

    Yields arrays of 0-based row indices into ``valid_starts`` and receives
    ``[k, d]`` gradient matrices aligned with the yielded rows.  This is a
    faithful transcription of the sequential stages; both ``CRBEngine`` and
    ``CRBEngineMulti`` drive it, so batching probes across fragments cannot
    change per-fragment semantics (same anchors, blocks, counters).
    """
    starts = fragment.valid_starts(
        config.context_length, config.prediction_length, config.window_stride)
    total = len(starts)
    cover_budget, error_budget, split_budget = config.budgets(total)
    kappa_mode = config.split_policy == "min_kappa_v5"

    def kappa_audit(used):
        if not kappa_mode:
            return None
        return dict(policy="min_kappa_v5", fragment_id=int(fragment.fragment_id),
                    priority="norm_endpoint_sum_over_sum_endpoint_norms",
                    zero_zero_kappa=1.0, priority_tie_policy="right",
                    recursive=True, tie_rng_enabled=False,
                    split_budget=split_budget, split_used=used,
                    split_unused=split_budget-used, split_stop_reason="budget_exhausted")
    if total == 0:
        return CRBResult(fragment.fragment_id, starts, {}, [], 0, 0, 0, 0, 0, 0,
                         split_audit=kappa_audit(0))

    anchors: dict[int, np.ndarray] = {}
    tie_rngs = {
        stage: np.random.default_rng(np.random.SeedSequence([
            int(config.placement_seed) % (2**32),
            int(fragment.fragment_id) % (2**32), stream]))
        for stream, stage in enumerate(("cover", "error", "split"))
    } if config.priority_tie_policy == "random" else {}

    def pick_ties(items, stage, limit=1):
        # Canonical ordering makes the draw independent of heap/insertion order.
        ordered = sorted(items, key=lambda item: item.left)
        limit = min(limit, len(ordered))
        if limit == len(ordered):
            return ordered
        if config.priority_tie_policy == "random":
            indices = tie_rngs[stage].choice(len(ordered), size=limit, replace=False)
            return [ordered[int(index)] for index in indices]
        if config.priority_tie_policy == "right":
            return list(reversed(ordered))[:limit]
        return ordered[:limit]

    def pop_priority_group(heap, all_items=False):
        if not heap:
            return []
        priority = heap[0][0]
        entries = []
        while heap and (all_items or heap[0][0] == priority):
            entries.append(heapq.heappop(heap))
        return entries

    def compute(indices: Iterable[int]):
        missing = sorted({int(index) for index in indices if int(index) not in anchors})
        if not missing:
            return 0
        values = yield np.asarray(missing, dtype=np.int64)
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != len(missing) or not np.isfinite(values).all():
            raise ValueError("gradient matrix does not match the request")
        for index, gradient in zip(missing, values):
            anchors[index] = gradient
        return len(missing)

    if total == 1:
        used = yield from compute([0])
        block = CRBEngine(config, None)._make_block(
            fragment.fragment_id, _Interval(0, 0), starts, anchors, True)
        return CRBResult(fragment.fragment_id, starts, anchors, [block], 0, 0, 0, used, 0, 0,
                         split_audit=kappa_audit(0))

    # Coverage: endpoints, then either repeated longest-interval bisection
    # or (ablation) uniformly random interior anchors.
    cover_used = yield from compute([0, total - 1])
    intervals = [_Interval(0, total - 1)]
    if config.initial_placement == "random" and cover_budget > 2 and total > 2:
        rng = np.random.default_rng(config.placement_seed)
        wanted = min(cover_budget - 2, total - 2)
        picks = np.sort(rng.choice(np.arange(1, total - 1), size=wanted,
                                   replace=False))
        cover_used += yield from compute(int(p) for p in picks)
        ordered = [0] + [int(p) for p in picks] + [total - 1]
        intervals = [_Interval(ordered[i], ordered[i + 1])
                     for i in range(len(ordered) - 1)]
    else:
        while cover_used < cover_budget:
            splittable = [interval for interval in intervals if interval.splittable]
            if not splittable:
                break
            max_width = max(interval.width for interval in splittable)
            frontier = pick_ties(
                [interval for interval in splittable if interval.width == max_width],
                "cover", cover_budget - cover_used)
            cover_used += yield from compute(interval.midpoint for interval in frontier)
            for interval in frontier:
                midpoint = interval.midpoint
                curvature = CRBEngine._observed_curvature(
                    interval.left, midpoint, interval.right, anchors)
                intervals.remove(interval)
                intervals.extend((
                    _Interval(interval.left, midpoint, curvature),
                    _Interval(midpoint, interval.right, curvature),
                ))

    # Error allocation: highest integrated quadratic interpolation risk,
    # or (ablation) continued longest-interval bisection.
    error_used = 0
    uniform_error = config.error_placement == "uniform"
    # Once an interval has measured curvature, its priority is immutable until
    # it is split. Unknown-curvature intervals all share the same optimistic
    # bound, so their widest interval is always the best unknown candidate.
    # Two heaps preserve the exact max(risk, -left) decision without rescanning
    # every surviving interval after every gradient request.
    optimistic = max((item.curvature for item in intervals if item.curvature is not None),
                     default=config.curvature_cap)
    measured_heap, unknown_heap = [], []
    active = {item.left: item for item in intervals}

    def enqueue(item):
        if not item.splittable:
            return
        if uniform_error:
            heapq.heappush(measured_heap, (-item.width, item.left, item))
        elif item.curvature is None:
            heapq.heappush(unknown_heap, (-item.width, item.left, item))
        else:
            risk = item.width ** 5 * item.curvature ** 2 / 120.0
            heapq.heappush(measured_heap, (-risk, item.left, item))

    for item in intervals:
        enqueue(item)
    have_observed = any(item.curvature is not None for item in intervals)
    while error_used < error_budget:
        if not measured_heap and not unknown_heap:
            break
        if config.priority_tie_policy != "left":
            measured_priority = measured_heap[0][0] if measured_heap else float("inf")
            unknown_priority = (-unknown_heap[0][2].width ** 5 * optimistic ** 2 / 120.0
                                if unknown_heap else float("inf"))
            priority = min(measured_priority, unknown_priority)
            contenders = []
            if measured_priority == priority:
                contenders.extend((measured_heap, entry) for entry
                                  in pop_priority_group(measured_heap))
            if unknown_priority == priority:
                # At optimistic=0, ALL widths have identical zero risk.
                contenders.extend((unknown_heap, entry) for entry
                                  in pop_priority_group(unknown_heap, optimistic == 0))
            interval = pick_ties([entry[2] for _, entry in contenders], "error")[0]
            for heap, entry in contenders:
                if entry[2] is not interval:
                    heapq.heappush(heap, entry)
        else:
            use_unknown = bool(unknown_heap)
            unknown_entry = (min(unknown_heap, key=lambda item: item[1])
                             if unknown_heap and optimistic == 0 else
                             unknown_heap[0] if unknown_heap else None)
            if unknown_heap and measured_heap:
                unknown = unknown_entry[2]
                unknown_key = (-unknown.width ** 5 * optimistic ** 2 / 120.0, unknown.left)
                use_unknown = unknown_key < measured_heap[0][:2]
            if use_unknown and unknown_entry is not unknown_heap[0]:
                unknown_heap.remove(unknown_entry)
                heapq.heapify(unknown_heap)
                interval = unknown_entry[2]
            else:
                interval = heapq.heappop(unknown_heap if use_unknown else measured_heap)[2]
        midpoint = interval.midpoint
        error_used += yield from compute([midpoint])
        curvature = CRBEngine._observed_curvature(interval.left, midpoint, interval.right, anchors)
        optimistic = max(optimistic, curvature) if have_observed else curvature
        have_observed = True
        del active[interval.left]
        children = (
            _Interval(interval.left, midpoint, curvature),
            _Interval(midpoint, interval.right, curvature),
        )
        for child in children:
            active[child.left] = child
            enqueue(child)
    intervals = list(active.values())

    # Freeze from this pipeline's distinct phase 1/2 anchors only. Extra rows
    # already in the shared cache (LESS, old splits, other tasks) do not enter.
    epsilon_audit = None
    if config.epsilon_policy == "trajectory_p10" or config.split_policy in ("closest_zero_v4", "closest_zero_v5"):
        norms = np.asarray([np.linalg.norm(g.astype(np.float64)) for g in anchors.values()])
        if not np.isfinite(norms).all():
            raise ValueError("Non-finite phase 1/2 gradient norms")
        epsilon = float(np.quantile(norms, 0.10, method="linear"))
        epsilon_audit = dict(policy="trajectory_p10", percentile=10, method="linear",
            count=len(anchors), epsilon=epsilon, frozen_before="split",
            gate=("split_endpoint" if config.split_policy in ("closest_zero_v4", "closest_zero_v5")
                  else "audit_only"),
            fragment_id=int(fragment.fragment_id))
        config = replace(config, mean_epsilon=epsilon, endpoint_epsilon=epsilon)

    # V2-V5 retain every interval. Split the current minimum-norm frontier in
    # batches. The endpoint average is the linearly reconstructed midpoint
    # gradient; ranking does not require evaluating every prospective midpoint.
    # closest_zero_v2 re-enqueues children and recurses into the minimum until
    # the budget is exhausted. closest_zero_v3 splits each interval at most
    # once, so the budget is only an upper bound. closest_zero_v4 additionally
    # gates the pool on the frozen trajectory P10: an interval competes only
    # when at least one endpoint norm is not weak, so purely flat regions are
    # never split. V5 recursively rechecks children with a strict > P10 gate,
    # and re-evaluates the current minimum after each individual split.
    one_shot = config.split_policy in ("closest_zero_v3", "closest_zero_v4")
    strict_recursive = config.split_policy == "closest_zero_v5"
    gated_pool = config.split_policy in ("closest_zero_v4", "closest_zero_v5")
    split_used = 0
    frontier_heap = []
    active = {item.left: item for item in intervals}

    def enqueue_split(item):
        if not item.splittable:
            return
        if strict_recursive:
            if not any(float(np.linalg.norm(anchors[index].astype(np.float64)))
                       > config.endpoint_epsilon for index in (item.left, item.right)):
                return
        elif gated_pool:
            left_ok = not CRBEngine._weak(
                np.linalg.norm(anchors[item.left].astype(np.float64)),
                config.endpoint_epsilon)
            right_ok = not CRBEngine._weak(
                np.linalg.norm(anchors[item.right].astype(np.float64)),
                config.endpoint_epsilon)
            if not (left_ok or right_ok):
                return
        if kappa_mode:
            # No endpoint gate or quantile, including two zero endpoints.
            heapq.heappush(frontier_heap, (
                CRBEngine._interval_kappa(item, anchors), -item.left, item))
        else:
            heapq.heappush(frontier_heap, (
                CRBEngine._interval_mean_norm(item, anchors), item.left, item))

    for item in intervals:
        enqueue_split(item)
    while split_used < split_budget and frontier_heap:
        minimum = frontier_heap[0][0]
        frontier = []
        limit = 1 if (strict_recursive or kappa_mode) else split_budget - split_used
        if kappa_mode:
            frontier = [heapq.heappop(frontier_heap)[2]]
        elif config.priority_tie_policy != "left":
            tied = pop_priority_group(frontier_heap)
            frontier = pick_ties([entry[2] for entry in tied], "split", limit)
            selected = {item.left for item in frontier}
            for entry in tied:
                if entry[2].left not in selected:
                    heapq.heappush(frontier_heap, entry)
        else:
            while (frontier_heap and frontier_heap[0][0] == minimum
                   and len(frontier) < limit):
                frontier.append(heapq.heappop(frontier_heap)[2])
        split_used += yield from compute(item.midpoint for item in frontier)
        for item in frontier:
            del active[item.left]
            for child in (_Interval(item.left, item.midpoint),
                          _Interval(item.midpoint, item.right)):
                active[child.left] = child
                if not one_shot:
                    enqueue_split(child)
    if kappa_mode and split_used != split_budget:
        raise RuntimeError("min_kappa_v5 did not consume its capped split budget")
    if strict_recursive:
        epsilon_audit.update(endpoint_comparator="strict_gt", recursive=True,
            split_selection="current_minimum_after_each_split",
            split_budget=split_budget, split_used=split_used,
            split_unused=split_budget-split_used,
            priority_tie_policy=config.priority_tie_policy,
            placement_seed=int(config.placement_seed),
            tie_rng_enabled=(config.priority_tie_policy == "random"),
            tie_seed_components=([int(config.placement_seed) % (2**32),
                                  int(fragment.fragment_id) % (2**32)]
                                 if config.priority_tie_policy == "random" else None),
            tie_rng_streams=({"cover": 0, "error": 1, "split": 2}
                            if config.priority_tie_policy == "random" else {}),
            split_stop_reason=("budget_exhausted" if split_used == split_budget
                               else "eligible_pool_exhausted"))
    passed = list(active.values())
    no_signal = antagonistic = banned = 0

    # Half-open blocks, except the final interval owns its right endpoint.
    passed.sort(key=lambda item: item.left)
    final_right = total - 1
    blocks = []
    engine_view = CRBEngine(config, None)
    for interval in passed:
        blocks.append(CRBEngine._make_block(
            engine_view, fragment.fragment_id, interval, starts, anchors,
            interval.right == final_right))
    return CRBResult(
        fragment.fragment_id, starts, anchors, blocks, no_signal, antagonistic,
        banned, cover_used, error_used, split_used, epsilon_audit, kappa_audit(split_used),
    )


class CRBEngine:
    """Run fixed-budget coverage, risk allocation, and minimum-norm splitting.

    Splitting recurses into the minimum frontier under ``closest_zero_v2`` and
    performs a single pass under ``closest_zero_v3``; ``closest_zero_v4``
    additionally gates the pool by frozen trajectory-P10 endpoint strength.
    ``closest_zero_v5`` recurses with a strict > P10 gate and current-minimum selection.
    ``min_kappa_v5`` has no threshold, minimizes relative cancellation, breaks
    exact ties to the right, and consumes the entire capped split budget.

    ``gradient_computer`` accepts a fragment and an array of *physical* window
    starts and returns one gradient sketch per row.  Coverage midpoints are
    computed in batches; the error stage remains sequential per trajectory
    because every observation changes that trajectory's highest-risk
    interval.
    """

    def __init__(self, config: CRBConfig, gradient_computer: BatchGradientComputer):
        self.config = config
        self.gradient_computer = gradient_computer

    def process_fragment(self, fragment: TimeSeriesFragment) -> CRBResult:
        starts = fragment.valid_starts(
            self.config.context_length,
            self.config.prediction_length,
            self.config.window_stride,
        )
        pipeline = _crb_pipeline(fragment, self.config)
        try:
            request = pipeline.send(None)
        except StopIteration as stop:
            return stop.value
        while True:
            physical = starts[np.asarray(request, dtype=np.int64)]
            values = np.asarray(self.gradient_computer(fragment, physical),
                                dtype=np.float32)
            if values.ndim != 2 or values.shape[0] != len(request) \
                    or not np.isfinite(values).all():
                raise ValueError("gradient_computer must return a finite [positions, dimension] matrix")
            try:
                request = pipeline.send(values)
            except StopIteration as stop:
                return stop.value

    # -- static stage helpers shared with the pipeline ----------------------
    @staticmethod
    def _observed_curvature(left: int, midpoint: int, right: int,
                            anchors: dict[int, np.ndarray]) -> float:
        alpha = (right - midpoint) / (right - left)
        beta = (midpoint - left) / (right - left)
        interpolated = alpha * anchors[left] + beta * anchors[right]
        delta = float(np.linalg.norm((anchors[midpoint] - interpolated).astype(np.float64)))
        return 2.0 * delta / ((midpoint - left) * (right - midpoint))

    @staticmethod
    def _interval_kappa(interval: _Interval, anchors: dict[int, np.ndarray]) -> float:
        left = anchors[interval.left].astype(np.float64)
        right = anchors[interval.right].astype(np.float64)
        denominator = float(np.linalg.norm(left) + np.linalg.norm(right))
        numerator = float(np.linalg.norm(left + right))
        if not np.isfinite(denominator) or not np.isfinite(numerator):
            raise ValueError("Non-finite kappa inputs")
        if denominator == 0:
            return 1.0  # No evidence of cancellation; still competes for budget.
        return min(1.0, max(0.0, numerator / denominator))

    @staticmethod
    def _interval_mean(interval: _Interval, anchors: dict[int, np.ndarray]) -> np.ndarray:
        mean = (anchors[interval.left].astype(np.float64) + anchors[interval.right].astype(np.float64)) * 0.5
        if not np.isfinite(mean).all():
            raise ValueError("Non-finite interval representative")
        return mean

    @staticmethod
    def _weak(norm: float, epsilon: float) -> bool:
        if not np.isfinite(norm):
            raise ValueError("Non-finite gradient/representative norm")
        return norm == 0 or norm < epsilon

    @staticmethod
    def _interval_mean_norm(interval: _Interval, anchors: dict[int, np.ndarray]) -> float:
        return float(np.linalg.norm(CRBEngine._interval_mean(interval, anchors).astype(np.float64)))

    @classmethod
    def _probe_state(cls, interval: _Interval, anchors: dict[int, np.ndarray],
                     config: CRBConfig) -> str:
        if not cls._weak(cls._interval_mean_norm(interval, anchors), config.mean_epsilon):
            return "pass"
        left_weak = cls._weak(np.linalg.norm(anchors[interval.left].astype(np.float64)), config.endpoint_epsilon)
        right_weak = cls._weak(np.linalg.norm(anchors[interval.right].astype(np.float64)), config.endpoint_epsilon)
        return "no_signal" if left_weak and right_weak else "antagonistic"

    def _mean_norm(self, interval: _Interval, anchors: dict[int, np.ndarray]) -> float:
        return self._interval_mean_norm(interval, anchors)

    def _probe(self, interval: _Interval, anchors: dict[int, np.ndarray]) -> str:
        return self._probe_state(interval, anchors, self.config)

    def _make_block(self, fragment_id: int, interval: _Interval,
                    starts: np.ndarray, anchors: dict[int, np.ndarray],
                    include_right: bool) -> GradientBlock:
        stop = interval.right + 1 if include_right else interval.right
        positions = starts[interval.left:stop].copy()
        mean = self._interval_mean(interval, anchors)
        norm = float(np.linalg.norm(mean))
        if not np.isfinite(norm):
            raise ValueError("Non-finite retained representative norm")
        direction = (mean / norm).astype(np.float32) if norm else np.zeros_like(mean, dtype=np.float32)
        return GradientBlock(fragment_id, interval.left, interval.right, positions, mean.astype(np.float32), direction)


class _StackedFragmentView:
    """Concatenated multi-fragment view for cross-fragment batched probes.

    The gradient computer sees one virtual trajectory: concatenated physical
    starts map back to the owning real fragment on ``extract_window``.  Only
    compatible with stateless computers (the CRB/engine computers qualify);
    caching computers keyed by fragment identity must use CRBEngine.
    """

    def __init__(self, fragments, starts_list):
        self.fragments = fragments
        self.starts_list = starts_list
        offsets = np.cumsum([0] + [f.length for f in fragments])
        self._offsets = offsets
        self._length = int(offsets[-1])

    @property
    def fragment_id(self) -> int:
        return -1

    @property
    def length(self) -> int:
        return self._length

    def extract_window(self, start: int, context_length: int,
                       prediction_length: int) -> np.ndarray:
        owner = int(np.searchsorted(self._offsets, start, side="right") - 1)
        local = int(start) - int(self._offsets[owner])
        return self.fragments[owner].extract_window(local, context_length,
                                                    prediction_length)


class CRBEngineMulti(CRBEngine):
    """Drive many fragments' pipelines with cross-fragment batched probes.

    Semantically identical to ``process_fragment`` per fragment (the same
    ``_crb_pipeline`` runs); probes from all active fragments are merged
    into one ``gradient_computer`` call per round so the sequential
    error/split stages keep the GPU busy on large pools.
    """

    def process_fragments(self, fragments: list[TimeSeriesFragment]) -> list[CRBResult]:
        results: list[CRBResult | None] = [None] * len(fragments)
        starts_by_fragment = [f.valid_starts(self.config.context_length,
                                            self.config.prediction_length,
                                            self.config.window_stride) for f in fragments]
        stacked = _StackedFragmentView(fragments, starts_by_fragment)
        pipelines: list = []
        for index, fragment in enumerate(fragments):
            pipeline = _crb_pipeline(fragment, self.config)
            try:
                request = pipeline.send(None)
            except StopIteration as stop:
                results[index] = stop.value
                continue
            pipelines.append((index, fragment, pipeline, request))
        while pipelines:
            physical = np.concatenate([
                starts_by_fragment[entry[0]][np.asarray(entry[3], dtype=np.int64)]
                + int(stacked._offsets[entry[0]])
                for entry in pipelines])
            compute_many = getattr(self.gradient_computer, "compute_many", None)
            if compute_many is not None:
                values = compute_many([(entry[1], starts_by_fragment[entry[0]][
                    np.asarray(entry[3], dtype=np.int64)]) for entry in pipelines])
            else:
                values = self.gradient_computer(stacked, physical)
            values = np.asarray(values, dtype=np.float32)
            if values.ndim != 2 or values.shape[0] != len(physical) \
                    or not np.isfinite(values).all():
                raise ValueError("gradient_computer must return a finite [positions, dimension] matrix")
            survivors: list = []
            cursor = 0
            for position, (index, fragment, pipeline, _request) in enumerate(pipelines):
                count = len(pipelines[position][3])
                piece = values[cursor:cursor + count]
                cursor += count
                try:
                    next_request = pipeline.send(piece)
                    survivors.append((index, fragment, pipeline, next_request))
                except StopIteration as stop:
                    results[index] = stop.value
            pipelines = survivors
        return [result for result in results]


def match_blocks(reference_blocks: list[GradientBlock], candidate_blocks: list[GradientBlock],
                 budget: int) -> list[list[BlockMatch]]:
    """Return at most ``budget`` source blocks for every reference block."""
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if not reference_blocks:
        return []
    if not candidate_blocks or budget == 0:
        return [[] for _ in reference_blocks]
    reference = np.stack([block.direction for block in reference_blocks]).astype(np.float32)
    candidate = np.stack([block.direction for block in candidate_blocks]).astype(np.float32)
    scores = reference @ candidate.T
    take = min(budget, len(candidate_blocks))
    result: list[list[BlockMatch]] = []
    for ref_index, row in enumerate(scores):
        indices = np.argsort(-row, kind="stable")[:take]
        result.append([BlockMatch(ref_index, int(index), float(row[index])) for index in indices])
    return result


class CRBPipeline:
    """Apply identical CRB processing to reference and source trajectories."""

    def __init__(self, config: CRBConfig, gradient_computer: BatchGradientComputer):
        self.config = config
        self.engine = CRBEngine(config, gradient_computer)

    def run(self, reference_fragments: list[TimeSeriesFragment],
            candidate_fragments: list[TimeSeriesFragment], *, block_budget: int) -> CRBRunReport:
        reference_results = [self.engine.process_fragment(fragment) for fragment in reference_fragments]
        candidate_results = [self.engine.process_fragment(fragment) for fragment in candidate_fragments]
        reference_blocks = [block for result in reference_results for block in result.blocks]
        candidate_blocks = [block for result in candidate_results for block in result.blocks]
        matches = match_blocks(reference_blocks, candidate_blocks, block_budget)
        return CRBRunReport(reference_results, candidate_results, reference_blocks,
                            candidate_blocks, matches)


def top_unique_candidate_blocks(report: CRBRunReport, budget: int) -> list[GradientBlock]:
    """Aggregate per-query matches by best cosine for fixed-size training arms."""
    best: dict[int, float] = {}
    for matches in report.matches_by_reference:
        for match in matches:
            best[match.candidate_block] = max(best.get(match.candidate_block, -np.inf), match.score)
    indices = sorted(best, key=lambda index: (-best[index], index))[:budget]
    return [report.candidate_blocks[index] for index in indices]


__all__ = [
    "CRBConfig", "GradientBlock", "CRBResult", "BlockMatch", "CRBRunReport",
    "CRBEngine", "CRBEngineMulti", "CRBPipeline", "match_blocks",
    "top_unique_candidate_blocks",
]
