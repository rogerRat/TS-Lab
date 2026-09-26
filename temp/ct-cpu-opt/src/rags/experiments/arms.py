"""Selection arms for the main experiment.

Every arm receives the same inputs (reference fragments, source pool, pool
window index, gradient probe) and returns the same contract:

* ``selected``   - float32 fixed windows or lazily sampled RegionWindows;
* ``cost``       - selection-cost accounting (gradient counts, wall time,
  proxy training steps, feature computations);
* ``diagnostics``- arm-specific records written to the result JSON.

Arms that need no source data (original) are expressed by an empty selection.
RAGS delivers whole block envelopes; the granularity control and LESS deliver
whole trajectories. Random draws from the disjoint slot grid. TSDS retains repeated trajectories;
LP-Mirror/LTSV retain native overlapping top-k.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from .regions import RegionWindows

from rags.core import TimeSeriesFragment
from rags.crb import CRBConfig, CRBEngine, CRBEngineMulti, CRBResult, GradientBlock


@dataclass
class SelectionCost:
    gradient_computations: int = 0
    feature_computations: int = 0      # non-gradient features (embeddings etc.)
    proxy_training_steps: int = 0      # baseline proxy/adapter training
    index_bytes: int = 0
    wall_seconds: float = 0.0
    notes: dict = field(default_factory=dict)


@dataclass
class ArmOutput:
    name: str
    selected: np.ndarray | RegionWindows
    window_ids: list[tuple[int, int]]
    cost: SelectionCost
    diagnostics: dict = field(default_factory=dict)


class WindowIndex:
    """Numpy random-access view over the pool's legal windows.

    Legal windows are enumerated once as aligned ``fragment_ids``/``starts``
    arrays (1-indexed legal starts, stride ``stride``); ``position`` is the
    row index into these arrays.  Designed for pools with millions of
    windows: no Python-tuple bookkeeping, batched window extraction.
    """

    def __init__(self, pool: list[TimeSeriesFragment], *, context_length: int,
                 prediction_length: int, stride: int = 1):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride
        self.by_fragment = {f.fragment_id: f for f in pool}
        fragment_ids: list[np.ndarray] = []
        starts: list[np.ndarray] = []
        self.fragment_lengths: dict[int, int] = {}
        self._bounds: dict[int, tuple[int, int]] = {}
        cursor = 0
        for fragment in pool:
            starts_i = fragment.valid_starts(context_length, prediction_length, stride)
            if len(starts_i) == 0:
                continue
            fragment_ids.append(np.full(len(starts_i), fragment.fragment_id, dtype=np.int64))
            starts.append(starts_i.astype(np.int64))
            self._bounds[fragment.fragment_id] = (cursor, cursor + len(starts_i))
            self.fragment_lengths[fragment.fragment_id] = int(len(starts_i))
            cursor += len(starts_i)
        if cursor == 0:
            raise ValueError("source pool has no legal windows")
        self.fragment_ids = np.concatenate(fragment_ids)
        self.starts = np.concatenate(starts)
        # Pack each trajectory once; all arms use the same vectorised gather.
        # Legal starts are still derived per trajectory, so windows never cross
        # boundaries in the packed storage.
        self._values = np.concatenate([f.data for f in pool]).astype(np.float32, copy=False)
        offset = 0
        self._value_offsets = {}
        for fragment in pool:
            self._value_offsets[fragment.fragment_id] = offset
            offset += fragment.length

    def __len__(self) -> int:
        return int(len(self.fragment_ids))

    @property
    def ids(self) -> list[tuple[int, int]]:
        return [(int(f), int(s)) for f, s in zip(self.fragment_ids, self.starts)]

    def bounds(self, fragment_id: int) -> tuple[int, int]:
        return self._bounds[fragment_id]

    def locate(self, fragment_id: int, start: int) -> int:
        """Global position of (fragment_id, legal start)."""
        lo, hi = self._bounds[fragment_id]
        offset = (int(start) - 1) // self.stride
        position = lo + offset
        if position >= hi or int(self.starts[position]) != int(start):
            raise KeyError((fragment_id, start))
        return position

    def windows_at(self, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.int64)
        width = self.context_length + self.prediction_length
        if len(positions) == 0:
            return np.zeros((0, width), dtype=np.float32)
        rows = np.empty((len(positions), width), dtype=np.float32)
        block = 65536
        for begin in range(0, len(positions), block):
            chunk = positions[begin : begin + block]
            fragment_ids = self.fragment_ids[chunk]
            starts = self.starts[chunk] - 1  # convert 1-indexed starts once
            unique, inverse = np.unique(fragment_ids, return_inverse=True)
            base = np.asarray([self._value_offsets[int(fid)] for fid in unique])[inverse]
            offsets = (base + starts)[:, None] + np.arange(width)
            rows[begin:begin + len(chunk)] = self._values[offsets]
        return rows

    def windows(self, ids: list[tuple[int, int]]) -> np.ndarray:
        """Compat path for (fragment_id, start) tuples (small pools)."""
        if not ids:
            return np.zeros((0, self.context_length + self.prediction_length),
                            dtype=np.float32)
        return self.windows_at(np.asarray(
            [self.locate(f, s) for f, s in ids], dtype=np.int64))

    def fragment_arrays(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(fragment_ids, starts) arrays for the given positions."""
        positions = np.asarray(positions, dtype=np.int64)
        return self.fragment_ids[positions], self.starts[positions]

    def positions_by_fragment(self, stride: int = 1) -> dict[int, np.ndarray]:
        return {fid: np.arange(lo, hi) for fid, (lo, hi) in self._bounds.items()}


# ---------------------------------------------------------------------------
# Pool-side CRB cache (shared across datasets, ablation, and curves)
# ---------------------------------------------------------------------------

def pool_crb_fingerprint(pool_manifest: dict, config: CRBConfig) -> str:
    payload = {
        "pool_seed": pool_manifest.get("seed"),
        "pool_equivalents": pool_manifest.get("equivalents_target"),
        "pool_points": pool_manifest.get("total_points"),
        "cover": config.cover_density, "error": config.error_density,
        "split": config.split_density, "stride": config.window_stride,
        "mean_epsilon": config.mean_epsilon,
        "endpoint_epsilon": config.endpoint_epsilon,
        "curvature_cap": config.curvature_cap,
        "initial": config.initial_placement, "error_placement": config.error_placement,
        "detection": config.detection,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


from assets.gradient.blocks import save_pool_blocks, load_pool_blocks, compute_pool_blocks


def _streamed_top_b(reference_matrix: np.ndarray, candidate_directions_loader,
                    candidate_count: int, take: int, chunk: int = 32768,
                    device: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-query top-B over chunked candidates; returns (values, indices).

    torch/GPU matmul + topk with a running merge when available, numpy
    argsort otherwise.  Ties inside a chunk resolve by candidate order.
    """
    try:
        import torch

        use_torch = torch.cuda.is_available()
    except Exception:
        use_torch = False
    if use_torch and device != "cpu":
        queries = torch.from_numpy(reference_matrix).to(device or "cuda")
        best_values = torch.full((len(queries), take), -float("inf"),
                                 device=queries.device)
        best_indices = torch.full((len(queries), take), -1, dtype=torch.long,
                                  device=queries.device)
        for lo in range(0, candidate_count, chunk):
            hi = min(lo + chunk, candidate_count)
            candidate_matrix = torch.from_numpy(
                candidate_directions_loader(lo, hi)).to(queries.device)
            scores = queries @ candidate_matrix.T
            k = min(take, hi - lo)
            vals, idx = torch.topk(scores, k=k, dim=1)
            merged_values = torch.cat([best_values, vals], dim=1)
            merged_indices = torch.cat([best_indices, idx + lo], dim=1)
            order = torch.argsort(-merged_values, dim=1, stable=True)[:, :take]
            best_values = torch.gather(merged_values, 1, order)
            best_indices = torch.gather(merged_indices, 1, order)
        return best_values.cpu().numpy(), best_indices.cpu().numpy()
    best_values = np.full((len(reference_matrix), take), -np.inf, dtype=np.float32)
    best_indices = np.full((len(reference_matrix), take), -1, dtype=np.int64)
    for lo in range(0, candidate_count, chunk):
        hi = min(lo + chunk, candidate_count)
        candidate_matrix = candidate_directions_loader(lo, hi)
        scores = reference_matrix @ candidate_matrix.T
        merged_values = np.concatenate([best_values, scores], axis=1)
        merged_indices = np.concatenate(
            [best_indices, np.arange(lo, hi, dtype=np.int64)[None, :].repeat(
                len(reference_matrix), 0)], axis=1)
        order = np.argsort(-merged_values, axis=1, kind="stable")[:, :take]
        best_values = np.take_along_axis(merged_values, order, axis=1)
        best_indices = np.take_along_axis(merged_indices, order, axis=1)
    return best_values, best_indices


def rags_crb_select(
    reference_fragments: list[TimeSeriesFragment],
    pool: list[TimeSeriesFragment],
    index: WindowIndex,
    *,
    gradient_computer,
    context_length: int,
    prediction_length: int,
    window_stride: int,
    cover_density: float,
    error_density: float,
    split_density: float,
    block_budget: int,
    n_select: int,
    mean_epsilon: float,
    endpoint_epsilon: float,
    tie_seed: int,
    curvature_cap: float = 1.0,
    pool_cache_dir: Path | None = None,
    pool_manifest: dict | None = None,
    pool_blocks: tuple[list[GradientBlock], dict] | None = None,
    pool_block_cache: dict | None = None,
    device: str | None = None,
) -> ArmOutput:
    """RAGS: unchanged CRB reconstruction/recall, whole-block point-budget delivery.

    Every reference block queries all passing source blocks (no global
    stop); each query returns its top-B blocks by unit-direction cosine.
    Recalled blocks receive their maximum returning cosine. Deliver their
    complete window envelopes in score order until N*(C+H) points, counting
    overlapping envelopes separately and reporting unique coverage separately.
    """
    from .regions import REVISION, block_regions, deliver_regions

    started = time.time()
    config = CRBConfig(
        context_length=context_length,
        prediction_length=prediction_length,
        window_stride=window_stride,
        cover_density=cover_density,
        error_density=error_density,
        split_density=split_density,
        mean_epsilon=mean_epsilon,
        endpoint_epsilon=endpoint_epsilon,
        curvature_cap=curvature_cap,
    )
    engine = CRBEngine(config, gradient_computer)

    # ---- source side: cached across datasets / steps ----------------------
    if pool_blocks is None and pool_block_cache is not None:
        pool_blocks = pool_block_cache.get("blocks")
    pool_cached = pool_blocks is not None
    cache_path = None
    if pool_blocks is None and pool_cache_dir is not None and pool_manifest is not None:
        cache_path = Path(pool_cache_dir) / ("pool_crb_%s.npz" % pool_crb_fingerprint(
            pool_manifest, config))
        if cache_path.exists():
            pool_blocks = load_pool_blocks(cache_path)
            pool_cached = True
            print("  [pool CRB] loaded cache: %d blocks" % len(pool_blocks[0]),
                  flush=True)
    if pool_blocks is None:
        pool_blocks = compute_pool_blocks(pool, index, config=config,
                                          gradient_computer=gradient_computer)
        if cache_path is not None:
            save_pool_blocks(cache_path, pool_blocks[0], pool_blocks[1])
    candidate_blocks, pool_counters = pool_blocks
    if pool_block_cache is not None:
        pool_block_cache["blocks"] = pool_blocks
    pool_done = time.perf_counter()

    # ---- reference side ---------------------------------------------------
    reference_results = CRBEngineMulti(config, gradient_computer).process_fragments(
        reference_fragments)
    reference_blocks = [b for r in reference_results for b in r.blocks]
    if not reference_blocks or not candidate_blocks:
        raise RuntimeError("CRB produced no usable blocks on one of the sides")
    reference_matrix = np.stack(
        [b.direction for b in reference_blocks]).astype(np.float32)
    reference_done = time.perf_counter()

    # ---- streamed top-B matching ------------------------------------------
    take = min(block_budget, len(candidate_blocks))

    def loader(lo: int, hi: int) -> np.ndarray:
        return np.stack([b.direction for b in candidate_blocks[lo:hi]]).astype(np.float32)

    top_values, top_indices = _streamed_top_b(
        reference_matrix, loader, len(candidate_blocks), take, device=device)
    matching_done = time.perf_counter()

    # ---- block best-cosine over returning (query, block) pairs -----------
    pair_indices = top_indices.reshape(-1)
    pair_scores = top_values.reshape(-1)
    # A source block can be recalled by many reference queries. Reduce its
    # scores once; no pool-wide window score array or expansion is needed.
    unique_blocks, inverse = np.unique(pair_indices, return_inverse=True)
    best_scores = np.full(len(unique_blocks), -np.inf, dtype=np.float32)
    np.maximum.at(best_scores, inverse, pair_scores.astype(np.float32))
    returned_windows = sum(len(candidate_blocks[int(j)].start_positions) for j in unique_blocks)
    rng = np.random.default_rng(tie_seed)
    order = np.lexsort((rng.random(len(unique_blocks)), -best_scores))
    windows = deliver_regions(index, block_regions(index, [
        candidate_blocks[int(j)].start_positions for j in unique_blocks[order]
    ]), n=n_select)

    def _side_summary(results: list[CRBResult]) -> dict:
        return {
            "trajectories": len(results),
            "legal_positions": int(sum(len(r.valid_starts) for r in results)),
            "gradients": int(sum(r.computed_gradients for r in results)),
            "blocks": int(sum(len(r.blocks) for r in results)),
            "discarded_no_signal": int(sum(r.discarded_no_signal for r in results)),
            "discarded_antagonistic": int(sum(r.discarded_antagonistic for r in results)),
            "banned_budget": int(sum(r.banned_budget for r in results)),
        }

    reference_summary = _side_summary(reference_results)
    cost = SelectionCost(
        gradient_computations=(reference_summary["gradients"] + pool_counters["gradients"]),
        index_bytes=int(len(candidate_blocks) * candidate_blocks[0].direction.nbytes),
        wall_seconds=0.0,
        notes={"reference": reference_summary, "candidate": pool_counters,
               "pool_cached": pool_cached},
    )
    return ArmOutput(
        name="rags",
        selected=windows,
        window_ids=[],
        cost=cost,
        diagnostics={
            "stage_seconds": {"reference": reference_done - pool_done,
                              "matching": matching_done - reference_done,
                              "deployment": time.perf_counter() - matching_done},
            "reference_queries": len(reference_blocks),
            "candidate_blocks": len(candidate_blocks),
            "block_budget": block_budget,
            "selection_revision": REVISION,
            "delivery": windows.coverage(),
            "returned_windows": int(returned_windows),
            "selected": int(len(windows.regions)),
            "requested": int(n_select),
            "score_stats": {
                "max": float(best_scores.max()),
                "mean": float(best_scores.mean()),
                "min_selected": float(best_scores[order[len(windows.regions) - 1]]),
            },
        },
    )


# ---------------------------------------------------------------------------
# Random (disjoint grid) and trajectory-level baseline
# ---------------------------------------------------------------------------

RANDOM_SAMPLING_REVISION = "feasible-pool-online-replacement-v1"


def streaming_random_select(index: WindowIndex, *, n_select: int, seed: int = 101) -> ArmOutput:
    """Expose only the supplied feasible pool; fine_tune draws every batch with replacement.

    n_select controls update budgeting, NOT a fixed preselected subset.
    The training seed controls the draws; this constructor does not sample.
    """
    from .regions import RegionWindows
    length = index.context_length + index.prediction_length
    fragments = [f for f in index.by_fragment.values() if f.length >= length]
    offsets = np.cumsum([0] + [f.length for f in fragments])
    selected = RegionWindows(np.concatenate([f.data for f in fragments]), offsets,
        [(f.fragment_id, 0, f.length) for f in fragments], window_length=length, budget_windows=n_select)
    return ArmOutput('random', selected, [], SelectionCost(),
        dict(random_sampling_revision=RANDOM_SAMPLING_REVISION,
             sampling='uniform legal starts in supplied feasible pool, replacement each training batch',
             fixed_preselection=False, actual_pool_points=int(offsets[-1]),
             source_entities=[f.entity_id for f in fragments],
             training_budget_windows=n_select, selection_point_budget_applies=False))


def random_select(index: WindowIndex, *, n_select: int, seed: int) -> ArmOutput:
    """Uniform without replacement over the pool's disjoint window grid."""
    from .deploy import disjoint_slots

    slots_f, slots_s = disjoint_slots(
        index.fragment_lengths,
        window_length=index.context_length + index.prediction_length)
    rng = np.random.default_rng(seed)
    chosen_slots = rng.choice(len(slots_s), size=min(n_select, len(slots_s)),
                              replace=False)
    order = np.sort(chosen_slots)
    positions = np.asarray(
        [index.locate(int(slots_f[i]), int(slots_s[i])) for i in order],
        dtype=np.int64)
    return ArmOutput(
        name="random",
        selected=index.windows_at(positions),
        window_ids=[(int(f), int(s))
                    for f, s in zip(*index.fragment_arrays(positions))],
        cost=SelectionCost(wall_seconds=0.0,
                           notes={"pool_size": len(index),
                                  "disjoint_slots": int(len(slots_s))}),
        diagnostics={"selected": int(len(positions)),
                     "deployment": "uniform over disjoint 576-point slot grid"},
    )

def trajectory_level_select(reference_fragments, pool, index, *, engine, n_select=None,
                            context_length, prediction_length, tie_seed=5,
                            cover_density=.02, error_density=.02, split_density=.005,
                            cache_dir=None, pool_manifest=None, resampling=None, gradient_assets=None):
    raise RuntimeError("Retired hard trajectory delivery; use canonical V5 full_pool selection with prepared assets")


def empty_arm(name: str) -> ArmOutput:
    return ArmOutput(
        name=name,
        selected=np.zeros((0, 0), dtype=np.float32),
        window_ids=[],
        cost=SelectionCost(),
        diagnostics={},
    )


__all__ = [
    "SelectionCost",
    "ArmOutput",
    "WindowIndex",
    "rags_crb_select",
    "random_select",
    "trajectory_level_select",
    "empty_arm",
    "compute_pool_blocks",
    "load_pool_blocks",
    "save_pool_blocks",
    "pool_crb_fingerprint",
]
