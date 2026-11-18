"""Coverage-then-cancellation anchor selection used by the main-crb protocol.

The protocol has exactly two stages over one trajectory's legal window starts:

1. ``q_cov`` coverage: evenly spread anchors (both endpoints included), giving
   contiguous closed intervals between neighbouring anchors.
2. ``q_can`` cancellation split: rank every interval by how strongly its two
   endpoint gradients cancel, then spend the split budget on the most
   cancelling intervals, each receiving one midpoint anchor.

Cancellation strength is the relative magnitude of the endpoint gradient sum::

    kappa = ||g_left + g_right|| / (||g_left|| + ||g_right||)

``kappa`` is scale invariant; ``0`` is perfect cancellation and ``1`` means no
cancellation. A zero denominator carries no evidence of cancellation, so
``kappa`` is defined as ``1`` exactly like the frozen CRB kappa contract.
Lower ``kappa`` ranks higher.

Budget semantics (final equivalent spacing ``q``)::

    cover spacing      = cover_spacing_ratio * q   (default 1.5 * q)
    split budget       = floor(split_fraction * interval_count)  (default 0.5)

The default cover:split anchor counts are therefore 2:1, and the expected total
anchor count is ``N / q``. ``q_can == q_cov`` holds in the budget sense that
both stages receive an equal share of the joint ``N / q`` budget only when
``split_fraction`` is ``1/3``; the shipped default ``1/2`` deliberately spends
the split stage on the stronger half of the intervals so that the ranking
actually decides which intervals get refined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .ads import uniform_anchor_positions


@dataclass(frozen=True)
class CRBSplitConfig:
    """Anchor selection configuration for the cover plus cancellation split."""

    q: float = 32.0
    cover_spacing_ratio: float = 1.5
    split_fraction: float = 0.5

    def __post_init__(self) -> None:
        if not np.isfinite(self.q) or self.q < 1:
            raise ValueError("q must be finite and at least 1")
        if not np.isfinite(self.cover_spacing_ratio) or self.cover_spacing_ratio <= 1:
            raise ValueError("cover_spacing_ratio must be finite and greater than 1")
        if not np.isfinite(self.split_fraction) or not 0 < self.split_fraction <= 1:
            raise ValueError("split_fraction must be in (0, 1]")

    @property
    def cover_spacing(self) -> float:
        return self.q * self.cover_spacing_ratio

    @property
    def split_spacing(self) -> float:
        """Nominal spacing implied by the split budget alone."""
        return self.q / self.split_fraction

    def split_budget(self, interval_count: int) -> int:
        if interval_count <= 0:
            return 0
        return min(interval_count, int(math.floor(self.split_fraction * interval_count)))


@dataclass
class CRBIntervalRecord:
    left: int
    right: int
    kappa: float
    rank: int
    split: bool
    midpoint: int | None

    @property
    def width(self) -> int:
        return self.right - self.left


@dataclass
class CRBSelectionResult:
    valid_count: int
    cover_spacing: float
    cover_positions: np.ndarray
    split_positions: np.ndarray
    anchor_positions: np.ndarray
    intervals: list[CRBIntervalRecord] = field(default_factory=list)
    split_budget: int = 0
    split_used: int = 0
    skipped_unsplittable: int = 0

    @property
    def cover_anchor_count(self) -> int:
        return int(len(self.cover_positions))

    @property
    def interval_count(self) -> int:
        return len(self.intervals)

    @property
    def total_anchor_count(self) -> int:
        return int(len(self.anchor_positions))

    @property
    def equivalent_spacing(self) -> float:
        if self.total_anchor_count == 0:
            return float("inf")
        return float(self.valid_count) / float(self.total_anchor_count)


def interval_kappa(left_gradient: np.ndarray, right_gradient: np.ndarray) -> float:
    """Return the relative endpoint cancellation score of one interval."""
    left = np.asarray(left_gradient, dtype=np.float64)
    right = np.asarray(right_gradient, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("interval endpoint gradients must have matching shapes")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("interval endpoint gradients must be finite")
    denominator = float(np.linalg.norm(left) + np.linalg.norm(right))
    if denominator == 0:
        return 1.0
    return float(np.linalg.norm(left + right) / denominator)


def _interval_kappas(gradients: np.ndarray, positions: np.ndarray) -> np.ndarray:
    rows = np.asarray(gradients[positions], dtype=np.float64)
    norms = np.linalg.norm(rows, axis=1)
    sums = np.linalg.norm(rows[:-1] + rows[1:], axis=1)
    denominators = norms[:-1] + norms[1:]
    kappas = np.ones(len(sums), dtype=np.float64)
    positive = denominators > 0
    kappas[positive] = sums[positive] / denominators[positive]
    if not np.isfinite(kappas).all():
        raise ValueError("interval cancellation scores must be finite")
    return kappas


def select_crb_positions(gradients: np.ndarray,
                         config: CRBSplitConfig) -> CRBSelectionResult:
    """Run coverage, cancellation ranking, and the capped split stage.

    ``gradients`` is one row per legal window start of a single trajectory; only
    the coverage anchors are read, and the split stage reuses those rows for the
    newly created endpoints. Ties on ``kappa`` prefer the wider interval, then
    the leftmost interval, so selection is fully deterministic.
    """
    matrix = np.asarray(gradients)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("gradients must be a nonempty two dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("gradients must be finite")
    total = int(matrix.shape[0])
    cover = np.asarray(uniform_anchor_positions(total, config.cover_spacing),
                       dtype=np.int64)
    if total > 1 and (len(cover) < 2 or cover[0] != 0 or cover[-1] != total - 1):
        raise ValueError("coverage anchors must include both endpoints")
    if len(cover) == 1:
        return CRBSelectionResult(
            valid_count=total, cover_spacing=config.cover_spacing,
            cover_positions=cover, split_positions=np.zeros(0, dtype=np.int64),
            anchor_positions=cover.copy(),
            intervals=[CRBIntervalRecord(0, 0, 1.0, 0, False, None)])

    kappas = _interval_kappas(matrix, cover)
    order = sorted(range(len(kappas)),
                   key=lambda index: (float(kappas[index]),
                                      -(int(cover[index + 1]) - int(cover[index])),
                                      int(cover[index])))
    budget = config.split_budget(len(kappas))
    split_positions: list[int] = []
    records: list[CRBIntervalRecord] = []
    skipped = 0
    ranks = {index: rank for rank, index in enumerate(order)}
    chosen: list[int] = []
    for index in order:
        if len(chosen) >= budget:
            break
        left, right = int(cover[index]), int(cover[index + 1])
        midpoint = (left + right) // 2
        if left >= midpoint or midpoint >= right:
            skipped += 1
            continue
        chosen.append(index)
        split_positions.append(midpoint)
    chosen_set = set(chosen)
    for index in range(len(kappas)):
        left, right = int(cover[index]), int(cover[index + 1])
        records.append(CRBIntervalRecord(
            left=left, right=right, kappa=float(kappas[index]),
            rank=ranks[index], split=index in chosen_set,
            midpoint=(left + right) // 2 if index in chosen_set else None))
    split_array = np.asarray(sorted(split_positions), dtype=np.int64)
    anchors = np.asarray(sorted(set(int(value) for value in cover)
                                | set(int(value) for value in split_array)),
                         dtype=np.int64)
    if len(anchors) != int(len(cover) + len(split_array)):
        raise RuntimeError("split anchors must not duplicate coverage anchors")
    return CRBSelectionResult(
        valid_count=total, cover_spacing=config.cover_spacing,
        cover_positions=cover, split_positions=split_array,
        anchor_positions=anchors, intervals=records,
        split_budget=budget, split_used=len(split_array),
        skipped_unsplittable=skipped)


def selection_audit(result: CRBSelectionResult, config: CRBSplitConfig) -> dict:
    """Return a JSON-safe summary of one trajectory's selection."""
    split_kappa = [record.kappa for record in result.intervals if record.split]
    kept_kappa = [record.kappa for record in result.intervals if not record.split]
    order = sorted(result.intervals, key=lambda record: record.rank)
    return {
        "valid_count": int(result.valid_count),
        "q": float(config.q),
        "cover_spacing_ratio": float(config.cover_spacing_ratio),
        "cover_spacing": float(result.cover_spacing),
        "split_fraction": float(config.split_fraction),
        "split_spacing_nominal": float(config.split_spacing),
        "cover_anchor_count": result.cover_anchor_count,
        "interval_count": result.interval_count,
        "split_budget": int(result.split_budget),
        "split_used": int(result.split_used),
        "skipped_unsplittable": int(result.skipped_unsplittable),
        "split_anchor_count": int(len(result.split_positions)),
        "total_anchor_count": result.total_anchor_count,
        "equivalent_spacing": float(result.equivalent_spacing),
        "target_spacing": float(config.q),
        "mean_kappa_split": float(np.mean(split_kappa)) if split_kappa else None,
        "mean_kappa_kept": float(np.mean(kept_kappa)) if kept_kappa else None,
        "interval_rank_order": [
            {"left": record.left, "right": record.right, "width": record.width,
             "kappa": record.kappa, "rank": record.rank, "split": record.split,
             "midpoint": record.midpoint}
            for record in order
        ],
    }


__all__ = [
    "CRBSplitConfig",
    "CRBIntervalRecord",
    "CRBSelectionResult",
    "interval_kappa",
    "select_crb_positions",
    "selection_audit",
]
