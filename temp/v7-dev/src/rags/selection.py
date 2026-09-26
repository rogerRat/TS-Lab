"""Stable ranking and candidate selection for RAGS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from .core import RAGSConfig, compute_cosine_similarity, compute_inner_product
from .perturbation import ReferenceVersion


@dataclass
class CandidateScore:
    """Per-candidate scoring summary across reference versions."""

    candidate_id: int
    fragment_id: int
    position: int
    scores: np.ndarray  # Shape: [R], one score per reference version
    stability_frequency: float  # π_i
    lower_quantile_score: float  # q_i^(α)
    mean_score: float

    def __post_init__(self) -> None:
        self.scores = np.asarray(self.scores, dtype=np.float32)

    @property
    def identifier(self) -> tuple[int, int]:
        return (self.fragment_id, self.position)


@dataclass
class FinalSelection:
    """Result of the full RAGS selection procedure."""

    selected: list[CandidateScore]
    coverage: float  # Fraction of budget satisfied by stability-constrained picks
    num_backfilled: int  # Candidates added by quantile ranking to reach budget
    stability_threshold: float
    lower_quantile: float


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores, descending."""
    k = min(k, len(scores))
    if k <= 0:
        return np.array([], dtype=np.int64)
    return np.argsort(-scores, kind="stable")[:k]


class StableRankingSelector:
    """Compute stability statistics and final selection (Section 4.4).

    Each candidate has R utility scores, one per perturbed reference
    version. Selection frequency π_i measures whether the candidate
    consistently enters the ranking budget; the lower-quantile score
    q_i^(α) measures conservative utility under unfavorable versions.
    """

    def __init__(self, config: RAGSConfig) -> None:
        self.config = config

    def score_candidates(
        self,
        candidate_gradients: np.ndarray,
        candidate_ids: list[tuple[int, int]],
        versions: list[ReferenceVersion],
    ) -> list[CandidateScore]:
        """Score all candidates against every reference version.

        Args:
            candidate_gradients: Shape [N, d], one projected gradient per candidate
            candidate_ids: (fragment_id, position) for each candidate
            versions: Reference versions produced by perturbation
        """
        gradients = np.asarray(candidate_gradients, dtype=np.float32)
        if gradients.ndim != 2:
            raise ValueError("candidate_gradients must be a 2-D array")
        if len(candidate_ids) != len(gradients):
            raise ValueError("candidate_ids must match candidate_gradients")

        num_candidates = len(gradients)
        num_versions = len(versions)
        if num_candidates == 0 or num_versions == 0:
            return []

        # Normalized candidate representations (Proposition 1 protocol)
        norms = np.linalg.norm(gradients, axis=1, keepdims=True)
        normalized = gradients / (norms + self.config.score_epsilon)

        score_matrix = np.empty((num_candidates, num_versions), dtype=np.float32)
        version_norms = np.empty(num_versions, dtype=np.float32)
        for r, version in enumerate(versions):
            g = np.asarray(version.reference_gradient, dtype=np.float32)
            version_norms[r] = np.linalg.norm(g)
            score_matrix[:, r] = normalized @ g / (np.linalg.norm(g) + self.config.score_epsilon)

        # Selection frequency: in how many versions does the candidate enter top-k
        k = self.config.ranking_budget
        membership = np.zeros((num_candidates, num_versions), dtype=np.float32)
        for r in range(num_versions):
            membership[_topk_indices(score_matrix[:, r].astype(np.float64), k), r] = 1.0
        frequencies = membership.mean(axis=1)

        alpha = self.config.lower_quantile
        quantile_scores = np.quantile(score_matrix, alpha, axis=1)
        mean_scores = score_matrix.mean(axis=1)

        results: list[CandidateScore] = []
        for i in range(num_candidates):
            fragment_id, position = candidate_ids[i]
            results.append(
                CandidateScore(
                    candidate_id=i,
                    fragment_id=fragment_id,
                    position=position,
                    scores=score_matrix[i],
                    stability_frequency=float(frequencies[i]),
                    lower_quantile_score=float(quantile_scores[i]),
                    mean_score=float(mean_scores[i]),
                )
            )
        return results

    def select(self, candidate_scores: list[CandidateScore]) -> FinalSelection:
        """Final selection: top-B by q_i^(α) among candidates with π_i ≥ π₀.

        If fewer than B candidates satisfy the stability constraint, the
        remainder is backfilled by quantile score and both the coverage and
        the backfill count are reported.
        """
        budget = self.config.selection_budget
        threshold = self.config.stability_threshold

        stable = [c for c in candidate_scores if c.stability_frequency >= threshold]
        stable.sort(key=lambda c: c.lower_quantile_score, reverse=True)

        selected = stable[:budget]
        num_backfilled = 0
        if len(selected) < budget:
            remaining = [c for c in candidate_scores if c not in selected]
            remaining.sort(key=lambda c: c.lower_quantile_score, reverse=True)
            backfill = remaining[: budget - len(selected)]
            num_backfilled = len(backfill)
            selected = selected + backfill

        coverage = len(stable) / budget if budget > 0 else 0.0
        coverage = min(coverage, 1.0)

        return FinalSelection(
            selected=selected,
            coverage=coverage,
            num_backfilled=num_backfilled,
            stability_threshold=threshold,
            lower_quantile=self.config.lower_quantile,
        )

    def version_norms(self, versions: list[ReferenceVersion]) -> np.ndarray:
        """Reference gradient norms per version, used to flag unstable versions."""
        return np.asarray(
            [np.linalg.norm(np.asarray(v.reference_gradient)) for v in versions],
            dtype=np.float64,
        )


__all__ = [
    "CandidateScore",
    "FinalSelection",
    "StableRankingSelector",
]
