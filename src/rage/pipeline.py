"""End-to-end RAGE pipeline orchestrating quadrature, perturbation, selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

from .core import RAGEConfig, TimeSeriesFragment, compute_cosine_similarity
from .quadrature import AdaptiveQuadratureEngine, QuadratureResult
from .perturbation import ReferencePerturbationEngine, ReferenceVersion
from .selection import StableRankingSelector, CandidateScore, FinalSelection


@dataclass
class RAGERunReport:
    """Summary of a full RAGE run."""

    estimated_reference_gradient: np.ndarray
    quadrature_results: list[QuadratureResult]
    versions: list[ReferenceVersion]
    candidate_scores: list[CandidateScore]
    selection: FinalSelection
    reference_gradient_norm: float
    version_norms: np.ndarray
    reference_backward_count: int
    total_start_positions: int


class RAGEPipeline:
    """Two-stage pipeline:

    Stage 1 — adaptive quadrature recovers the full-reference ranking from
    a limited number of reference gradient anchors.
    Stage 2 — dependency-preserving perturbations produce reference versions;
    selection frequency and lower-quantile scores decide the final pick.
    """

    def __init__(
        self,
        config: RAGEConfig,
        gradient_computer: Callable[[np.ndarray, int, int], np.ndarray],
    ) -> None:
        self.config = config
        self.gradient_computer = gradient_computer
        self.quadrature_engine = AdaptiveQuadratureEngine(config, gradient_computer)
        self.perturbation_engine = ReferencePerturbationEngine(config)
        self.selector = StableRankingSelector(config)

    def run(
        self,
        reference_fragments: list[TimeSeriesFragment],
        candidate_fragments: list[TimeSeriesFragment],
        candidate_gradient_computer: Callable[[np.ndarray, int, int], np.ndarray] | None = None,
    ) -> RAGERunReport:
        """Execute the full pipeline.

        Args:
            reference_fragments: target reference trajectories (segments)
            candidate_fragments: source pool fragments
            candidate_gradient_computer: optional separate gradient function
                for candidate windows; defaults to the reference one
        """
        # ---- Stage 1: adaptive quadrature over reference fragments ----
        quadrature_results = [
            self.quadrature_engine.process_fragment(fragment, index)
            for index, fragment in enumerate(reference_fragments)
        ]
        estimated_G, _ = self.quadrature_engine.compute_full_reference_gradient(
            quadrature_results, reference_fragments
        )
        reference_backward_count = sum(r.computed_gradients for r in quadrature_results)
        total_positions = sum(r.total_positions for r in quadrature_results)

        # ---- Stage 2: dependency-preserving reference versions ----
        valid_starts_list = [
            fragment.valid_starts(self.config.context_length, self.config.prediction_length)
            for fragment in reference_fragments
        ]
        versions = self.perturbation_engine.generate_reference_versions(
            quadrature_results, reference_fragments, valid_starts_list
        )

        # ---- Candidate gradients (computed once, reused across versions) ----
        computer = candidate_gradient_computer or self.gradient_computer
        candidate_gradients: list[np.ndarray] = []
        candidate_ids: list[tuple[int, int]] = []
        for fragment_index, fragment in enumerate(candidate_fragments):
            for position in fragment.valid_starts(
                self.config.context_length, self.config.prediction_length
            ):
                window = fragment.extract_window(
                    int(position), self.config.context_length, self.config.prediction_length
                )
                candidate_gradients.append(
                    np.asarray(computer(window, self.config.context_length, self.config.prediction_length),
                               dtype=np.float32)
                )
                candidate_ids.append((fragment_index, int(position)))

        candidate_scores: list[CandidateScore] = []
        if candidate_gradients:
            candidate_scores = self.selector.score_candidates(
                np.stack(candidate_gradients), candidate_ids, versions
            )

        selection = self.selector.select(candidate_scores)
        version_norms = self.selector.version_norms(versions)

        return RAGERunReport(
            estimated_reference_gradient=estimated_G,
            quadrature_results=quadrature_results,
            versions=versions,
            candidate_scores=candidate_scores,
            selection=selection,
            reference_gradient_norm=float(np.linalg.norm(estimated_G)),
            version_norms=version_norms,
            reference_backward_count=reference_backward_count,
            total_start_positions=total_positions,
        )


__all__ = ["RAGEPipeline", "RAGERunReport"]
