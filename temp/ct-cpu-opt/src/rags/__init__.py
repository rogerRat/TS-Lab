"""RAGS: Adaptive Reference Quadrature and Stability-Constrained Data Selection."""

from .core import (
    RAGSConfig,
    TimeSeriesFragment,
    WindowStartPoint,
    AdaptiveInterval,
    GradientTrajectory,
    CandidateWindow,
    compute_cosine_similarity,
    compute_inner_product,
)
from .quadrature import (
    AdaptiveQuadratureEngine,
    QuadratureResult,
    MidpointResidualComputer,
)
from .perturbation import (
    ReferencePerturbationEngine,
    MovingBlockBootstrap,
    ReferenceVersion,
)
from .selection import (
    StableRankingSelector,
    CandidateScore,
    FinalSelection,
)
from .pipeline import RAGSPipeline, RAGSRunReport
from .crb import (
    CRBConfig,
    GradientBlock,
    CRBResult,
    BlockMatch,
    CRBRunReport,
    CRBEngine,
    CRBPipeline,
    match_blocks,
    top_unique_candidate_blocks,
)

__all__ = [
    "RAGSConfig",
    "TimeSeriesFragment",
    "WindowStartPoint",
    "AdaptiveInterval",
    "GradientTrajectory",
    "CandidateWindow",
    "compute_cosine_similarity",
    "compute_inner_product",
    "AdaptiveQuadratureEngine",
    "QuadratureResult",
    "MidpointResidualComputer",
    "ReferencePerturbationEngine",
    "MovingBlockBootstrap",
    "ReferenceVersion",
    "StableRankingSelector",
    "CandidateScore",
    "FinalSelection",
    "RAGSPipeline",
    "RAGSRunReport",
    "CRBConfig",
    "GradientBlock",
    "CRBResult",
    "BlockMatch",
    "CRBRunReport",
    "CRBEngine",
    "CRBPipeline",
    "match_blocks",
    "top_unique_candidate_blocks",
]
