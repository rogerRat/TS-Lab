"""Gradient computation and bounded random projection for TS3."""

from .chronos import (
    ChronosTinyConfig,
    ChronosTinyGradientEngine,
    DEFAULT_CHRONOS_TINY_MODEL,
    DEFAULT_LAYER,
    DEFAULT_WARMUP_MODEL_COUNT,
    DEFAULT_WARMUP_MODEL_ROOT,
    DEFAULT_WARMUP_SEEDS,
    per_sample_cross_entropy,
    select_parameter,
)
from .exceptions import ModelNotFoundError
from .projection import ParameterHashProjector, TensorSketchProjector
from .foundation import FoundationGradientConfig, FoundationGradientEngine
from .base import GradientEngine, NativeProbe
from .chronos import ChronosT5Config, ChronosT5Engine
from .timesfm import TimesFMGradientEngine
from .timemoe import TimeMoEGradientEngine

_LAZY_EXPORTS = {
    "FromScratchWarmupRunner": ("warmup", "FromScratchWarmupRunner"),
    "WarmupCandidateRunner": ("warmup", "WarmupCandidateRunner"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    from importlib import import_module
    module_name, attribute = target
    return getattr(import_module(".%s" % module_name, __name__), attribute)

__all__ = [
    "GradientEngine", "NativeProbe", "ChronosT5Config", "ChronosT5Engine",
    "TimesFMGradientEngine", "TimeMoEGradientEngine",
    "ChronosTinyConfig", "ChronosTinyGradientEngine", "DEFAULT_CHRONOS_TINY_MODEL",
    "DEFAULT_LAYER", "DEFAULT_WARMUP_MODEL_COUNT", "DEFAULT_WARMUP_MODEL_ROOT",
    "DEFAULT_WARMUP_SEEDS",
    "ModelNotFoundError", "ParameterHashProjector", "TensorSketchProjector",
    "FoundationGradientConfig", "FoundationGradientEngine",
    "FromScratchWarmupRunner", "WarmupCandidateRunner",
    "per_sample_cross_entropy", "select_parameter",
]
