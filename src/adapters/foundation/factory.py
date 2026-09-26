"""Middle-layer adapter construction without eager optional imports."""

from __future__ import annotations

from .base import FoundationModelAdapter
from .spec import get_model_spec


def build_adapter(key: str, *, device: str = "auto") -> FoundationModelAdapter:
    spec = get_model_spec(key)
    if key == "timemoe-200m":
        from .timemoe import TimeMoEAdapter
        return TimeMoEAdapter(spec, device=device)
    if key == "timesfm-3.0":
        from .timesfm import TimesFM3Adapter
        return TimesFM3Adapter(spec, device=device)
    if key == "timesfm-2.5":
        from .timesfm import TimesFM25Adapter
        return TimesFM25Adapter(spec, device=device)
    if key == "sundial-base-128m":
        from .sundial import SundialAdapter
        return SundialAdapter(spec, device=device)
    if key == "timerxl-84m":
        from .timerxl import TimerXLAdapter
        return TimerXLAdapter(spec, device=device)
    raise AssertionError("registered model has no adapter: %s" % key)


__all__ = ["build_adapter"]
