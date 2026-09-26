"""Sundial gradient adapter: deterministic per-window TimeFlow losses.

Flow matching samples noise per loss evaluation; selection caches require a
fixed loss function per window, so probe losses draw noise/timesteps from a
content-seeded generator (see adapters.foundation.sundial.flow_losses).
"""
from gradient.base import NativeProbe


class SundialGradientEngine(NativeProbe):
    max_gradient_batch = 16

    def __init__(self, engine, *, model_key="thuml/sundial-base-128m", **kwargs):
        super().__init__(engine, model_key=model_key, **kwargs)

    def per_window_losses(self, windows):
        from adapters.foundation.sundial import flow_losses
        return flow_losses(self.model, windows, deterministic=True)


__all__ = ["SundialGradientEngine"]
