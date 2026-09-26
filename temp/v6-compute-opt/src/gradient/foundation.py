"""Middle-layer projected-gradient engine for trainable foundation adapters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from adapters.foundation import ModelCapability, build_adapter
from gradient.projection import ParameterHashProjector


DEFAULT_TIMESFM25_LAYER = "output_projection_point.output_layer.weight"


@dataclass(frozen=True)
class FoundationGradientConfig:
    model_key: str = "timesfm-2.5"
    context_length: int = 2048
    prediction_length: int = 64
    layer: str = DEFAULT_TIMESFM25_LAYER
    projection_dimension: int = 4096
    projection_seed: int = 42
    gradient_batch_size: int = 1

    def __post_init__(self) -> None:
        for name in (
            "context_length", "prediction_length", "projection_dimension",
            "gradient_batch_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("%s must be a positive integer" % name)
        if not self.model_key or not self.layer:
            raise ValueError("model_key and layer must be non-empty")

    @property
    def window_length(self) -> int:
        return self.context_length + self.prediction_length


def _select_parameter(model, name: str):
    exact = [(key, value) for key, value in model.named_parameters() if key == name]
    if len(exact) == 1:
        return exact[0]
    suffix = [(key, value) for key, value in model.named_parameters() if key.endswith("." + name)]
    if len(suffix) == 1:
        return suffix[0]
    if not exact and not suffix:
        raise ValueError("gradient layer not found: %s" % name)
    raise ValueError("gradient layer is ambiguous: %s" % name)


class FoundationGradientEngine:
    """One native-loss gradient per authorized, materialized static window."""

    def __init__(self, config: FoundationGradientConfig | None = None, *, device: str = "auto", adapter=None) -> None:
        self.config = config or FoundationGradientConfig()
        self.adapter = adapter or build_adapter(self.config.model_key, device=device)
        self.adapter.spec.require(ModelCapability.GRADIENT)
        self.model = self.adapter.load()
        self.model.eval()
        self.layer_name, self.parameter = _select_parameter(self.model, self.config.layer)
        self.device = next(self.model.parameters()).device
        self.projector = ParameterHashProjector(
            self.parameter.numel(),
            projection_dimension=self.config.projection_dimension,
            seed=self.config.projection_seed,
            device=self.device,
        )

    def gradient_sketch(self, window: np.ndarray) -> tuple[float, np.ndarray]:
        values = np.asarray(window, dtype=np.float32).reshape(1, -1)
        loss = self.adapter.training_loss(
            values,
            context_length=self.config.context_length,
            prediction_length=self.config.prediction_length,
        )
        gradient = torch.autograd.grad(loss, self.parameter, allow_unused=False)[0]
        sketch = self.projector.project(gradient)
        return float(loss.detach().float().cpu()), sketch.detach().cpu().numpy()

    def batch_gradient_sketch(self, windows: np.ndarray, batch_size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(windows, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.config.window_length:
            raise ValueError("gradient windows have invalid geometry")
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("gradient windows must be non-empty and finite")
        _ = batch_size  # interface compatibility; native gradients remain per-example
        losses = np.empty(len(values), dtype=np.float64)
        sketches = np.empty((len(values), self.config.projection_dimension), dtype=np.float32)
        for row, window in enumerate(values):
            losses[row], sketches[row] = self.gradient_sketch(window)
        if not np.isfinite(losses).all() or not np.isfinite(sketches).all():
            raise ValueError("projected gradient output is not finite")
        return losses, sketches


__all__ = [
    "DEFAULT_TIMESFM25_LAYER", "FoundationGradientConfig", "FoundationGradientEngine",
]
