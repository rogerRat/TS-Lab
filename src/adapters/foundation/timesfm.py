"""Middle-layer TimesFM adapters backed by official Google implementations."""

from __future__ import annotations

import numpy as np

from .base import FoundationModelAdapter
from .components import as_finite_batch, median_forecast, split_static_windows
from .spec import ModelCapability


def _device(value: str) -> str:
    if value != "auto":
        return value
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_dtype(torch_module, target: str):
    """Use the explicit BF16 policy on every supported device."""
    return torch_module.bfloat16


class TimesFM3Adapter(FoundationModelAdapter):
    """Latest TimesFM inference adapter; training is intentionally unavailable."""

    def load(self):
        if self._model is None:
            try:
                from timesfm3 import TimesFM3Forecaster
            except ImportError as exc:
                raise RuntimeError("install the timesfm3 dependency profile") from exc
            self._model = TimesFM3Forecaster.from_pretrained(
                self.spec.model_id,
                revision=self.spec.revision,
                device=_device(self.device),
            )
        return self._model

    def forecast(self, contexts: np.ndarray, *, prediction_length: int) -> np.ndarray:
        self.spec.require(ModelCapability.FORECAST)
        values = as_finite_batch(contexts, name="contexts")
        if values.shape[1] > self.spec.max_context:
            raise ValueError("context exceeds the pinned TimesFM 3.0 contract")
        outputs = self.load().predict_batch(
            [row for row in values], prediction_length, return_quantiles=False,
        )
        return np.stack([np.asarray(item.forecast) for item in outputs]).astype(np.float32)


class TimesFM25Adapter(FoundationModelAdapter):
    """Official Transformers TimesFM 2.5 path with native training loss."""

    def load(self):
        if self._model is None:
            try:
                import torch
                from transformers import TimesFm2_5ModelForPrediction
            except ImportError as exc:
                raise RuntimeError("install the timesfm2_5 dependency profile") from exc
            target = _device(self.device)
            dtype = _load_dtype(torch, target)
            self._model = TimesFm2_5ModelForPrediction.from_pretrained(
                self.spec.model_id,
                revision=self.spec.revision,
                torch_dtype=dtype,
            ).to(target)
        return self._model

    def _forward(self, contexts: np.ndarray, futures: np.ndarray | None = None):
        import torch
        model = self.load()
        target = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        kwargs = {
            "past_values": torch.as_tensor(contexts, dtype=dtype, device=target),
            "forecast_context_len": contexts.shape[1],
        }
        if futures is not None:
            kwargs["future_values"] = torch.as_tensor(futures, dtype=dtype, device=target)
        with torch.autocast(device_type=target.type, dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
            return model(**kwargs)

    def forecast(self, contexts: np.ndarray, *, prediction_length: int) -> np.ndarray:
        self.spec.require(ModelCapability.FORECAST)
        values = as_finite_batch(contexts, name="contexts")
        if values.shape[1] > self.spec.max_context:
            raise ValueError("context exceeds the pinned TimesFM 2.5 contract")
        import torch
        self.load().eval()
        with torch.no_grad():
            output = self._forward(values)
        predictions = getattr(output, "mean_predictions", None)
        if predictions is None:
            predictions = getattr(output, "prediction_outputs", None)
        if predictions is None:
            raise ValueError("TimesFM 2.5 output has no supported prediction field")
        return median_forecast(predictions.float().cpu().numpy(), prediction_length=prediction_length)

    def training_loss(self, windows: np.ndarray, *, context_length: int, prediction_length: int):
        self.spec.require(ModelCapability.TRAINING_LOSS)
        contexts, futures = split_static_windows(
            windows, context_length=context_length, prediction_length=prediction_length,
        )
        if context_length > self.spec.max_context:
            raise ValueError("context exceeds the pinned TimesFM 2.5 contract")
        output = self._forward(contexts, futures)
        if output.loss is None:
            raise ValueError("TimesFM 2.5 did not return its native training loss")
        return output.loss

    def context_embeddings(self, contexts):
        """Mean final context-patch states; experiment contexts are 512 points."""
        import torch
        self.load().eval()
        with torch.no_grad():
            output = self._forward(as_finite_batch(contexts, name="contexts"))
            return output.last_hidden_state.mean(dim=1).float().cpu().numpy()


__all__ = ["TimesFM25Adapter", "TimesFM3Adapter"]
