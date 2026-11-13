"""Moirai 2.0 univariate adapter with fixed-window quantile training loss."""

from __future__ import annotations

import math
import numpy as np

from .base import FoundationModelAdapter
from .components import as_finite_batch, split_static_windows
from .timesfm import _device


class Moirai2Adapter(FoundationModelAdapter):
    """Use the released module and forecast conversion for univariate windows."""

    def load(self):
        if self._model is None:
            from uni2ts.model.moirai2 import Moirai2Module
            from configs.paths import PRETRAINED_MODELS_DIR
            self._model = Moirai2Module.from_pretrained(
                self.spec.model_id, revision=self.spec.revision,
                cache_dir=str(PRETRAINED_MODELS_DIR / "huggingface"),
            ).to(_device(self.device))
        return self._model

    def _forecast_wrapper(self, context_length: int, prediction_length: int):
        from uni2ts.model.moirai2 import Moirai2Forecast
        return Moirai2Forecast(
            module=self.load(), prediction_length=prediction_length,
            context_length=context_length, target_dim=1,
            feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
        ).to(_device(self.device))

    def _validate_geometry(self, context_length: int, prediction_length: int):
        model = self.load()
        patch = model.patch_size
        if context_length < 1 or context_length > self.spec.max_context:
            raise ValueError("Moirai context exceeds the model contract")
        if prediction_length < 1 or prediction_length > model.num_predict_token * patch:
            raise ValueError("Moirai fixed-window horizon exceeds direct prediction tokens")
        if math.ceil(context_length / patch) + math.ceil(prediction_length / patch) > model.max_seq_len:
            raise ValueError("Moirai token sequence exceeds the checkpoint limit")

    def forecast(self, contexts: np.ndarray, *, prediction_length: int) -> np.ndarray:
        import torch
        values = as_finite_batch(contexts, name="contexts")
        self._validate_geometry(values.shape[1], prediction_length)
        wrapper = self._forecast_wrapper(values.shape[1], prediction_length)
        past = torch.as_tensor(values[:, :, None], dtype=torch.float32, device=_device(self.device))
        with torch.no_grad():
            output = wrapper(
                past_target=past,
                past_observed_target=torch.ones_like(past, dtype=torch.bool),
                past_is_pad=torch.zeros(past.shape[:2], dtype=torch.bool, device=past.device),
            )
        levels = self.load().quantile_levels
        median = min(range(len(levels)), key=lambda index: abs(levels[index] - 0.5))
        return output[:, median, :prediction_length].detach().float().cpu().numpy()

    def per_window_losses(self, windows: np.ndarray, *, context_length: int,
                          prediction_length: int):
        import torch
        context, future = split_static_windows(
            windows, context_length=context_length,
            prediction_length=prediction_length,
        )
        self._validate_geometry(context_length, prediction_length)
        model = self.load()
        was_training = model.training
        wrapper = self._forecast_wrapper(context_length, prediction_length)
        model.train(was_training)
        past = torch.as_tensor(context[:, :, None], dtype=torch.float32, device=_device(self.device))
        target = torch.as_tensor(future[:, :, None], dtype=torch.float32, device=_device(self.device))
        converted = wrapper._convert(
            model.patch_size, past,
            torch.ones_like(past, dtype=torch.bool),
            torch.zeros(past.shape[:2], dtype=torch.bool, device=past.device),
            future_target=target,
            future_observed_target=torch.ones_like(target, dtype=torch.bool),
        )
        predictions, scaled_target = model(*converted, training_mode=True)
        context_tokens = math.ceil(context_length / model.patch_size)
        future_tokens = math.ceil(prediction_length / model.patch_size)
        batch = past.shape[0]
        predicted = predictions[:, context_tokens - 1].reshape(
            batch, model.num_predict_token, model.num_quantiles, model.patch_size,
        )[:, :future_tokens]
        actual = scaled_target[:, context_tokens:context_tokens + future_tokens]
        observed = converted[1][:, context_tokens:context_tokens + future_tokens]
        levels = torch.as_tensor(model.quantile_levels, dtype=torch.float32,
                                 device=predicted.device)[None, None, :, None]
        error = actual[:, :, None, :] - predicted.float()
        pinball = torch.maximum(levels * error, (levels - 1) * error)
        masked = pinball * observed[:, :, None, :]
        denominator = observed.sum(dim=(1, 2)).clamp_min(1)
        return 2 * masked.sum(dim=(1, 2, 3)) / denominator

    def training_loss(self, windows: np.ndarray, *, context_length: int,
                      prediction_length: int):
        return self.per_window_losses(
            windows, context_length=context_length,
            prediction_length=prediction_length,
        ).mean()


__all__ = ["Moirai2Adapter"]
