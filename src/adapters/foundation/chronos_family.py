"""Chronos-2 and Chronos-Bolt adapters using the released model loss."""

from __future__ import annotations

import math
import numpy as np

from .base import FoundationModelAdapter
from .components import as_finite_batch, split_static_windows


class ChronosFamilyAdapter(FoundationModelAdapter):
    """Univariate TS-Lab windows with independent series in every batch."""

    def load(self):
        if self._model is None:
            from chronos import BaseChronosPipeline
            from configs.paths import PRETRAINED_MODELS_DIR
            pipeline = BaseChronosPipeline.from_pretrained(
                self.spec.model_id,
                revision=self.spec.revision,
                cache_dir=str(PRETRAINED_MODELS_DIR / "huggingface"),
                device_map=self.device,
            )
            self._model = pipeline.model
        return self._model

    def _tensor(self, values):
        import torch
        model = self.load()
        return torch.as_tensor(values, dtype=torch.float32, device=next(model.parameters()).device)

    def _forward(self, context, future=None):
        model = self.load()
        if self.spec.family == "chronos2":
            patches = math.ceil((future.shape[1] if future is not None else self._horizon)
                                / model.chronos_config.output_patch_size)
            return model(
                context=context,
                group_ids=self._independent_groups(context),
                num_output_patches=patches,
                future_target=future,
            )
        return model(context=context, target=future)

    @staticmethod
    def _independent_groups(context):
        import torch
        return torch.arange(context.shape[0], device=context.device)

    def forecast(self, contexts: np.ndarray, *, prediction_length: int) -> np.ndarray:
        import torch
        values = as_finite_batch(contexts, name="contexts")
        if values.shape[1] > self.spec.max_context:
            raise ValueError("context exceeds the model contract")
        if prediction_length < 1:
            raise ValueError("prediction_length must be positive")
        model = self.load()
        limit = (1024 if self.spec.family == "chronos2"
                 else model.chronos_config.prediction_length)
        if prediction_length > limit:
            raise ValueError("prediction length exceeds the model contract")
        self._horizon = prediction_length
        with torch.no_grad():
            output = self._forward(self._tensor(values))
            levels = model.chronos_config.quantiles
            median = min(range(len(levels)), key=lambda index: abs(levels[index] - 0.5))
            result = output.quantile_preds[:, median, :prediction_length]
        return result.detach().float().cpu().numpy()

    def per_window_losses(self, windows: np.ndarray, *, context_length: int,
                          prediction_length: int):
        """One model forward and one independent quantile loss per window."""
        import torch
        context, future = split_static_windows(
            windows, context_length=context_length,
            prediction_length=prediction_length,
        )
        if context_length > self.spec.max_context:
            raise ValueError("context exceeds the model contract")
        model = self.load()
        if self.spec.family == "chronos_bolt" and prediction_length > model.chronos_config.prediction_length:
            raise ValueError("Chronos-Bolt training horizon cannot exceed the fixed head")
        self._horizon = prediction_length
        context_tensor = self._tensor(context)
        future_tensor = self._tensor(future)
        output = self._forward(context_tensor)
        predicted = output.quantile_preds
        batch, quantiles, output_length = predicted.shape
        _, loc_scale = model.instance_norm(context_tensor)
        normalized_pred, _ = model.instance_norm(predicted.reshape(batch, -1), loc_scale)
        normalized_pred = normalized_pred.reshape(batch, quantiles, output_length)
        normalized_target, _ = model.instance_norm(future_tensor, loc_scale)
        if prediction_length < output_length:
            normalized_target = torch.nn.functional.pad(
                normalized_target, (0, output_length - prediction_length), value=0.0,
            )
        mask = torch.arange(output_length, device=predicted.device) < prediction_length
        levels = torch.as_tensor(model.chronos_config.quantiles,
                                 dtype=torch.float32, device=predicted.device)[None, :, None]
        error = normalized_target[:, None, :] - normalized_pred
        loss = 2 * torch.abs(error * ((error <= 0).float() - levels))
        loss = loss * mask[None, None, :]
        if self.spec.family == "chronos_bolt":
            return loss.mean(dim=-2).sum(dim=-1)
        return loss.mean(dim=-1).sum(dim=-1)

    def training_loss(self, windows: np.ndarray, *, context_length: int,
                      prediction_length: int):
        if context_length > self.spec.max_context:
            raise ValueError("context exceeds the model contract")
        context, future = split_static_windows(
            windows, context_length=context_length,
            prediction_length=prediction_length,
        )
        if self.spec.family == "chronos_bolt" and prediction_length > 64:
            raise ValueError("Chronos-Bolt training horizon cannot exceed 64")
        if self.spec.family == "chronos2" and prediction_length > 1024:
            raise ValueError("Chronos-2 training horizon cannot exceed 1024")
        self._horizon = prediction_length
        output = self._forward(self._tensor(context), self._tensor(future))
        if output.loss is None:
            raise RuntimeError("Chronos model returned no training loss")
        return output.loss


__all__ = ["ChronosFamilyAdapter"]
