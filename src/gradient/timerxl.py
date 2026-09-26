"""Timer-XL gradient adapter: native shifted-patch MSE per-window losses."""
from gradient.base import NativeProbe
import torch


class TimerXLGradientEngine(NativeProbe):
    max_gradient_batch = 32

    def __init__(self, engine, *, model_key="thuml/timer-base-84m", **kwargs):
        super().__init__(engine, model_key=model_key, **kwargs)
        # NativeProbe defaults to the Chronos/TimesFM 512->64 contract. The
        # Timer protocol is part of the gradient identity, so copy the live
        # run's context/horizon into the probe before any cache is opened.
        self.config.context_length = int(engine.config.context_length)
        self.config.prediction_length = int(engine.config.prediction_length)

    def per_window_losses(self, windows):
        """Checkpoint-native shifted-patch MSE, one independent loss per window."""
        from adapters.foundation.timerxl import native_training_batch, native_v6_training_batch, v6_single_patch_loss
        parameter = next(self.model.parameters())
        if (getattr(self.config, "context_length", None),
                getattr(self.config, "prediction_length", None)) == (480, 96):
            batch = native_v6_training_batch(windows, device=parameter.device, dtype=parameter.dtype)
            return v6_single_patch_loss(self.model, batch)
        batch = native_training_batch(windows, device=parameter.device,
                                      dtype=parameter.dtype,
                                      patch=self.model.config.input_token_len)
        hidden = self.model.model(input_ids=batch["input_ids"], use_cache=False,
                                  return_dict=True).last_hidden_state
        patch = self.model.config.input_token_len
        losses = []
        for lm_head, output_token_len in zip(self.model.lm_heads,
                                             self.model.config.output_token_lens):
            predictions = lm_head(hidden)
            seq_len = predictions.shape[1] * patch
            labels = batch["labels"][:, :seq_len - patch + output_token_len]
            shift_labels = labels.unfold(dimension=-1, size=output_token_len, step=patch)
            losses.append(torch.nn.functional.mse_loss(
                predictions.float(), shift_labels.float(), reduction="none").mean(dim=-1))
        # Same reduction as calc_ar_loss with all-ones masks, per window.
        return torch.stack(losses).mean(dim=0).mean(dim=-1)


__all__ = ["TimerXLGradientEngine"]
