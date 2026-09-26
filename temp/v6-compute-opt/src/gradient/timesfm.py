"""TimesFM gradient adapters: native full-model and optional selected-layer probe."""
from gradient.base import NativeProbe
from gradient.foundation import FoundationGradientConfig, FoundationGradientEngine
import torch


class TimesFMGradientEngine(NativeProbe):
    # BF16 native kernels at batch 32 changed single-window loss/gradients in
    # real-weight checks. Use verified shapes, including decomposition of tails.
    max_gradient_batch = 16
    power_of_two_batches = True

    def __init__(self, engine, *, model_key="google/timesfm-2.5-200m-transformers", **kwargs):
        super().__init__(engine, model_key=model_key, **kwargs)

    def per_window_losses(self, windows):
        parameter = next(self.model.parameters())
        values = torch.as_tensor(windows, dtype=parameter.dtype, device=self.device)
        context = self.config.context_length
        past, future = values[:, :context], values[:, context:]
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=parameter.dtype == torch.bfloat16):
            # Upstream uses a batch-global minimum for clipping. Apply its
            # single-window rule per row to keep cache values batch independent.
            output = self.model(past_values=past, forecast_context_len=context,
                                truncate_negative=False)
            mean = output.mean_predictions[:, :future.shape[1]]
            full = output.full_predictions[:, :future.shape[1]]
            if self.model.config.infer_is_positive:
                positive = past.amin(dim=1) >= 0
                mean = torch.where(positive[:, None], mean.clamp_min(0), mean)
                full = torch.where(positive[:, None, None], full.clamp_min(0), full)
            loss = torch.nn.functional.mse_loss(mean, future, reduction="none").mean(dim=1)
            decode = min(self.model.config.decode_index, full.shape[-1] - 1)
            indices = [i for i in range(full.shape[-1]) if i != decode]
            if indices:
                predictions = full[..., indices]
                quantiles = []
                for i, q in enumerate(self.model.config.quantiles):
                    error = future - predictions[..., i]
                    quantiles.append(torch.maximum((q - 1) * error, q * error).mean(dim=1))
                loss = loss + torch.stack(quantiles).mean(dim=0)
            return loss


class TimesFMLayerGradientEngine(FoundationGradientEngine):
    def __init__(self, config=None, **kwargs):
        super().__init__(config or FoundationGradientConfig(), **kwargs)
