"""Time-MoE via the pinned official repository, without copying its model/loss."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

from .base import FoundationModelAdapter
from .components import as_finite_batch
from .spec import ModelCapability
from .timesfm import _device

SOURCE_REVISION = "67460eb4af69eeed5fa86e591846cb0ae0744c26"


def official_classes():
    from configs.paths import PROJECT_ROOT
    root = Path(os.environ.get("RAGE_TIMEMOE_SOURCE", PROJECT_ROOT / "third_parties/Time-MoE-pinned")).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from time_moe.models.modeling_time_moe import TimeMoeForPrediction
    from time_moe.models.configuration_time_moe import TimeMoeConfig
    return TimeMoeConfig, TimeMoeForPrediction


def native_training_batch(windows, *, device):
    """Official shifted next-value pairs on each selected sequence, no extra data.

    Each 576-point selected window is a complete training sequence. As in
    TimeMoEWindowDataset(max_length=575), supervise all 575 next-value positions
    and retain all native horizon heads and router loss. No cross-window packing.
    Training uses upstream zero_scaler semantics (population std); inference
    normalizes only the available history (official inference sample std).
    """
    import torch
    values = as_finite_batch(windows, name="windows")
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    normalized = (values - mean) / np.where(std == 0, 1, std)
    return {"input_ids": torch.tensor(normalized[:, :-1], device=device),
            "labels": torch.tensor(normalized[:, 1:], device=device),
            "loss_masks": torch.ones((len(values), values.shape[1] - 1), device=device)}


class TimeMoEAdapter(FoundationModelAdapter):
    def load(self):
        if self._model is None:
            import torch
            _, model_class = official_classes()
            self._model = model_class.from_pretrained(
                self.spec.model_id, revision=self.spec.revision,
                torch_dtype=torch.bfloat16, attn_implementation="eager",
            ).to(_device(self.device))
        return self._model

    def training_loss(self, windows, *, context_length, prediction_length):
        self.spec.require(ModelCapability.TRAINING_LOSS)
        values = as_finite_batch(windows, name="windows")
        if values.shape[1] != context_length + prediction_length or values.shape[1] > self.spec.max_context:
            raise ValueError("Time-MoE requires a valid selected C+H window within 4096 points")
        model = self.load()
        batch = native_training_batch(values, device=next(model.parameters()).device)
        import torch
        p = next(model.parameters())
        batch["input_ids"] = batch["input_ids"].to(p.dtype)
        with torch.autocast(device_type=p.device.type, dtype=torch.bfloat16, enabled=p.dtype == torch.bfloat16):
            return model(**batch, use_cache=False, return_dict=True).loss.float()

    def forecast(self, contexts, *, prediction_length):
        import torch
        values = as_finite_batch(contexts, name="contexts")
        if prediction_length <= 0 or values.shape[1] + prediction_length > self.spec.max_context:
            raise ValueError("Time-MoE context plus horizon must not exceed 4096")
        model = self.load()
        model.eval()
        with torch.no_grad():
            x = torch.tensor(values, device=next(model.parameters()).device)
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True) if x.shape[1] > 1 else torch.ones_like(mean)
            std = torch.where(std == 0, torch.ones_like(std), std)
            output = model.generate(((x - mean) / std).to(next(model.parameters()).dtype), max_new_tokens=prediction_length,
                                    do_sample=False, use_cache=True)
            return (output[:, -prediction_length:] * std + mean).float().cpu().numpy()

    def context_embeddings(self, contexts):
        """Mean final context states with the same normalization as inference."""
        import torch
        values = as_finite_batch(contexts, name="contexts")
        model = self.load()
        model.eval()
        with torch.no_grad():
            x = torch.tensor(values, device=next(model.parameters()).device)
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True) if x.shape[1] > 1 else torch.ones_like(mean)
            std = torch.where(std == 0, torch.ones_like(std), std)
            output = model.model(input_ids=((x - mean) / std).to(next(model.parameters()).dtype), use_cache=False, return_dict=True)
            return output.last_hidden_state.mean(dim=1).float().cpu().numpy()
