"""Timer-XL (HF thuml/timer-base-84m) via the revision-pinned official checkpoint code.

The pinned checkpoint ships TimerForPrediction with a native shifted-patch MSE
training path (``labels`` + patch-level ``loss_masks``) and a deterministic
greedy ``generate`` with RevIN-style context normalization enabled by default.
TS3 keeps both semantics unchanged: training replays the native loss on
pre-normalized selected windows; inference is the official generation path.
"""
from __future__ import annotations

import numpy as np

from .base import FoundationModelAdapter
from .components import as_finite_batch
from .spec import ModelCapability
from .timesfm import _device


def native_training_batch(windows, *, device, dtype, patch):
    """Official shifted next-patch pairs on each selected window, no extra data.

    Each 576-point selected window is a complete training sequence. Following
    the checkpoint's calc_ar_loss contract (predictions at patch j supervise
    labels patch j), labels are the input shifted by one patch
    (input_ids=w[:-P], labels=w[P:]) and every patch is supervised
    (loss_masks=1). Windows are pre-normalized with population std over the
    full window (repo convention, cf. the Time-MoE adapter) and revin stays
    off so input and labels share one scale. Inference instead normalizes the
    available history only (checkpoint generate default revin=True).
    """
    import torch
    values = as_finite_batch(windows, name="windows")
    if values.shape[1] % patch or values.shape[1] < 2 * patch:
        raise ValueError("Timer-XL training windows must provide at least two full patches")
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    normalized = (values - mean) / np.where(std == 0, 1, std)
    supervised = values.shape[1] // patch - 1
    return {"input_ids": torch.tensor(normalized[:, :-patch], device=device, dtype=dtype),
            "labels": torch.tensor(normalized[:, patch:], device=device, dtype=dtype),
            "loss_masks": torch.ones((len(values), supervised), device=device, dtype=dtype)}


def native_v6_training_batch(windows, *, device, dtype):
    """Timer V6's single native 96-point patch objective.

    The 576-point window is explicitly split into 480 history and 96 target
    points.  Normalization statistics come from history only; the target is
    normalized with those same statistics and is never fed to the model.
    """
    import torch
    values = as_finite_batch(windows, name="windows")
    if values.shape[1] != 576:
        raise ValueError("Timer V6 windows must contain exactly 480 history + 96 target points")
    history, target = values[:, :480], values[:, 480:]
    mean = history.mean(axis=1, keepdims=True)
    std = history.std(axis=1, keepdims=True)
    scale = np.where(std == 0, 1, std)
    return {
        "input_ids": torch.tensor((history - mean) / scale, device=device, dtype=dtype),
        "labels": torch.tensor((target - mean) / scale, device=device, dtype=dtype),
        "loss_masks": torch.ones((len(values), 1), device=device, dtype=dtype),
    }


def v6_single_patch_loss(model, batch):
    """Compute exactly the 96-point next-patch loss without shifted patches."""
    import torch
    hidden = model.model(input_ids=batch["input_ids"], use_cache=False,
                         return_dict=True).last_hidden_state
    try:
        head = model.lm_heads[model.output_token_len_map[96]]
    except (AttributeError, KeyError):
        heads = [(head, int(size)) for head, size in zip(
            model.lm_heads, model.config.output_token_lens) if int(size) == 96]
        if len(heads) != 1:
            raise ValueError("Timer V6 requires exactly one 96-point output head")
        head = heads[0][0]
    prediction = head(hidden[:, -1:, :])
    target = batch["labels"][:, None, :]
    return torch.nn.functional.mse_loss(prediction.float(), target.float(), reduction="none").mean(dim=-1).mean(dim=-1)


class TimerXLAdapter(FoundationModelAdapter):
    def load(self):
        if self._model is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM
            except ImportError as exc:
                raise RuntimeError("install the isolated timerxl dependency profile") from exc
            self._model = AutoModelForCausalLM.from_pretrained(
                self.spec.model_id,
                revision=self.spec.revision,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            ).to(_device(self.device))
        return self._model

    def training_loss(self, windows: np.ndarray, *, context_length: int, prediction_length: int):
        self.spec.require(ModelCapability.TRAINING_LOSS)
        values = as_finite_batch(windows, name="windows")
        if values.shape[1] != context_length + prediction_length or values.shape[1] > self.spec.max_context:
            raise ValueError("Timer-XL requires a valid selected C+H window within its context contract")
        model = self.load()
        parameter = next(model.parameters())
        if (context_length, prediction_length) == (480, 96):
            batch = native_v6_training_batch(windows, device=parameter.device, dtype=parameter.dtype)
            return v6_single_patch_loss(model, batch).mean()
        batch = native_training_batch(values, device=parameter.device, dtype=parameter.dtype,
                                      patch=model.config.input_token_len)
        return model(**batch, use_cache=False, return_dict=True).loss.float()

    def forecast(self, contexts: np.ndarray, *, prediction_length: int) -> np.ndarray:
        self.spec.require(ModelCapability.FORECAST)
        values = as_finite_batch(contexts, name="contexts")
        if prediction_length <= 0 or values.shape[1] > self.spec.max_context:
            raise ValueError("context exceeds the pinned Timer-XL contract")
        import torch
        model = self.load()
        target = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            # Official generation semantics: the checkpoint tail-truncates the
            # context to a patch multiple (512 -> 480) and normalizes the kept
            # history (generate default revin=True); output is deterministic.
            inputs = torch.as_tensor(values, dtype=next(model.parameters()).dtype, device=target)
            if values.shape[1] == 480 and prediction_length == 96:
                # V6/V7 needs exactly one native continuous-valued patch.
                # Replay the pinned TSGenerationMixin greedy loop's first
                # forward, without entering the incompatible modern HF token
                # sampler or creating a KV cache that cannot be reused here.
                if model.config.input_token_len != 96 or list(model.config.output_token_lens) != [96]:
                    raise ValueError('Timer single-patch inference requires the pinned 96-point head')
                prepared = model.prepare_inputs_for_generation(inputs,
                    attention_mask=torch.ones((len(inputs), 5), dtype=torch.long, device=target),
                    use_cache=False)
                output = model(**prepared, return_dict=True, max_output_length=96).logits
            else:
                output = model.generate(inputs, max_new_tokens=prediction_length)
        forecast = output.float().cpu().numpy()[:, -prediction_length:]
        # Numerics guard: a context whose kept tail collapses to a constant under
        # bf16 quantization makes generate's revin divide by a zero std and return
        # all-NaN rows (observed on boomlet_1676 validation, std ~7e-5 << bf16 ulp).
        # Only those rows are replaced, by the kept-tail mean in float32; every
        # finite official forecast stays bit-identical.
        broken = ~np.isfinite(forecast).all(axis=1)
        if broken.any():
            patch = model.config.input_token_len
            kept = values.shape[1] // patch * patch
            fallback = values[:, -kept:].mean(axis=1, keepdims=True)[broken]
            forecast[broken] = np.broadcast_to(fallback, (broken.sum(), prediction_length))
        return forecast

    def context_embeddings(self, contexts):
        """Mean final context-patch states under generate's tail truncation."""
        import torch
        values = as_finite_batch(contexts, name="contexts")
        model = self.load()
        parameter = next(model.parameters())
        patch = model.config.input_token_len
        kept = values.shape[1] // patch * patch
        model.eval()
        with torch.no_grad():
            x = torch.tensor(values[:, -kept:], device=parameter.device, dtype=parameter.dtype)
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            std = torch.where(std == 0, torch.ones_like(std), std)
            output = model.model(input_ids=(x - mean) / std, use_cache=False, return_dict=True)
            return output.last_hidden_state.mean(dim=1).float().cpu().numpy()


__all__ = ["TimerXLAdapter", "native_training_batch", "native_v6_training_batch",
           "v6_single_patch_loss"]
