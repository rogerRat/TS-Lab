"""Middle-layer Sundial adapter using the revision-pinned official checkpoint.

Inference is the checkpoint's official sampling path (unchanged).  Sundial
ships and runs in float32 (its patch embedding concatenates a float32 mask,
so bf16 casts break the checkpoint code); the production pipeline keeps this
native precision instead of its bf16 student default.  Training
has no released official recipe, so RAGE assembles the checkpoint-native
TimeFlow components (``flow_loss.py`` ships inside the pinned snapshot) on
selected 576-point windows: shifted per-patch flow targets truncated by
``mask_y`` to each patch's in-window future, raw values (revin off, matching
official inference).  Gradient/probe losses use content-seeded noise so one
window always maps to one deterministic loss; training exposures draw fresh
noise from the trainer's global RNG, as native flow matching requires.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .base import FoundationModelAdapter
from .components import as_finite_batch, median_forecast
from .spec import ModelCapability


def flow_supervision(windows, *, patch, target_length):
    """Shifted next-patch flow targets on each selected 576-point window.

    Mirrors the checkpoint forward's labels contract: the flow target at patch
    j starts one patch ahead (``labels = w[P:]`` unfolded with step P).  A
    576-point window cannot fill the native 720-channel target, so targets are
    zero-padded and ``mask_y`` keeps only each patch's in-window future, the
    checkpoint's own mechanism for unavailable target channels.  Every patch
    of input ``w[:-P]`` is supervised.
    """
    values = as_finite_batch(windows, name="windows")
    length = values.shape[1]
    if length % patch or length < 2 * patch:
        raise ValueError("Sundial training windows must provide at least two full patches")
    patches = length // patch - 1
    targets = np.zeros((len(values), patches, target_length), dtype=np.float32)
    mask_y = np.zeros_like(targets)
    for j in range(patches):
        start = (j + 1) * patch
        available = min(target_length, length - start)
        targets[:, j, :available] = values[:, start:start + available]
        mask_y[:, j, :available] = 1.0
    return values[:, :-patch], targets, mask_y


def flow_losses(model, windows, *, deterministic):
    """Per-window TimeFlow losses from checkpoint-native components.

    Returns one scalar per window (mean over its supervised patches and over
    ``diffusion_batch_mul`` independent noise draws), exactly the checkpoint's
    masked TimeFlow reduction restricted to one window.  ``deterministic``
    seeds each window's noise/timestep draws from a content hash so gradient
    caches see a fixed loss function; training passes False so every exposure
    resamples noise from the global RNG.
    """
    import torch
    config = model.config
    patch, target_length = config.input_token_len, config.output_token_lens[-1]
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    input_ids, targets, mask_y = flow_supervision(windows, patch=patch,
                                                  target_length=target_length)
    count, patches = input_ids.shape[0], input_ids.shape[1] // patch
    inputs = torch.as_tensor(input_ids, device=device, dtype=dtype)
    targets = torch.as_tensor(targets, device=device, dtype=dtype)
    mask_y = torch.as_tensor(mask_y, device=device, dtype=dtype)
    hidden = model.model(input_ids=inputs, use_cache=False,
                         return_dict=True).last_hidden_state
    multiplier = config.diffusion_batch_mul
    weights = 1.0 / torch.arange(1, target_length + 1, dtype=torch.float32,
                                 device=device)
    losses = torch.zeros(count, device=device, dtype=torch.float32)
    for repeat in range(multiplier):
        if deterministic:
            noise = torch.empty(count, patches, target_length, device=device,
                                dtype=torch.float32)
            timestep = torch.empty(count, patches, device=device, dtype=torch.float32)
            for row in range(count):
                digest = hashlib.blake2b(
                    np.ascontiguousarray(windows[row], dtype=np.float32).tobytes(),
                    digest_size=8, key=b"rage-sundial-flow")
                generator = torch.Generator(device=device)
                generator.manual_seed((int.from_bytes(digest.digest(), "little")
                                       * multiplier + repeat) % (2**63 - 1))
                noise[row] = torch.randn(patches, target_length, generator=generator,
                                         device=device)
                timestep[row] = torch.rand(patches, generator=generator, device=device)
        else:
            noise = torch.randn(count, patches, target_length, device=device,
                                dtype=torch.float32)
            timestep = torch.rand(count, patches, device=device, dtype=torch.float32)
        noised = timestep[..., None] * targets + (1 - timestep[..., None]) * noise
        velocity = model.flow_loss.net(
            noised.reshape(count * patches, target_length).to(dtype),
            (timestep * 1000).reshape(count * patches).to(dtype),
            hidden.reshape(count * patches, -1))
        error = ((velocity.reshape(count, patches, target_length).float()
                  - targets) ** 2 * weights * mask_y).sum(dim=-1)
        losses = losses + error.mean(dim=1) / multiplier
    return losses


class SundialAdapter(FoundationModelAdapter):
    def load(self):
        if self._model is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM
            except ImportError as exc:
                raise RuntimeError("install the isolated sundial dependency profile") from exc
            target = self.device
            if target == "auto":
                target = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = AutoModelForCausalLM.from_pretrained(
                self.spec.model_id,
                revision=self.spec.revision,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            ).to(target)
        return self._model

    def forecast(self, contexts: np.ndarray, *, prediction_length: int) -> np.ndarray:
        self.spec.require(ModelCapability.FORECAST)
        values = as_finite_batch(contexts, name="contexts")
        if values.shape[1] > self.spec.max_context:
            raise ValueError("context exceeds the pinned Sundial contract")
        import torch
        model = self.load()
        target = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            samples = model.generate(
                torch.as_tensor(values, dtype=next(model.parameters()).dtype, device=target),
                max_new_tokens=prediction_length,
                num_samples=20,
            )
        return median_forecast(samples.float().cpu().numpy(), prediction_length=prediction_length)

    def training_loss(self, windows: np.ndarray, *, context_length: int, prediction_length: int):
        self.spec.require(ModelCapability.TRAINING_LOSS)
        values = as_finite_batch(windows, name="windows")
        if values.shape[1] != context_length + prediction_length or values.shape[1] > self.spec.max_context:
            raise ValueError("Sundial requires a valid selected C+H window within its context contract")
        return flow_losses(self.load(), values, deterministic=False).mean()

    def context_embeddings(self, contexts):
        """Mean final context-patch states on raw values, as at inference."""
        import torch
        values = as_finite_batch(contexts, name="contexts")
        if values.shape[1] > self.spec.max_context:
            raise ValueError("context exceeds the pinned Sundial contract")
        model = self.load()
        parameter = next(model.parameters())
        model.eval()
        with torch.no_grad():
            x = torch.as_tensor(values, device=parameter.device, dtype=parameter.dtype)
            output = model.model(input_ids=x, use_cache=False, return_dict=True)
            return output.last_hidden_state.mean(dim=1).float().cpu().numpy()


__all__ = ["SundialAdapter", "flow_losses", "flow_supervision"]
