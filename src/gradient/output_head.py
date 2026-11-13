"""Output-head gradient proxies for supported forecasting model families."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from .base import _FullModelProjector


class OutputHeadGradientProbe:
    """Compute output-head-only per-window gradients for cache preparation.

    Chronos-T5 and Timer-XL use closed-form head derivatives under no_grad.
    TimesFM 2.5 has nonlinear residual projection heads, so autograd is limited
    to those head parameters while all backbone parameters are frozen.
    """

    def __init__(self, probe, family: str):
        if family not in {"chronos", "timerxl", "timesfm"}:
            raise ValueError("unsupported output-head family: %s" % family)
        self.base = probe
        self.model = probe.model
        self.device = torch.device(probe.device)
        self.family = family
        base_config = probe.config
        self.config = SimpleNamespace(
            model_path=getattr(base_config, "model_path", family),
            context_length=int(base_config.context_length),
            prediction_length=int(base_config.prediction_length),
            projection_dimension=int(base_config.projection_dimension),
            projection_seed=int(base_config.projection_seed),
            layer="output_head",
            gradient_kind="output_head",
        )
        self.execution = getattr(probe, "execution", None)
        self.gradient_kind = "output_head"
        self._head_named = self._select_head_parameters()
        if not self._head_named:
            raise ValueError("no trainable output-head parameters found for %s" % family)
        self._head_parameters = [parameter for _, parameter in self._head_named]
        self.projector = _FullModelProjector(
            self._head_named, self.config.projection_dimension,
            self.config.projection_seed, self.device,
        )
        self.last_gradient_execution = {}

        if self.family == "timesfm":
            target_ids = {id(parameter) for parameter in self._head_parameters}
            for parameter in self.model.parameters():
                parameter.requires_grad_(id(parameter) in target_ids)
            self.model.eval()

    def _select_head_parameters(self):
        if self.family == "chronos":
            head = getattr(self.model, "lm_head", None)
            if not isinstance(head, torch.nn.Linear):
                raise ValueError("Chronos-T5 requires a linear lm_head")
            return [("lm_head." + name, parameter)
                    for name, parameter in head.named_parameters()
                    if parameter.requires_grad]
        if self.family == "timerxl":
            heads = getattr(self.model, "lm_heads", None)
            if heads is None or not len(heads):
                raise ValueError("Timer-XL checkpoint has no lm_heads")
            ids = {id(parameter) for head in heads for parameter in head.parameters()}
            return [(name, parameter) for name, parameter in self.model.named_parameters()
                    if id(parameter) in ids and parameter.requires_grad]
        names = ("output_projection_point.", "output_projection_quantiles.")
        return [(name, parameter) for name, parameter in self.model.named_parameters()
                if name.startswith(names) and parameter.requires_grad]

    def batch_gradient_sketch(self, windows, batch_size=None):
        values = np.asarray(windows, dtype=np.float32)
        expected = self.config.context_length + self.config.prediction_length
        if values.ndim != 2 or values.shape[1] != expected:
            raise ValueError("gradient windows have invalid geometry")
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("gradient windows must be non-empty and finite")
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive")

        if self.family == "chronos":
            losses, sketches = self._chronos_batch(values)
        elif self.family == "timerxl":
            losses, sketches = self._timerxl_batch(values)
        else:
            losses, sketches = self._timesfm_batch(values)
        loss_array = np.asarray(losses, dtype=np.float64).reshape(-1)
        sketch_array = np.asarray(sketches, dtype=np.float32)
        if sketch_array.shape != (len(values), self.config.projection_dimension):
            raise ValueError("output-head gradient sketch has invalid shape")
        if not np.isfinite(loss_array).all() or not np.isfinite(sketch_array).all():
            raise ValueError("output-head gradient result is not finite")
        self.last_gradient_execution = dict(
            batch_size=len(values) if self.family == "chronos" else 1,
            backend=("head-autograd" if self.family == "timesfm" else "analytic-head"),
            oom_retries=0,
        )
        return loss_array, sketch_array

    def _project(self, gradients):
        return self.projector.project(gradients).detach().float().cpu().numpy()

    def _chronos_batch(self, values):
        from rage.engines import per_sample_cross_entropy

        tensors = self.base.batch_tensors(values)
        holder = {}

        def capture(_module, args):
            if not args or not isinstance(args[0], torch.Tensor):
                raise ValueError("Chronos lm_head input was not captured")
            holder["hidden"] = args[0]

        hook = self.model.lm_head.register_forward_pre_hook(capture)
        try:
            with torch.no_grad():
                output = self.model(**self.base._logits_inputs(tensors), return_dict=True)
                logits = output.logits.float()
                hidden = holder.get("hidden")
                if hidden is None:
                    raise ValueError("Chronos lm_head did not receive a hidden-state tensor")
                hidden = hidden.detach().float()
                labels = tensors["labels"]
                losses = per_sample_cross_entropy(logits, labels)
                valid = labels.ne(-100)
                safe_labels = labels.masked_fill(~valid, 0)
                dlogits = logits.softmax(dim=-1)
                dlogits.scatter_add_(
                    -1, safe_labels.unsqueeze(-1),
                    -valid.unsqueeze(-1).to(dlogits.dtype),
                )
                dlogits.mul_(valid.unsqueeze(-1).to(dlogits.dtype))
                dlogits.div_(valid.sum(dim=1).clamp_min(1).view(-1, 1, 1))
                head = self.model.lm_head
                sketches = []
                for row in range(len(values)):
                    weight_gradient = torch.einsum(
                        "sv,sd->vd", dlogits[row], hidden[row],
                    )
                    gradients = [weight_gradient]
                    if head.bias is not None:
                        gradients.append(dlogits[row].sum(dim=0))
                    sketches.append(self._project(gradients))
        finally:
            hook.remove()
        return losses.detach().cpu().numpy(), sketches

    @staticmethod
    def _linear_head_gradient(head, hidden, residual, scale):
        doutput = residual.float() * scale
        weight = torch.einsum("bto,btd->od", doutput, hidden.float())
        gradients = [weight]
        if head.bias is not None:
            gradients.append(doutput.sum(dim=(0, 1)))
        return gradients

    def _timerxl_batch(self, values):
        from adapters.foundation.timerxl import (
            native_training_batch, native_v6_training_batch,
        )

        losses, sketches = [], []
        is_v6 = (self.config.context_length, self.config.prediction_length) == (480, 96)
        heads = self.model.lm_heads
        with torch.no_grad():
            for window in values:
                batch = (native_v6_training_batch(
                    window[None], device=self.device,
                    dtype=next(self.model.parameters()).dtype,
                ) if is_v6 else native_training_batch(
                    window[None], device=self.device,
                    dtype=next(self.model.parameters()).dtype,
                    patch=self.model.config.input_token_len,
                ))
                hidden = self.model.model(
                    input_ids=batch["input_ids"], use_cache=False, return_dict=True,
                ).last_hidden_state
                gradient_by_id = {}
                if is_v6:
                    output_length = 96
                    head = None
                    try:
                        head = heads[self.model.output_token_len_map[output_length]]
                    except (AttributeError, KeyError):
                        matches = [(item, int(length)) for item, length in zip(
                            heads, self.model.config.output_token_lens)
                            if int(length) == output_length]
                        if len(matches) != 1:
                            raise ValueError("Timer-XL requires exactly one 96-point output head")
                        head = matches[0][0]
                    prediction = head(hidden[:, -1:, :]).float()
                    target = batch["labels"][:, None, :].float()
                    residual = prediction - target
                    gradients = self._linear_head_gradient(
                        head, hidden[:, -1:, :], residual, 2.0 / output_length,
                    )
                    for parameter, gradient in zip(head.parameters(), gradients):
                        gradient_by_id[id(parameter)] = gradient
                    loss = residual.square().mean(dim=-1).mean(dim=-1)
                else:
                    patch = self.model.config.input_token_len
                    head_count = len(heads)
                    head_losses = []
                    for head, output_token_len in zip(
                        heads, self.model.config.output_token_lens,
                    ):
                        output_token_len = int(output_token_len)
                        prediction = head(hidden).float()
                        seq_len = prediction.shape[1] * patch
                        labels = batch["labels"][:, :seq_len - patch + output_token_len]
                        targets = labels.unfold(
                            dimension=-1, size=output_token_len, step=patch,
                        ).float()
                        if targets.shape != prediction.shape:
                            raise ValueError("Timer-XL head and shifted labels do not align")
                        residual = prediction - targets
                        scale = 2.0 / (
                            output_token_len * prediction.shape[1] * head_count
                        )
                        gradients = self._linear_head_gradient(
                            head, hidden, residual, scale,
                        )
                        for parameter, gradient in zip(head.parameters(), gradients):
                            gradient_by_id[id(parameter)] = gradient
                        head_losses.append(residual.square().mean(dim=-1).mean(dim=-1))
                    loss = torch.stack(head_losses).mean(dim=0)
                projected = self._project([
                    gradient_by_id[id(parameter)] for parameter in self._head_parameters
                ])
                losses.append(loss.detach().cpu().numpy())
                sketches.append(projected)
        return losses, sketches

    def _timesfm_batch(self, values):
        losses, sketches = [], []
        self.model.eval()
        for window in values:
            loss_vector = self.base.per_window_losses(window[None])
            gradients = torch.autograd.grad(
                loss_vector[0], self._head_parameters, allow_unused=False,
            )
            losses.append(float(loss_vector[0].detach().float().cpu()))
            sketches.append(self._project(gradients))
            del gradients, loss_vector
        return losses, sketches


__all__ = ["OutputHeadGradientProbe"]

