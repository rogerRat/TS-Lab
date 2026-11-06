"""ChronosTiny gradient engine with last-layer and arbitrary-layer support."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .exceptions import ModelNotFoundError
from .projection import ParameterHashProjector, TensorSketchProjector
from .base import _FullModelProjector, IndependentGradientWorkers
from gradient.count_sketch import add_count_sketch
from rage.execution import SelectionExecution


DEFAULT_CHRONOS_T5_SMALL = (
    "cache/pretrained_models/amazon/chronos-t5-small"
)
DEFAULT_CHRONOS_T5_BASE = "cache/pretrained_models/amazon/chronos-t5-base"


@dataclass(frozen=True)
class ChronosT5Config:
    """Official Chronos-T5 model, protocol, and shared projection settings."""

    model_path: str = DEFAULT_CHRONOS_T5_BASE
    context_length: int = 512
    prediction_length: int = 64
    n_tokens: int = 4096  # official Chronos-T5 vocabulary size
    projection_dimension: int = 4096
    projection_seed: int = 42
    gradient_batch_size: int = 32
    eval_batch_size: int = 128
    seed: int = 42
    torch_dtype: str = "bfloat16"
    # vmap per-sample gradients (B6 batching): ~5x throughput at batch 16.
    # Falls back to the sequential loop on OOM or when disabled.
    use_vmap_gradients: bool = True
    vmap_batch_size: int = 0
    inference_batch_size: int = 0
    selection_memory_fraction: float = 0.7
    selection_cpu_threads: int = 4
    selection_generation: str = "cuda_graph"

    def __post_init__(self) -> None:
        for name in ("context_length", "prediction_length",
                     "projection_dimension", "gradient_batch_size",
                     "eval_batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("%s must be a positive integer" % name)
        if self.torch_dtype not in ("float32", "bfloat16"):
            raise ValueError("torch_dtype must be float32 or bfloat16")

    @property
    def window_length(self) -> int:
        return self.context_length + self.prediction_length


class ChronosT5Engine(IndependentGradientWorkers):
    worker_student = "chronos-base"
    """Per-window full-parameter gradient sketches and official-style training."""

    def __init__(self, config: ChronosT5Config | None = None, *,
                 device: str = "auto") -> None:
        config = config or ChronosT5Config()
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        selected_device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto" else device
        )
        from transformers import AutoModelForSeq2SeqLM

        model_path = Path(config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                "Chronos-T5 checkpoint not found at %s; download it first "
                "(e.g. via hf-mirror)" % model_path)
        dtype = torch.float32 if config.torch_dtype == "float32" else torch.bfloat16
        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(model_path), local_files_only=True, dtype=dtype,
        )
        model.to(selected_device)
        model.eval()
        tokenizer = self._load_tokenizer(model, config)
        self.config = config
        self.checkpoint_path = str(model_path)
        self.device = selected_device
        self.model = model
        self.execution = SelectionExecution(gradient_batch=config.vmap_batch_size,
                                            inference_batch=config.inference_batch_size,
                                            memory_fraction=config.selection_memory_fraction,
                                            cpu_threads=config.selection_cpu_threads,
                                            generation=config.selection_generation)
        self.tokenizer = tokenizer
        self.n_tokens = int(model.config.vocab_size)
        if self.n_tokens != config.n_tokens:
            raise ValueError(
                "model vocabulary %d does not match configured n_tokens %d"
                % (self.n_tokens, config.n_tokens))
        self.parameter_count = sum(p.numel() for p in model.parameters())
        self._named_trainable = [
            (name, parameter) for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        self.projector = _FullModelProjector(
            self._named_trainable, config.projection_dimension,
            config.projection_seed, selected_device,
        )
        self._parameters = [parameter for _n, parameter in self._named_trainable]

    @staticmethod
    def _load_tokenizer(model: nn.Module, config: ChronosT5Config) -> Any:
        from chronos import ChronosConfig

        raw = getattr(model.config, "chronos_config", None)
        if raw is None:
            raise ValueError("checkpoint has no chronos_config; not an official Chronos model")
        chronos_config = ChronosConfig(**{
            **raw,
            "context_length": config.context_length,
            "prediction_length": config.prediction_length,
        })
        return chronos_config.create_tokenizer()

    def batch_tensors(self, windows: np.ndarray | torch.Tensor) -> dict[str, torch.Tensor]:
        values = torch.as_tensor(windows, dtype=torch.float32)
        if values.ndim != 2 or values.shape[1] != self.config.window_length:
            raise ValueError("windows must have shape [N, context+prediction]")
        if not torch.isfinite(values).all():
            raise ValueError("windows contain non-finite values")
        cpu_values = values.detach().cpu()
        contexts = cpu_values[:, :self.config.context_length]
        futures = cpu_values[:, self.config.context_length:]
        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(contexts)
        labels, labels_mask = self.tokenizer.label_input_transform(futures, scale)
        labels = labels.masked_fill(labels_mask == 0, -100)
        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "labels": labels.to(self.device),
        }

    def prepare_training_tensors(
        self, windows: np.ndarray, *, max_bytes: int = 64 * 1024 * 1024,
    ) -> dict[str, torch.Tensor] | None:
        """Tokenize a fixed training set once, bounded to 64 MiB on device.

        Chronos mean scaling and binning operate independently per row and
        do not depend on model weights. Oversized sets retain on-demand
        tokenization; no cross-call cache can become stale after data edits.
        """
        rows = len(windows)
        estimated = rows * ((self.config.context_length + 1) * 9
                            + (self.config.prediction_length + 1) * 8)
        if not rows or estimated > max_bytes:
            return None
        result = None
        for begin in range(0, rows, 256):
            chunk = self.batch_tensors(windows[begin:begin + 256])
            if result is None:
                result = {key: tensor.new_empty((rows, *tensor.shape[1:]))
                          for key, tensor in chunk.items()}
            for key, tensor in chunk.items():
                result[key][begin:begin + len(tensor)].copy_(tensor)
        return result

    def window_loss(self, window: np.ndarray) -> float:
        tensors = self.batch_tensors(np.asarray(window, dtype=np.float32).reshape(1, -1))
        with torch.no_grad():
            output = self.model(**self._logits_inputs(tensors), return_dict=True)
            loss = per_sample_cross_entropy(output.logits, tensors["labels"])
        return float(loss[0].detach().cpu())

    def gradient_sketch(self, window: np.ndarray) -> tuple[float, np.ndarray]:
        """One forward+backward for one window; returns (loss, projected grad)."""
        tensors = self.batch_tensors(np.asarray(window, dtype=np.float32).reshape(1, -1))
        output = self.model(**self._logits_inputs(tensors), return_dict=True)
        loss = per_sample_cross_entropy(output.logits, tensors["labels"])[0]
        gradients = torch.autograd.grad(loss, self._parameters, allow_unused=True)
        projected = []
        for gradient, parameter in zip(gradients, self._parameters):
            if gradient is None:
                gradient = torch.zeros_like(parameter)
            projected.append(gradient)
        sketch = self.projector.project(projected)
        return float(loss.detach().cpu()), sketch.detach()

    def _logits_inputs(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Keep teacher forcing while avoiding T5's unused second CE loss."""
        return {
            "input_ids": tensors["input_ids"],
            "attention_mask": tensors["attention_mask"],
            "decoder_input_ids": self.model.prepare_decoder_input_ids_from_labels(
                labels=tensors["labels"]),
            "use_cache": False,
        }

    def batch_gradient_sketch(self, windows: np.ndarray,
                              batch_size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(windows, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.config.window_length:
            raise ValueError("gradient windows have invalid geometry")
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("gradient windows must be non-empty and finite")
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive")
        size = min(batch_size or self.execution.batch_size(self, "gradient", fallback=8, count=len(values)),
                   len(values))
        # BF16 vmap changes forward geometry and fails native-window agreement.
        # Keep FP32 vmap available for explicit historical use, but BF16 assets
        # use independent native forwards, optionally in persistent GPU workers.
        bf16 = next(self.model.parameters()).dtype == torch.bfloat16
        if (bf16 and self.device.type == "cuda" and size > 1
                and self.execution.gradient_workers > 1
                and self.parameter_count == 201374976
                and self.config.projection_dimension == 4096 and self.config.projection_seed == 42):
            return self._process_gradients(values, min(size, 8))
        if (self.config.use_vmap_gradients and self.device.type == "cuda"
                and not bf16 and size > 1 and len(values) >= 4):
            while getattr(self, "_gradient_batch_limit", 128) > 1:
                try:
                    return self._vmap_batch_gradient_sketch(values, batch_size=size)
                except torch.OutOfMemoryError:
                    self._gradient_batch_limit = max(1, self._last_gradient_batch // 2)
                    print(f"[engine] reducing shared gradient batch to {self._gradient_batch_limit}",
                          flush=True)
                # Retry after the exception frame and its tensors are released.
                torch.cuda.empty_cache()
        size = 1
        losses: list[np.ndarray] = []
        sketches: list[np.ndarray] = []
        for start in range(0, len(values), size):
            for row in values[start:start + size]:
                loss, sketch = self.gradient_sketch(row)
                losses.append(np.asarray([loss], dtype=np.float32))
                sketches.append(sketch.cpu().numpy().reshape(1, -1))
        loss_array = np.concatenate(losses).astype(np.float64, copy=False)
        sketch_array = np.concatenate(sketches).astype(np.float32, copy=False)
        if sketch_array.shape != (len(values), self.config.projection_dimension):
            raise ValueError("projected gradient shape is invalid")
        if not np.isfinite(loss_array).all() or not np.isfinite(sketch_array).all():
            raise ValueError("projected gradient output is not finite")
        self.last_gradient_execution = dict(batch_size=1, backend="sequential", oom_retries=0)
        return loss_array, sketch_array

    def _vmap_batch_gradient_sketch(self, values: np.ndarray,
                                    batch_size=None) -> tuple[np.ndarray, np.ndarray]:
        """Per-sample gradients for a window block via ``torch.func`` vmap.

        Semantics identical to the sequential loop (one forward+backward per
        window, full-parameter gradient under the shared hash projection);
        T5's tied weights are handled by replacing every tied name with the
        same detached tensor and summing the per-name gradients.  transformers
        materialises the tied ``lm_head.weight`` as its own parameter during
        the first call, so hashers are looked up BY NAME (falling back to the
        alias group) and the tied Parameter objects are restored afterwards
        to keep the trainable set stable for later fine-tuning.  The vmap
        scaffolding (detached parameter views, alias groups, hasher lookup,
        grad_fn) is cached across calls; the views share storage with the
        live parameters, and the cache is rebuilt whenever the parameter
        object identities change.
        """
        from torch.func import functional_call, vmap, grad_and_value

        model = self.model
        dedup = [(name, parameter) for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
        cache = getattr(self, "_vmap_cache", None)
        if cache is not None and all(a is b for a, b in zip(cache["objects"],
                                                            [p for _n, p in dedup])):
            params, buffers, alias, lookup, grad_fn, saved_tied, \
                input_embeddings, output_embeddings, saved_input_weight = (
                    cache["params"], cache["buffers"], cache["alias"],
                    cache["lookup"], cache["grad_fn"], cache["saved_tied"],
                    cache["input_embeddings"], cache["output_embeddings"],
                    cache["saved_input_weight"])
        else:
            named_all = list(model.named_parameters(remove_duplicate=False))
            alias = [[other for other, q in named_all if q is parameter]
                     for _name, parameter in dedup]
            hasher_by_name = dict(zip(self.projector.names,
                                      self.projector.hashers))
            lookup: list[tuple[torch.Tensor, torch.Tensor]] = []
            for index, (name, _parameter) in enumerate(dedup):
                hasher = hasher_by_name.get(name)
                if hasher is None:  # materialised tied alias: hash with its group
                    hasher = next((hasher_by_name[other]
                                   for other in alias[index]
                                   if other in hasher_by_name), None)
                if hasher is None:
                    raise ValueError("no hasher for parameter %s" % name)
                lookup.append((hasher.hashes.to(self.device), hasher.signs.to(self.device)))
            params = {name: parameter.detach() for name, parameter in named_all}
            buffers = dict(model.named_buffers())

            def loss_one(p, b, ids, am, labels):
                output = functional_call(
                    model, (p, b), (),
                    kwargs={"input_ids": ids, "attention_mask": am,
                            "decoder_input_ids": model.prepare_decoder_input_ids_from_labels(
                                labels=labels), "return_dict": True, "use_cache": False},
                    tie_weights=False,
                )
                return per_sample_cross_entropy(output.logits, labels).squeeze(0)

            grad_fn = vmap(grad_and_value(loss_one), in_dims=(None, None, 0, 0, 0))
            output_embeddings = model.get_output_embeddings() if hasattr(
                model, "get_output_embeddings") else None
            input_embeddings = model.get_input_embeddings() if hasattr(
                model, "get_input_embeddings") else None
            saved_input_weight = (input_embeddings.weight
                                  if input_embeddings is not None else None)
            saved_tied = (isinstance(saved_input_weight, torch.nn.Parameter)
                          and output_embeddings is not None
                          and output_embeddings.weight is saved_input_weight)
            self._vmap_cache = {
                "objects": [p for _n, p in dedup], "params": params,
                "buffers": buffers, "alias": alias, "lookup": lookup,
                "grad_fn": grad_fn, "saved_tied": saved_tied,
                "input_embeddings": input_embeddings,
                "output_embeddings": output_embeddings,
                "saved_input_weight": saved_input_weight,
            }

        batch = batch_size or self.execution.batch_size(self, "gradient", fallback=16, count=len(values))
        batch = min(batch, getattr(self, "_gradient_batch_limit", batch))
        self._last_gradient_batch = batch
        dimension = self.config.projection_dimension
        losses = np.empty(len(values), dtype=np.float64)
        sketches = np.empty((len(values), dimension), dtype=np.float32)
        try:
            for begin in range(0, len(values), batch):
                stop = min(len(values), begin + batch)
                tensors = self.batch_tensors(values[begin:stop])
                grads, loss_values = grad_fn(
                    params, buffers,
                    tensors["input_ids"].unsqueeze(1),
                    tensors["attention_mask"].unsqueeze(1),
                    tensors["labels"].unsqueeze(1),
                )
                rows = stop - begin
                sketch = torch.zeros(rows, dimension, dtype=torch.float32,
                                     device=self.device)
                for index in range(len(lookup)):
                    total = None
                    for other in alias[index]:
                        piece = grads[other]
                        total = piece if total is None else total + piece
                    hashes, signs = lookup[index]
                    add_count_sketch(sketch, total.reshape(rows, -1), hashes, signs)
                losses[begin:stop] = loss_values.detach().cpu().numpy()
                sketches[begin:stop] = sketch.detach().cpu().numpy()
                del grads, sketch
        finally:
            if saved_tied:
                # both modules originally shared one Parameter object; put it
                # back so the trainable parameter set is exactly as before
                input_embeddings.weight = saved_input_weight
                output_embeddings.weight = saved_input_weight
        if not np.isfinite(losses).all() or not np.isfinite(sketches).all():
            raise ValueError("vmap gradient output is not finite")
        return losses, sketches

DEFAULT_CHRONOS_TINY_MODEL = (
    "cache/pretrained_models/google/t5-efficient-tiny/"
    "3441d7e8bf3f89841f366d39452b95200416e4a9"
)
DEFAULT_LAYER = "lm_head.weight"
DEFAULT_WARMUP_MODEL_ROOT = "outputs/models/warmup/chronos-tiny-from-scratch"
DEFAULT_WARMUP_MODEL_COUNT = 4
DEFAULT_WARMUP_SEEDS: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChronosTinyConfig:
    """Model, objective, projection, and warm-up protocol."""

    model_path: str = DEFAULT_CHRONOS_TINY_MODEL
    warmup_model_root: str = DEFAULT_WARMUP_MODEL_ROOT
    warmup_model_count: int = DEFAULT_WARMUP_MODEL_COUNT
    warmup_seeds: tuple[str | int, ...] | None = None
    require_warmup_models: bool = True
    initialization_mode: str = "pretrained"
    d_model: int = 256
    d_kv: int = 64
    d_ff: int = 1024
    num_layers: int = 4
    num_decoder_layers: int = 4
    num_heads: int = 4
    dropout_rate: float = 0.1
    tie_embeddings: bool = False
    context_length: int = 512
    prediction_length: int = 64
    n_tokens: int = 4096
    n_special_tokens: int = 2
    pad_token_id: int = 0
    eos_token_id: int = 1
    tokenizer_low_limit: float = -15.0
    tokenizer_high_limit: float = 15.0
    layer: str = DEFAULT_LAYER
    projection_dimension: int = 4096
    projection_repeats: int = 1
    projection_seed: int = 42
    token_chunk_size: int = 8
    gradient_batch_size: int = 8
    warmup_steps: int = 10_000
    warmup_batch_size: int = 32
    warmup_learning_rates: tuple[float, ...] = (1.0e-4, 3.0e-4, 1.0e-3)
    seed: int = 42

    def __post_init__(self) -> None:
        for name in (
            "context_length", "prediction_length", "n_tokens", "n_special_tokens",
            "projection_dimension", "projection_repeats", "token_chunk_size",
            "gradient_batch_size", "warmup_steps", "warmup_batch_size", "d_model",
            "d_kv", "d_ff", "num_layers", "num_decoder_layers", "num_heads",
            "warmup_model_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("%s must be a positive integer" % name)
        if not self.warmup_learning_rates or any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in self.warmup_learning_rates
        ):
            raise ValueError("warmup_learning_rates must be positive and finite")
        if not self.layer:
            raise ValueError("layer must be non-empty")
        if self.initialization_mode not in {"pretrained", "from_scratch"}:
            raise ValueError("initialization_mode must be pretrained or from_scratch")
        if not math.isfinite(float(self.dropout_rate)) or not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout_rate must be finite and in [0, 1)")
        if not isinstance(self.tie_embeddings, bool):
            raise TypeError("tie_embeddings must be bool")
        if not isinstance(self.require_warmup_models, bool):
            raise TypeError("require_warmup_models must be bool")
        if self.warmup_seeds is not None and any(
            isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value)
            for value in self.warmup_seeds
        ):
            raise ValueError("warmup_seeds must contain non-empty string or integer labels")

    @property
    def window_length(self) -> int:
        return self.context_length + self.prediction_length

    def as_mapping(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "warmup_model_root": self.warmup_model_root,
            "warmup_model_count": self.warmup_model_count,
            "warmup_seeds": None if self.warmup_seeds is None else [str(value) for value in self.warmup_seeds],
            "require_warmup_models": self.require_warmup_models,
            "initialization_mode": self.initialization_mode,
            "architecture": {
                "d_model": self.d_model,
                "d_kv": self.d_kv,
                "d_ff": self.d_ff,
                "num_layers": self.num_layers,
                "num_decoder_layers": self.num_decoder_layers,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
                "tie_embeddings": self.tie_embeddings,
            },
            "context_length": self.context_length,
            "prediction_length": self.prediction_length,
            "n_tokens": self.n_tokens,
            "n_special_tokens": self.n_special_tokens,
            "layer": self.layer,
            "projection_dimension": self.projection_dimension,
            "projection_repeats": self.projection_repeats,
            "projection_seed": self.projection_seed,
            "token_chunk_size": self.token_chunk_size,
            "gradient_batch_size": self.gradient_batch_size,
            "warmup_steps": self.warmup_steps,
            "warmup_batch_size": self.warmup_batch_size,
            "warmup_learning_rates": list(self.warmup_learning_rates),
            "seed": self.seed,
            "objective": "per-example mean cross-entropy over valid decoder tokens",
            "default_layer_rationale": "final lm_head.weight is the standard compact last-layer gradient signal",
        }


def _model_output_logits(output: Any) -> torch.Tensor:
    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, (tuple, list)) and output:
        logits = output[0]
    if not isinstance(logits, torch.Tensor):
        raise ValueError("model output does not expose logits")
    return logits


def per_sample_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels have incompatible sequence shapes")
    valid = labels.ne(-100)
    safe_labels = labels.masked_fill(~valid, 0)
    token_losses = torch.nn.functional.cross_entropy(
        logits.float().transpose(1, 2), safe_labels, reduction="none",
    )
    return token_losses.masked_fill(~valid, 0.0).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def select_parameter(model: nn.Module, layer: str = DEFAULT_LAYER) -> tuple[str, nn.Parameter]:
    """Select one trainable parameter by exact name, alias, or re: regex."""
    requested = {"default": DEFAULT_LAYER, "last": DEFAULT_LAYER, "lm_head": DEFAULT_LAYER}.get(layer, layer)
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    # T5 ties lm_head.weight to shared.weight and named_parameters() removes
    # duplicate objects. Keep the semantic layer name so the lm_head fast path
    # still applies to the actual output module.
    if requested == DEFAULT_LAYER:
        head = getattr(model, "lm_head", None)
        parameter = getattr(head, "weight", None)
        if isinstance(parameter, nn.Parameter) and parameter.requires_grad:
            return DEFAULT_LAYER, parameter
    exact = [(name, parameter) for name, parameter in named if name == requested]
    if exact:
        return exact[0]
    if requested.startswith("re:"):
        import re
        matches = [(name, parameter) for name, parameter in named if re.fullmatch(requested[3:], name)]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError("layer regex matched no trainable parameter: %s" % layer)
        raise ValueError("layer regex matched multiple parameters: %s" % layer)
    suffix = [(name, parameter) for name, parameter in named if name.endswith("." + requested)]
    if len(suffix) == 1:
        return suffix[0]
    if not suffix:
        available = ", ".join(name for name, _parameter in named)
        raise ValueError("trainable layer not found: %s; available: %s" % (layer, available))
    raise ValueError("layer is ambiguous: %s" % layer)


def _module_for_parameter(model: nn.Module, parameter_name: str) -> nn.Module | None:
    if "." not in parameter_name:
        return None
    module_name = parameter_name.rsplit(".", 1)[0]
    return dict(model.named_modules()).get(module_name)


class ChronosTinyGradientEngine:
    """Compute projected per-example gradients for a fixed ChronosTiny model."""

    def __init__(
        self,
        config: ChronosTinyConfig | None = None,
        *,
        device: str = "auto",
        model: nn.Module | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        config = config or ChronosTinyConfig()
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        selected_device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto" else device
        )
        if model is None:
            models, tokenizer, model_seeds = self._load_chronos(config)
        elif tokenizer is None:
            raise ValueError("a tokenizer is required when injecting a model")
        else:
            models, model_seeds = [model], ("injected",)
        for loaded_model in models:
            loaded_model.to(device=selected_device, dtype=torch.bfloat16)
            loaded_model.eval()
        self.config = config
        self.device = selected_device
        self.models = tuple(models)
        self.model_seeds = tuple(model_seeds)
        self.model = self.models[0]
        self.tokenizer = tokenizer
        self.layer_name, self.parameter = select_parameter(self.model, config.layer)
        self.parameters = tuple(select_parameter(loaded_model, config.layer)[1] for loaded_model in self.models)
        self.layer_modules = tuple(
            _module_for_parameter(loaded_model, config.layer) for loaded_model in self.models
        )
        self.layer_module = self.layer_modules[0]
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self.parameter_size = int(self.parameter.numel())
        if any(int(parameter.numel()) != self.parameter_size for parameter in self.parameters):
            raise ValueError("selected layer has inconsistent sizes across warmup models")
        self.fast_lm_head = self._can_use_lm_head_fast_path()
        if self.fast_lm_head:
            self.tensor_projector = TensorSketchProjector(
                int(self.parameter.shape[0]), int(self.parameter.shape[1]),
                projection_dimension=config.projection_dimension,
                repeats=config.projection_repeats,
                seed=config.projection_seed,
                device=selected_device,
                token_chunk_size=config.token_chunk_size,
            )
        else:
            self.parameter_projector = ParameterHashProjector(
                self.parameter_size,
                projection_dimension=config.projection_dimension,
                seed=config.projection_seed,
                device=selected_device,
            )

    @staticmethod
    def _warmup_model_paths(config: ChronosTinyConfig) -> list[tuple[str, Path]]:
        root = Path(config.warmup_model_root).expanduser().resolve()
        if config.warmup_seeds is None:
            candidates = sorted(
                (item.name[5:], item / "model")
                for item in root.iterdir()
                if item.is_dir() and item.name.startswith("seed-")
            ) if root.is_dir() else []
            candidates.sort(
                key=lambda item: (0, int(item[0])) if item[0].isdigit() else (1, item[0])
            )
        else:
            candidates = [
                (str(seed), root / ("seed-%s" % seed) / "model")
                for seed in config.warmup_seeds
            ]
        existing = [(label, path) for label, path in candidates if path.is_dir()]
        if len(existing) < config.warmup_model_count:
            missing = [str(path) for _label, path in candidates if not path.is_dir()]
            raise ModelNotFoundError(
                "ChronosTiny gradient analysis requires usable warmup models; "
                "found %d checkpoint directories but require at least %d under %s; "
                "missing model directories: %s"
                % (len(existing), config.warmup_model_count, root, ", ".join(missing))
            )
        return existing

    @staticmethod
    def _assert_warmup_models_available(config: ChronosTinyConfig) -> list[tuple[str, Path]]:
        """Validate all required local warm-up checkpoints before analysis."""
        paths = ChronosTinyGradientEngine._warmup_model_paths(config)
        from transformers import AutoConfig, AutoModelForSeq2SeqLM

        errors: list[str] = []
        valid: list[tuple[str, Path]] = []
        for seed, path in paths:
            try:
                AutoConfig.from_pretrained(str(path), local_files_only=True)
                weight_files = tuple(path.glob("*.safetensors")) + tuple(path.glob("*.bin"))
                if not weight_files or any(item.stat().st_size <= 0 for item in weight_files):
                    raise ValueError("no non-empty model weight file")
                model = AutoModelForSeq2SeqLM.from_pretrained(str(path), local_files_only=True)
                parameter_count = sum(parameter.numel() for parameter in model.parameters())
                if parameter_count != 9_443_072:
                    raise ValueError("parameter count is %d, expected 9443072" % parameter_count)
                del model
                valid.append((seed, path))
            except Exception as exc:
                errors.append("seed-%s (%s): %s" % (seed, path, exc))
        if len(valid) < config.warmup_model_count:
            raise ModelNotFoundError(
                "ChronosTiny gradient analysis requires usable warmup models; "
                "only %d usable checkpoints found, require at least %d; %s"
                % (len(valid), config.warmup_model_count, "; ".join(errors))
            )
        return valid[:config.warmup_model_count]

    @staticmethod
    def _load_chronos(config: ChronosTinyConfig) -> tuple[list[nn.Module], Any, tuple[str, ...]]:
        from chronos import ChronosConfig
        from transformers import AutoModelForSeq2SeqLM, T5Config

        tokenizer_config = ChronosConfig(
            tokenizer_class="MeanScaleUniformBins",
            tokenizer_kwargs={
                "low_limit": config.tokenizer_low_limit,
                "high_limit": config.tokenizer_high_limit,
            },
            context_length=config.context_length,
            prediction_length=config.prediction_length,
            n_tokens=config.n_tokens,
            n_special_tokens=config.n_special_tokens,
            pad_token_id=config.pad_token_id,
            eos_token_id=config.eos_token_id,
            use_eos_token=True,
            model_type="seq2seq",
            num_samples=20,
            temperature=1.0,
            top_k=50,
            top_p=1.0,
        )
        path = Path(config.model_path)
        if config.require_warmup_models:
            warmup_paths = ChronosTinyGradientEngine._assert_warmup_models_available(config)
            models = [
                AutoModelForSeq2SeqLM.from_pretrained(str(model_path), local_files_only=True)
                for _label, model_path in warmup_paths
            ]
            model_seeds = tuple(label for label, _model_path in warmup_paths)
            for loaded_model in models:
                loaded_model.resize_token_embeddings(config.n_tokens)
                loaded_model.config.pad_token_id = config.pad_token_id
                loaded_model.config.eos_token_id = config.eos_token_id
                loaded_model.config.chronos_config = dict(tokenizer_config.__dict__)
            return models, tokenizer_config.create_tokenizer(), model_seeds
        elif config.initialization_mode == "from_scratch":
            # The Tiny architecture is part of the checked-in RAGE protocol. It
            # is deliberately constructed locally so warm-up never downloads or
            # accidentally consumes pretrained weights.
            hf_config = T5Config(
                vocab_size=config.n_tokens,
                d_model=config.d_model,
                d_kv=config.d_kv,
                d_ff=config.d_ff,
                num_layers=config.num_layers,
                num_decoder_layers=config.num_decoder_layers,
                num_heads=config.num_heads,
                dropout_rate=config.dropout_rate,
                tie_word_embeddings=config.tie_embeddings,
                pad_token_id=config.pad_token_id,
                eos_token_id=config.eos_token_id,
                decoder_start_token_id=config.pad_token_id,
                feed_forward_proj="relu",
            )
            model = AutoModelForSeq2SeqLM.from_config(hf_config)
        else:
            if not path.exists():
                raise FileNotFoundError("ChronosTiny model path does not exist: %s" % path)
            model = AutoModelForSeq2SeqLM.from_pretrained(str(path), local_files_only=True)
        model.resize_token_embeddings(config.n_tokens)
        model.config.pad_token_id = config.pad_token_id
        model.config.eos_token_id = config.eos_token_id
        model.config.chronos_config = dict(tokenizer_config.__dict__)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if not 8_000_000 <= parameter_count <= 10_000_000:
            raise ValueError("ChronosTiny parameter count is outside the 9M contract: %d" % parameter_count)
        if not hasattr(model, "lm_head") or not isinstance(model.lm_head, nn.Linear):
            raise ValueError("ChronosTiny must expose a linear lm_head")
        if model.lm_head.weight.shape[0] != config.n_tokens:
            raise ValueError("ChronosTiny lm_head vocabulary does not match tokenizer")
        return [model], tokenizer_config.create_tokenizer(), ("single",)

    def _can_use_lm_head_fast_path(self) -> bool:
        return bool(
            self.layer_name.endswith("lm_head.weight")
            and isinstance(self.layer_module, nn.Linear)
            and self.parameter.ndim == 2
            and self.parameter.shape[0] > 1
        )

    def batch_tensors(self, windows: np.ndarray | torch.Tensor) -> dict[str, torch.Tensor]:
        values = torch.as_tensor(windows, dtype=torch.float32)
        if values.ndim != 2 or values.shape[1] != self.config.window_length:
            raise ValueError("ChronosTiny windows must have shape [N, context_length + prediction_length]")
        if not torch.isfinite(values).all():
            raise ValueError("ChronosTiny windows contain non-finite values")
        cpu_values = values.detach().cpu()
        contexts = cpu_values[:, :self.config.context_length]
        futures = cpu_values[:, self.config.context_length:]
        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(contexts)
        labels, labels_mask = self.tokenizer.label_input_transform(futures, scale)
        labels = labels.masked_fill(labels_mask == 0, -100)
        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "labels": labels.to(self.device),
        }

    def _fast_batch(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        model: nn.Module | None = None,
        layer_module: nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.model if model is None else model
        layer_module = self.layer_module if layer_module is None else layer_module
        hidden_holder: dict[str, torch.Tensor] = {}

        def capture(_module: nn.Module, args: tuple[Any, ...]) -> None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise ValueError("lm_head did not receive a hidden-state tensor")
            hidden_holder["hidden"] = args[0]

        if layer_module is None:
            raise ValueError("cannot locate module for selected layer: %s" % self.layer_name)
        hook = layer_module.register_forward_pre_hook(capture)
        try:
            with torch.no_grad():
                output = model(**tensors, return_dict=True)
                logits = _model_output_logits(output).float()
                hidden = hidden_holder.get("hidden")
                if hidden is None:
                    raise ValueError("lm_head forward hook did not capture hidden states")
                hidden = hidden.detach().float()
                labels = tensors["labels"]
                losses = per_sample_cross_entropy(logits, labels)
                valid = labels.ne(-100)
                safe_labels = labels.masked_fill(~valid, 0)
                dlogits = logits.softmax(dim=-1)
                dlogits.scatter_add_(-1, safe_labels.unsqueeze(-1), -valid.unsqueeze(-1).to(dlogits.dtype))
                dlogits.mul_(valid.unsqueeze(-1).to(dlogits.dtype))
                dlogits.div_(valid.sum(dim=1).clamp_min(1).view(-1, 1, 1))
                projected = self.tensor_projector.project(dlogits, hidden)  # type: ignore[union-attr]
        finally:
            hook.remove()
        return losses.detach(), projected.detach()

    def _generic_single(
        self,
        window: np.ndarray | torch.Tensor,
        *,
        model: nn.Module | None = None,
        parameter: nn.Parameter | None = None,
    ) -> tuple[float, torch.Tensor]:
        model = self.model if model is None else model
        parameter = self.parameter if parameter is None else parameter
        tensors = self.batch_tensors(np.asarray(window, dtype=np.float32).reshape(1, -1))
        output = model(**tensors, return_dict=True)
        logits = _model_output_logits(output)
        losses = per_sample_cross_entropy(logits, tensors["labels"])
        gradient = torch.autograd.grad(losses[0], parameter, allow_unused=True)[0]
        if gradient is None:
            raise ValueError("selected layer is unused by the model forward: %s" % self.layer_name)
        projected = self.parameter_projector.project(gradient)  # type: ignore[attr-defined]
        return float(losses[0].detach().cpu()), projected.detach()

    def _single_model_gradient_sketch(
        self,
        values: np.ndarray,
        size: int,
        model: nn.Module,
        parameter: nn.Parameter,
        layer_module: nn.Module | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        losses: list[np.ndarray] = []
        sketches: list[np.ndarray] = []
        if self.fast_lm_head:
            for start in range(0, len(values), size):
                loss, projected = self._fast_batch(
                    self.batch_tensors(values[start:start + size]),
                    model=model,
                    layer_module=layer_module,
                )
                losses.append(loss.cpu().numpy())
                sketches.append(projected.cpu().numpy())
        else:
            for row in values:
                loss, projected = self._generic_single(row, model=model, parameter=parameter)
                losses.append(np.asarray([loss], dtype=np.float32))
                sketches.append(projected.cpu().numpy().reshape(1, -1))
        loss_array = np.concatenate(losses).astype(np.float64, copy=False)
        sketch_array = np.concatenate(sketches).astype(np.float32, copy=False)
        if sketch_array.shape != (len(values), self.config.projection_dimension):
            raise ValueError("projected gradient shape is invalid")
        if not np.isfinite(loss_array).all() or not np.isfinite(sketch_array).all():
            raise ValueError("projected gradient output is not finite")
        return loss_array, sketch_array

    def batch_gradient_sketch(
        self, windows: np.ndarray, batch_size: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(windows, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.config.window_length:
            raise ValueError("ChronosTiny gradient windows have invalid geometry")
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("ChronosTiny gradient windows must be non-empty and finite")
        size = batch_size or self.config.gradient_batch_size
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("gradient batch_size must be positive")
        self.model.eval()
        raw_results = [
            self._single_model_gradient_sketch(
                values, size, model, parameter, layer_module,
            )
            for model, parameter, layer_module in zip(
                self.models, self.parameters, self.layer_modules,
            )
        ]
        if len(raw_results) == 1:
            return raw_results[0]
        # Preserve each checkpoint's result independently. Axis 0 is the
        # checkpoint/seed axis; no ensemble averaging is performed.
        return (
            np.stack([loss for loss, _sketch in raw_results], axis=0),
            np.stack([sketch for _loss, sketch in raw_results], axis=0),
        )


__all__ = [
    "ChronosTinyConfig", "ChronosTinyGradientEngine", "DEFAULT_CHRONOS_TINY_MODEL",
    "DEFAULT_LAYER", "DEFAULT_WARMUP_MODEL_ROOT", "DEFAULT_WARMUP_SEEDS",
    "per_sample_cross_entropy", "select_parameter",
]
