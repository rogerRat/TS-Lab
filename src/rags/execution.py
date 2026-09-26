"""Shared execution policy for selection: batching, prepared inputs and forecasts.

No sampling, ranking or optimizer decisions belong here. All selection arms
use the same policy; inference batching never changes an optimizer's batch.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SelectionExecution:
    gradient_batch: int = 0  # 0: choose from available device memory
    inference_batch: int = 0
    memory_fraction: float = 0.7
    cpu_threads: int = 4
    generation: str = "cuda_graph"
    gradient_workers: int = 2  # independent native windows for sparse-routing models
    gradient_vjp_batch: int = 0  # 0: adapter default; independent Jacobian rows per backward

    def __post_init__(self):
        if self.gradient_batch < 0 or self.inference_batch < 0 or self.gradient_workers < 1 or self.gradient_vjp_batch < 0:
            raise ValueError("batch sizes must be nonnegative and gradient_workers positive")
        if not 0 < self.memory_fraction <= 1:
            raise ValueError("memory_fraction must be in (0,1]")

    def batch_size(self, engine, kind, *, fallback, count):
        explicit = self.gradient_batch if kind == "gradient" else self.inference_batch
        if explicit:
            return min(explicit, count)
        device = torch.device(engine.device)
        if device.type != "cuda":
            return min(fallback, count)
        available, _ = torch.cuda.mem_get_info(device)
        available += torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        config = engine.model.config
        width = getattr(config, "d_model", getattr(config, "hidden_size", 512))
        layers = getattr(config, "num_layers", getattr(config, "num_hidden_layers", 6))
        context = engine.config.context_length + 1
        horizon = engine.config.prediction_length + 1
        # Conservative eager attention/activation allowance, plus one complete
        # per-sample parameter gradient for vmap. Bound the number of tiny
        # operations too: the largest fitting batch is not always the fastest.
        activations = layers * (context + horizon) * width * 4 * 32
        if kind == "gradient":
            per_row = activations + 4 * sum(p.numel() for p in engine.model.parameters())
            ceiling = 128
        else:
            heads = getattr(config, "num_heads", 8)
            attention = 3 * 4 * heads * (context ** 2 + horizon ** 2)
            hidden = 12 * 4 * width * (context + horizon)
            kv = 2 * 4 * layers * width * (context + horizon) if kind == "inference" else 0
            per_row = attention + hidden + kv
            ceiling = 1024
        capacity = max(1, int(available * self.memory_fraction / per_row))
        size = min(ceiling, 2 ** (capacity.bit_length() - 1))
        return min(size, count)


def execution_policy(engine):
    return getattr(engine, "execution", SelectionExecution())


def clear_execution_cache(engine):
    """Release GPU-address-bound graphs/views before moving the probe off GPU."""
    for name in ("_greedy_graphs", "_vmap_cache"):
        engine.__dict__.pop(name, None)


@contextmanager
def selection_execution(engine):
    """Small CPU tensor operations must not fan out to hundreds of threads."""
    previous = torch.get_num_threads()
    threads = execution_policy(engine).cpu_threads
    from threadpoolctl import threadpool_limits
    if threads:
        torch.set_num_threads(threads)
    try:
        # FAISS ships its own OpenMP runtime; cap it alongside Torch and BLAS.
        with threadpool_limits(limits=threads or None):
            yield
    finally:
        torch.set_num_threads(previous)


@dataclass
class PreparedContexts:
    windows: np.ndarray
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    scale: torch.Tensor

    def __len__(self):
        return len(self.windows)


def prepare_contexts(engine, windows):
    """Tokenize fixed histories once; these tensors are independent of weights.

    Lifetime is explicit in the caller. No hashes, implicit stale cache or
    cross-update reuse of model outputs, encoder features or KV states.
    """
    if isinstance(windows, PreparedContexts):
        return windows
    windows = np.asarray(windows, dtype=np.float32)
    contexts = torch.from_numpy(windows[:, :engine.config.context_length])
    ids, mask, scale = engine.tokenizer.context_input_transform(contexts)
    return PreparedContexts(windows, ids.to(engine.device), mask.to(engine.device), scale)


@torch.no_grad()
def forecast_mse(engine, windows, *, batch_size=32):
    """Original deterministic greedy forecast MSE, with shared input batching."""
    prepared = prepare_contexts(engine, windows)
    horizon = engine.config.prediction_length
    batch = execution_policy(engine).batch_size(
        engine, "inference", fallback=batch_size, count=len(prepared))
    errors = []
    for begin in range(0, len(prepared), batch):
        end = begin + batch
        from rags.generation import generate_tokens
        tokens, _ = generate_tokens(engine, prepared.input_ids[begin:end],
                                    prepared.attention_mask[begin:end])
        forecast = engine.tokenizer.output_transform(
            tokens.cpu()[:, None, :], prepared.scale[begin:end])[:, 0]
        truth = torch.from_numpy(prepared.windows[begin:end, engine.config.context_length:])
        # Preserve the original row order and 32-row summation groups even if
        # more rows were generated together on the GPU.
        errors.append((forecast - truth).square())
    total = sum(float(chunk.sum()) for chunk in torch.cat(errors).split(batch_size))
    return total / (len(prepared) * horizon)


@torch.no_grad()
def encoder_embeddings(engine, windows, *, batch_size=128):
    """Shared frozen encoder execution; mean pooling is mask-aware as before."""
    native = getattr(engine, "encoder_embeddings", None)
    if callable(native):
        return native(windows, batch_size=batch_size)
    batch = execution_policy(engine).batch_size(
        engine, "encoder", fallback=batch_size, count=len(windows))
    rows = []
    for begin in range(0, len(windows), batch):
        prepared = prepare_contexts(engine, windows[begin:begin + batch])
        hidden = engine.model.get_encoder()(
            input_ids=prepared.input_ids, attention_mask=prepared.attention_mask,
            return_dict=True).last_hidden_state
        mask = prepared.attention_mask.unsqueeze(-1)
        rows.append(((hidden * mask).sum(1) / mask.sum(1).clamp_min(1)).float().cpu())
    return torch.cat(rows).numpy()
