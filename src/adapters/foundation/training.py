"""Middle-layer fine-tuning service for adapters with an official loss."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .base import FoundationModelAdapter
from .components import as_finite_batch
from .spec import ModelCapability


@dataclass(frozen=True)
class FineTuneConfig:
    context_length: int
    prediction_length: int
    steps: int = 100
    batch_size: int = 1
    gradient_accumulation: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("context_length", "prediction_length", "steps", "batch_size",
                     "gradient_accumulation"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("%s must be a positive integer" % name)
        for name in ("learning_rate", "max_grad_norm"):
            if name == "max_grad_norm" and self.max_grad_norm is None:
                continue
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be positive and finite" % name)
        if not math.isfinite(float(self.weight_decay)) or self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative and finite")


@dataclass(frozen=True)
class FineTuneResult:
    losses: tuple[float, ...]
    optimizer_steps: int
    examples_seen: int

    def as_mapping(self) -> dict[str, object]:
        return {
            "losses": list(self.losses),
            "optimizer_steps": self.optimizer_steps,
            "examples_seen": self.examples_seen,
            "final_loss": self.losses[-1],
        }


class FoundationFineTuner:
    """Deterministic AdamW loop over an immutable array of selected windows."""

    def __init__(self, adapter: FoundationModelAdapter, config: FineTuneConfig) -> None:
        adapter.spec.require(ModelCapability.FINE_TUNE)
        adapter.spec.require(ModelCapability.TRAINING_LOSS)
        if config.context_length > adapter.spec.max_context:
            raise ValueError("training context exceeds the model contract")
        self.adapter = adapter
        self.config = config

    def fit(self, windows: np.ndarray) -> FineTuneResult:
        import torch

        values = as_finite_batch(windows, name="windows")
        expected = self.config.context_length + self.config.prediction_length
        if values.shape[1] != expected:
            raise ValueError("windows must have context_length + prediction_length columns")
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        model = self.adapter.load()
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        generator = np.random.default_rng(self.config.seed)
        optimizer.zero_grad(set_to_none=True)
        history: list[float] = []
        optimizer_steps = 0
        examples_seen = 0
        for step in range(self.config.steps):
            positions = generator.integers(0, len(values), size=self.config.batch_size)
            batch = values[positions]
            loss = self.adapter.training_loss(
                batch,
                context_length=self.config.context_length,
                prediction_length=self.config.prediction_length,
            )
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError("foundation-model training loss must be a finite scalar")
            history.append(float(loss.detach().float().cpu()))
            group_start = (step // self.config.gradient_accumulation) * self.config.gradient_accumulation
            group_size = min(self.config.gradient_accumulation, self.config.steps - group_start)
            (loss / group_size).backward()
            examples_seen += len(batch)
            boundary = (step + 1) % self.config.gradient_accumulation == 0
            final = step + 1 == self.config.steps
            if boundary or final:
                if self.config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
        model.eval()
        return FineTuneResult(tuple(history), optimizer_steps, examples_seen)


__all__ = ["FineTuneConfig", "FineTuneResult", "FoundationFineTuner"]
