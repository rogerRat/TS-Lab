"""Warm-up candidate generation for gradient experiments."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import numpy as np
import torch


def _safe_component(value: str) -> str:
    value = str(value)
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError("path component is invalid: %s" % value)
    return value


def _validate_windows(windows: np.ndarray, window_length: int) -> np.ndarray:
    values = np.asarray(windows, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != window_length:
        raise ValueError("warmup windows must have shape [N, %d]" % window_length)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("warmup windows must be non-empty and finite")
    return values


class WarmupCandidateRunner:
    """Train and persist several small proxy checkpoints from one base model.

    Every learning-rate candidate starts from the exact same in-memory base
    state and consumes the same deterministic resampled windows. Existing
    candidate directories are rejected to keep experiment outputs immutable.
    """

    def __init__(self, engine: Any, output_root: Path = Path("outputs/models/warmup")) -> None:
        self.engine = engine
        self.output_root = Path(output_root)

    def run(
        self,
        windows: np.ndarray,
        *,
        learning_rates: tuple[float, ...] | list[float] | None = None,
        steps: int | None = None,
        batch_size: int | None = None,
        run_id: str = "chronos-tiny-seed42",
        seed: int | None = None,
    ) -> dict[str, Any]:
        config = self.engine.config
        values = _validate_windows(windows, config.window_length)
        rates = tuple(float(value) for value in (learning_rates or config.warmup_learning_rates))
        count = int(steps or config.warmup_steps)
        size = int(batch_size or config.warmup_batch_size)
        if not rates or any(value <= 0.0 or not np.isfinite(value) for value in rates):
            raise ValueError("learning_rates must be positive and finite")
        if count <= 0 or size <= 0:
            raise ValueError("warmup steps and batch_size must be positive")
        run_name = _safe_component(run_id)
        run_dir = (self.output_root / run_name).resolve()
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError("warmup output is not empty: %s" % run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        base_state = {
            name: value.detach().cpu().clone()
            for name, value in self.engine.model.state_dict().items()
        }
        base_rng_state = torch.get_rng_state()
        candidate_records: list[dict[str, Any]] = []
        seed_value = config.seed if seed is None else int(seed)
        try:
            for index, learning_rate in enumerate(rates):
                self.engine.model.load_state_dict(base_state, strict=True)
                torch.set_rng_state(base_rng_state)
                optimizer = torch.optim.AdamW(self.engine.model.parameters(), lr=learning_rate)
                rng = np.random.default_rng(seed_value)
                losses: list[float] = []
                self.engine.model.train()
                for _step in range(count):
                    indices = rng.integers(0, len(values), size=size)
                    tensors = self.engine.batch_tensors(values[indices])
                    optimizer.zero_grad(set_to_none=True)
                    output = self.engine.model(**tensors)
                    loss = getattr(output, "loss", None)
                    if loss is None:
                        logits = getattr(output, "logits", None)
                        if logits is None:
                            raise ValueError("warmup model output has neither loss nor logits")
                        from .chronos import per_sample_cross_entropy
                        loss = per_sample_cross_entropy(logits, tensors["labels"]).mean()
                    if not torch.isfinite(loss):
                        raise ValueError("warmup loss became non-finite at step %d" % (_step + 1))
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                self.engine.model.eval()
                candidate_name = "candidate-%02d-lr-%s" % (index, format(learning_rate, ".0e").replace("+", "p"))
                candidate_dir = run_dir / _safe_component(candidate_name)
                self._save_model(candidate_dir)
                record = {
                    "candidate": candidate_name,
                    "learning_rate": learning_rate,
                    "steps": count,
                    "batch_size": size,
                    "seed": seed_value,
                    "initial_loss": losses[0],
                    "final_loss": losses[-1],
                    "loss_trace_tail": losses[-10:],
                    "model_dir": str(candidate_dir.relative_to(run_dir)),
                }
                (candidate_dir / "warmup.json").write_text(
                    json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8",
                )
                candidate_records.append(record)
        finally:
            self.engine.model.load_state_dict(base_state, strict=True)
            self.engine.model.eval()
        manifest = {
            "schema_version": "tsl.gradient-warmup.v1",
            "run_id": run_name,
            "model_path": str(config.model_path),
            "window_length": config.window_length,
            "layer": self.engine.layer_name,
            "source_window_count": int(len(values)),
            "candidates": candidate_records,
            "protocol": config.as_mapping(),
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8",
        )
        return manifest

    def _save_model(self, target: Path) -> None:
        target.mkdir(parents=True, exist_ok=False)
        model = getattr(self.engine.model, "_orig_mod", self.engine.model)
        save_pretrained = getattr(model, "save_pretrained", None)
        if callable(save_pretrained):
            save_pretrained(target, safe_serialization=True)
        else:
            torch.save(model.state_dict(), target / "model_state.pt")


class FromScratchWarmupRunner:
    """Train one independent Tiny checkpoint from lazily sampled source data."""

    def __init__(self, engine: Any, output_root: Path = Path("outputs/models/warmup")) -> None:
        self.engine = engine
        self.output_root = Path(output_root)

    def run(
        self,
        batch_provider: Any,
        *,
        steps: int,
        batch_size: int,
        learning_rate: float,
        run_id: str,
        seed: int,
        provenance: dict[str, Any] | None = None,
        log_every: int = 100,
    ) -> dict[str, Any]:
        """Train and save a single model without materializing the source set."""
        if not hasattr(batch_provider, "next_batch"):
            raise TypeError("batch_provider must expose next_batch()")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("steps must be a positive integer")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if learning_rate <= 0.0 or not np.isfinite(learning_rate):
            raise ValueError("learning_rate must be positive and finite")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        run_name = _safe_component(run_id)
        run_dir = (self.output_root / run_name).resolve()
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError("warmup output is not empty: %s" % run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        optimizer = torch.optim.AdamW(self.engine.model.parameters(), lr=float(learning_rate))
        losses: list[float] = []
        self.engine.model.train()
        for step in range(steps):
            windows = np.asarray(batch_provider.next_batch(), dtype=np.float32)
            expected_shape = (batch_size, self.engine.config.window_length)
            if windows.shape != expected_shape or not np.isfinite(windows).all():
                raise ValueError("warmup batch has invalid shape or non-finite values")
            tensors = self.engine.batch_tensors(windows)
            optimizer.zero_grad(set_to_none=True)
            output = self.engine.model(**tensors)
            loss = getattr(output, "loss", None)
            if loss is None:
                logits = getattr(output, "logits", None)
                if logits is None:
                    raise ValueError("warmup model output has neither loss nor logits")
                from .chronos import per_sample_cross_entropy
                loss = per_sample_cross_entropy(logits, tensors["labels"]).mean()
            if not torch.isfinite(loss):
                raise ValueError("warmup loss became non-finite at step %d" % (step + 1))
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            if log_every > 0 and ((step + 1) % log_every == 0 or step == 0):
                print("warmup %s step=%d/%d loss=%.6f" % (run_name, step + 1, steps, loss_value), flush=True)

        self.engine.model.eval()
        model_dir = run_dir / "model"
        self._save_model(model_dir)
        record = {
            "schema_version": "tsl.gradient-warmup.v2",
            "run_id": run_name,
            "seed": seed,
            "initialization_mode": "from_scratch",
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": float(learning_rate),
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "loss_trace_tail": losses[-10:],
            "model_dir": "model",
            "layer": self.engine.layer_name,
            "parameter_count": self.engine.parameter_count,
            "protocol": self.engine.config.as_mapping(),
            "source": provenance or {},
        }
        (run_dir / "warmup.json").write_text(
            json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8",
        )
        return record

    def _save_model(self, target: Path) -> None:
        target.mkdir(parents=True, exist_ok=False)
        model = getattr(self.engine.model, "_orig_mod", self.engine.model)
        save_pretrained = getattr(model, "save_pretrained", None)
        if callable(save_pretrained):
            save_pretrained(target, safe_serialization=True)
        else:
            torch.save(model.state_dict(), target / "model_state.pt")


__all__ = ["FromScratchWarmupRunner", "WarmupCandidateRunner"]
