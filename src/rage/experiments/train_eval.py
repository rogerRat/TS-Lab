"""Unified single-stage fine-tuning schedule and MASE evaluation.

All arms share the identical schedule (experiment plan domain D1/D2): full-
parameter fine-tuning on the selected N source windows for the production
exposure epochs, with every scheduled epoch checkpoint evaluated. There is no
target-side adaptation stage (domain A7: reference data never enters training).
Evaluation uses the model's own categorical distribution readout (median /
quantile forecasts); MASE denominators are the per-series full-history
seasonal-naive scales (domain E1).

Production training uses :class:`TrainingRecipe` (frozen per-model settings
from ``configs/rage/shared_tasks_10.json``): per-model weight decay, AdamW
betas/eps, micro-batch gradient accumulation, grad-norm clipping, warmup 0,
and a learning-rate schedule whose total length (``schedule_epochs``) is
independent of the stop epoch, including fractional stop epochs such as 2.625
(steps = ceil(E * N / effective_batch)). Passing ``recipe=None`` keeps the
legacy constant-learning-rate diagnostic path used by probing untouched.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from collections.abc import Mapping, Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from rage.core import TimeSeriesFragment

from .metrics import point_metrics, quantile_forecasts
from .schedule import epoch_checkpoints


def crb_gradient_computer(engine, context_length: int, prediction_length: int):
    """BatchedGradient computer over (fragment, physical starts) for CRBEngine."""

    def compute(fragment: TimeSeriesFragment, starts: np.ndarray) -> np.ndarray:
        windows = np.stack([
            fragment.extract_window(int(start), context_length, prediction_length)
            for start in starts
        ])
        _losses, sketches = engine.batch_gradient_sketch(windows)
        return sketches

    return compute


@dataclass(frozen=True)
class TrainingRecipe:
    """Frozen production training settings for one student model.

    Built from the per-model block of ``configs/rage/shared_tasks_10.json``
    (authoritative record for the safe-history development probes).  The
    learning-rate schedule spans ``schedule_epochs`` epochs regardless of the
    stop epoch, so stopping early leaves the LR on its full-length trajectory.
    """

    learning_rate: float
    epochs: float                      # stop epoch; fractional allowed (e.g. 2.625)
    batch_size: int                    # effective batch per optimizer step
    micro_batch_size: int              # forward/backward chunk (must divide batch)
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.0
    schedule: str = "linear"           # "linear" | "cosine"
    min_lr_ratio: float = 0.0
    schedule_epochs: float = 6.0
    exclude_bias_norm: bool = False    # no weight decay on bias/norm parameters

    def __post_init__(self) -> None:
        positive = dict(learning_rate=self.learning_rate, epochs=self.epochs,
                        batch_size=self.batch_size, micro_batch_size=self.micro_batch_size,
                        eps=self.eps, schedule_epochs=self.schedule_epochs)
        for name, value in positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"TrainingRecipe.{name} must be finite and positive")
        if self.batch_size % self.micro_batch_size:
            raise ValueError("effective batch must be a multiple of micro_batch_size")
        if self.schedule not in ("linear", "cosine"):
            raise ValueError("schedule must be 'linear' or 'cosine'")
        if not 0.0 <= self.min_lr_ratio <= 1.0 or not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("LR ratios must be within [0, 1) (min_lr_ratio <= 1)")
        if len(self.betas) != 2 or not all(0.0 < beta < 1.0 for beta in self.betas):
            raise ValueError("betas must be two values in (0, 1)")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")

    @classmethod
    def from_settings(cls, settings: Mapping) -> "TrainingRecipe":
        """Parse one per-model block of shared_tasks_10.json."""
        optimizer = settings["optimizer"]
        return cls(
            learning_rate=float(settings["learning_rates"][0]),
            epochs=float(settings["epochs"][0]),
            batch_size=int(settings["batch_size"]),
            micro_batch_size=int(optimizer["micro_batch_size"]),
            weight_decay=float(settings["weight_decay"]),
            betas=(float(optimizer["betas"][0]), float(optimizer["betas"][1])),
            eps=float(optimizer["eps"]),
            max_grad_norm=float(optimizer["max_grad_norm"]),
            warmup_ratio=float(optimizer["warmup_ratio"]),
            schedule=str(optimizer["schedule"]),
            min_lr_ratio=float(optimizer["min_lr_ratio"]),
            schedule_epochs=float(settings["schedule_epochs"]),
            exclude_bias_norm=bool(optimizer.get("exclude_bias_norm", False)))

    def as_dict(self) -> dict:
        data = asdict(self)
        data["betas"] = tuple(self.betas)
        return data

    @property
    def accumulation(self) -> int:
        return self.batch_size // self.micro_batch_size

    def steps_for(self, n_windows: int, epochs: float | None = None) -> int:
        """Optimizer steps for ``epochs`` exposures: ceil(E * N / effective batch)."""
        epochs = self.epochs if epochs is None else epochs
        return int(math.ceil(float(epochs) * int(n_windows) / self.batch_size))

    def lr_multiplier(self, step: int, total_steps: int) -> float:
        """Deterministic per-step multiplier; the scheduler is a pure function.

        ``step`` is the 0-based index of the optimizer step about to run, so
        resuming only needs the completed-step count (no scheduler state).
        """
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        warmup_steps = int(self.warmup_ratio * total_steps)
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(max(step / total_steps, 0.0), 1.0)
        if self.schedule == "linear":
            return 1.0 + (self.min_lr_ratio - 1.0) * progress
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress))


def observation_epochs(stop_epoch: float) -> list[float]:
    """Half-epoch observation grid up to the stop epoch, plus the stop itself.

    Keeps the probing-era half-epoch evaluation density and guarantees the
    fractional stop epoch (e.g. TimesFM's 2.625) is always evaluated.
    """
    grid = [round(0.5 * index, 10) for index in range(1, int(math.ceil(stop_epoch / 0.5)))
            if 0.5 * index < stop_epoch]
    return sorted(set(grid + [float(stop_epoch)]))


def _build_optimizer(model: torch.nn.Module, recipe: TrainingRecipe) -> torch.optim.AdamW:
    """AdamW per recipe with optional bias/norm weight-decay exclusion.

    Mirrors ``precision.make_adamw`` (``foreach=False``; optimizer state dtype
    follows the BF16 parameters), adding the parameter groups the recipe needs.
    """
    if recipe.exclude_bias_norm:
        decay, no_decay = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            target = no_decay if (parameter.ndim <= 1 or name.endswith(".bias")
                                  or "norm" in name.lower()) else decay
            target.append(parameter)
        groups = [{"params": decay, "weight_decay": recipe.weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
    else:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                   "weight_decay": recipe.weight_decay}]
    return torch.optim.AdamW(groups, lr=recipe.learning_rate, betas=tuple(recipe.betas),
                             eps=recipe.eps, foreach=False)


def fine_tune(
    engine,
    windows: np.ndarray,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    weight_decay: float = 0.01,
    on_step: Callable[[int], None] | None = None,
    max_seconds: float | None = None,
    sampling_probabilities: np.ndarray | None = None,
    resume_path: Path | None = None,
    checkpoint_steps=None,
    recipe: TrainingRecipe | None = None,
    schedule_total_steps: int | None = None,
    training_options: dict | None = None,
) -> dict:
    """AdamW full-parameter fine-tuning; restores nothing (caller owns state).

    ``recipe`` switches on the production semantics: parameter-group AdamW with
    frozen betas/eps, micro-batch gradient accumulation, grad-norm clipping,
    and a step-indexed learning-rate schedule spanning ``schedule_total_steps``
    (default: the schedule length implied by ``recipe.schedule_epochs`` times
    the pool size, or ``steps`` when no recipe is given).  Without a recipe the
    legacy constant-learning-rate diagnostic path is unchanged.
    """
    if len(windows) == 0 or steps == 0:
        return {"initial_loss": None, "final_loss": None, "steps": 0}
    options = None
    if training_options is not None:
        if recipe is not None:
            raise ValueError("Use recipe or training_options")
        from .optimization import optimizer_options, learning_rate_factor
        options = optimizer_options(training_options, batch_size=batch_size, steps=steps)
    if recipe is not None:
        if batch_size != recipe.batch_size:
            raise ValueError("batch_size must equal recipe.batch_size")
        if learning_rate != recipe.learning_rate:
            raise ValueError("learning_rate must equal recipe.learning_rate")
    schedule_total = int(schedule_total_steps or steps)
    model = engine.model
    native_loss = getattr(engine, "training_loss", None)
    logits_inputs = getattr(engine, "_logits_inputs", lambda tensors: tensors)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if options is not None:
        from .precision import make_adamw
        parameters = model
        if options["exclude_bias_norm"]:
            decay, no_decay = [], []
            for name, parameter in model.named_parameters():
                (no_decay if name.endswith("bias") or "norm" in name.lower() else decay).append(parameter)
            parameters = [{"params": decay, "weight_decay": weight_decay},
                          {"params": no_decay, "weight_decay": 0.}]
        optimizer = make_adamw(parameters, lr=learning_rate, weight_decay=weight_decay,
                              betas=tuple(options["betas"]), eps=options["eps"])
    elif recipe is not None:
        optimizer = _build_optimizer(model, recipe)
    else:
        from .precision import make_adamw
        optimizer = make_adamw(model, lr=learning_rate, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    from .regions import RegionWindows
    from .resampling import ResampledWindows
    resampled = isinstance(windows, ResampledWindows)
    if resampled and sampling_probabilities is not None:
        raise ValueError("Static resampling probabilities must not be applied twice")
    regional = isinstance(windows, (RegionWindows, ResampledWindows))
    if not regional:
        windows = np.asarray(windows, dtype=np.float32)
    prepare = getattr(engine, "prepare_training_tensors", None)
    prepared = (prepare(windows) if not regional and prepare is not None
                and steps * min(batch_size, len(windows)) > len(windows)
                else None)
    model.train()
    initial_loss: torch.Tensor | None = None
    tail_losses: deque[torch.Tensor] = deque(maxlen=50)
    started = time.monotonic()
    completed = 0
    prior_elapsed = 0.
    seen = np.zeros(len(windows), dtype=bool) if resampled else None
    contract = dict(steps=steps, batch_size=batch_size, learning_rate=learning_rate,
                    weight_decay=weight_decay, seed=seed,
                    sampling=windows.metadata if resampled else None)
    if recipe is not None:
        contract.update(recipe=recipe.as_dict(), schedule_total_steps=schedule_total)
    if options is not None:
        contract["training_options"] = options
    contract["parameter_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
    resume_path = Path(resume_path) if resume_path is not None else None
    save_at = set(checkpoint_steps or [steps])
    if resume_path is not None and resume_path.exists():
        # Only load this run's trusted, locally generated optimizer checkpoint.
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state["contract"] != contract:
            from .v7_amendment import approved_prefix_contract
            if not (getattr(engine, 'allow_v7_epoch_prefix_resume', False) and
                    approved_prefix_contract(state['contract'],contract)):
                raise ValueError("Resume training/model/pool contract changed; use a new cell")
        if resampled and not np.array_equal(state["probabilities"], windows.probabilities):
            raise ValueError("Resume sampling probabilities changed; use a new cell")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        completed, seen = state["step"], state["seen"]
        prior_elapsed = state["training_elapsed_seconds"]
        initial_loss = state["initial_loss"].to(next(model.parameters()).device)
        tail_losses.extend(value.to(initial_loss.device) for value in state["tail_losses"])
        del state
    resumed_from = completed
    for step in range(completed, steps):
        if max_seconds is not None and time.monotonic() - started >= max_seconds:
            break
        # One frozen-sample draw of the effective batch per optimizer step;
        # micro-batching only partitions the forward/backward of that draw.
        indices = (windows.sample_indices(rng, batch_size) if resampled else
                   rng.integers(0, len(windows), size=batch_size) if sampling_probabilities is None
                   else rng.choice(len(windows), size=batch_size, replace=True, p=sampling_probabilities))
        if seen is not None:
            seen[indices] = True
        optimizer.zero_grad(set_to_none=True)
        if options is not None:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * learning_rate_factor(step, options)
        if recipe is not None:
            multiplier = recipe.lr_multiplier(step, schedule_total)
            for group in optimizer.param_groups:
                group["lr"] = recipe.learning_rate * multiplier
        micro_batches = (np.array_split(indices, recipe.accumulation)
                         if recipe is not None else [indices])
        if options is not None:
            micro = options["micro_batch_size"]
            micro_batches = [indices[offset:offset + micro] for offset in range(0, len(indices), micro)]
        micro_scale = 1.0 / len(micro_batches)
        micro_losses = []
        for micro_indices in micro_batches:
            if native_loss is not None:
                loss = native_loss(windows[micro_indices])
            else:
                from rage.engines import per_sample_cross_entropy
                if prepared is None:
                    tensors = engine.batch_tensors(windows[micro_indices])
                else:
                    device = next(iter(prepared.values())).device
                    rows = torch.as_tensor(micro_indices, dtype=torch.long, device=device)
                    tensors = {key: value.index_select(0, rows) for key, value in prepared.items()}
                output = model(**logits_inputs(tensors), return_dict=True)
                loss = per_sample_cross_entropy(output.logits, tensors["labels"]).mean()
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError("training loss must be a finite scalar")
            if len(micro_batches) == 1:
                loss.backward()
            else:
                (loss * (len(micro_indices) / len(indices) if options is not None else micro_scale)).backward()
            micro_losses.append(loss.detach())
        if options is not None and options["max_grad_norm"] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), options["max_grad_norm"], error_if_nonfinite=True)
        if recipe is not None and recipe.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                recipe.max_grad_norm)
        optimizer.step()
        detached_loss = (micro_losses[0] if len(micro_losses) == 1
                         else torch.stack(micro_losses).mean())
        if options is not None:
            detached_loss = sum(loss * len(rows) / len(indices) for loss, rows in zip(micro_losses, micro_batches))
        if initial_loss is None:
            initial_loss = detached_loss
        tail_losses.append(detached_loss)
        completed = step + 1
        temporary = None
        if resume_path is not None and completed in save_at:
            resume_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = resume_path.with_suffix(resume_path.suffix + ".tmp")
            torch.save(dict(contract=contract, model=model.state_dict(),
                optimizer=optimizer.state_dict(), step=completed, numpy_rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                seen=seen, initial_loss=initial_loss, tail_losses=list(tail_losses),
                training_elapsed_seconds=prior_elapsed + time.monotonic() - started,
                probabilities=windows.probabilities if resampled else sampling_probabilities), temporary)
        if on_step is not None:
            on_step(step + 1)
        if temporary is not None:
            # Keep the previous recoverable state until this checkpoint's
            # evaluation callback and durable curve write both succeed.
            temporary.replace(resume_path)
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    if not completed:
        return {"initial_loss": None, "final_loss": None, "steps": 0}
    result = {
        "initial_loss": float(initial_loss.cpu()),
        "final_loss": float(tail_losses[-1].cpu()),
        "tail_mean": float(torch.stack(tuple(tail_losses)).mean().cpu()),
        "steps": completed,
        "resumed_from_step": resumed_from,
        "cumulative_training_seconds": prior_elapsed + time.monotonic() - started,
    }
    if seen is not None:
        draws = completed * batch_size
        result.update(sample_exposures=draws, unique_window_exposures=int(seen.sum()),
                      repeated_window_exposures=draws - int(seen.sum()),
                      repeat_fraction=1 - int(seen.sum()) / draws,
                      pool_equivalents=windows.pool_n, actual_epochs=draws / windows.pool_n)
    return result


def evaluate(
    engine,
    test_windows: np.ndarray,
    test_entities: list[str],
    *,
    context_length: int,
    batch_size: int = 128,
    test_scales: np.ndarray | None = None,
) -> dict:
    """Per-series MASE and their dataset mean; separate constant-series MAE.

    ``test_scales`` carries each window's full-history seasonal-naive scale
    (domain E1); when omitted the legacy per-window context scale is used.
    """
    engine.model.eval()
    # Return unused training/scoring reserves to other parallel students before
    # autoregressive evaluation requests its larger attention workspace.
    if torch.device(getattr(engine, "device", "cpu")).type == "cuda":
        torch.cuda.empty_cache()
    if hasattr(engine, "forecast"):
        median = np.concatenate([engine.forecast(test_windows[i:i + batch_size, :context_length])
                                 for i in range(0, len(test_windows), batch_size)])
    else:
        quantiles, _scales = quantile_forecasts(engine, test_windows, batch_size=batch_size)
        median = quantiles[:, :, 4]
    return point_metrics(test_windows, median, test_entities,
                         context_length=context_length, scales=test_scales)


def _online_curve_recipe(engine, windows, *, base_state, recipe, seed, checkpoints,
                         schedule_total_steps, evaluate_fn, on_record=None,
                         resume_path=None, observations_path=None):
    """Recipe-aware port of ``calibration.online_curve``.

    Record and observe semantics are identical (one continuous optimizer,
    forked evaluation RNG so evaluation cannot consume training randomness,
    rolling observations file, latest-boundary resume state); calibration.py
    keeps the legacy constant-LR diagnostic path used by probing.
    """
    from .students import atomic_json
    observations_path = Path(observations_path) if observations_path is not None else None
    records = (json.loads(observations_path.read_text()) if observations_path is not None
               and observations_path.exists() else [])
    started = time.monotonic()
    eval_seconds = 0.
    step_epochs = {}
    for epoch, step in checkpoints:
        step_epochs.setdefault(step, []).append(epoch)
    if getattr(engine, 'allow_v7_epoch_prefix_resume', False):
        records = [r for r in records if r['step'] <= max(step_epochs)]
    device = torch.device(engine.device)
    devices = ([device.index if device.index is not None else torch.cuda.current_device()]
               if device.type == "cuda" else [])

    def observe(step):
        nonlocal eval_seconds
        if step not in step_epochs:
            return
        was_training = engine.model.training
        eval_started = time.monotonic()
        try:
            with torch.random.fork_rng(devices=devices):
                metrics = evaluate_fn()
        finally:
            engine.model.train(was_training)
        eval_seconds += time.monotonic() - eval_started
        for epoch in step_epochs[step]:
            record = dict(epoch=epoch, step=step, metrics=metrics,
                          sample_exposures=step * recipe.batch_size,
                          training_seconds=time.monotonic() - started - eval_seconds,
                          evaluation_seconds=eval_seconds)
            if hasattr(windows, "pool_n"):
                record["actual_epoch"] = step * recipe.batch_size / windows.pool_n
            records[:] = [previous for previous in records if previous["epoch"] != epoch]
            records.append(record)
            records.sort(key=lambda item: item["epoch"])
            if observations_path is not None:
                atomic_json(observations_path, records)
            if on_record:
                on_record(record)

    engine.model.load_state_dict(base_state)
    try:
        log = fine_tune(engine, windows, steps=max(step_epochs),
                        batch_size=recipe.batch_size, learning_rate=recipe.learning_rate,
                        seed=seed, on_step=observe, resume_path=resume_path,
                        checkpoint_steps=list(step_epochs), recipe=recipe,
                        schedule_total_steps=schedule_total_steps)
        return dict(curve=records, train=log, learning_rate=recipe.learning_rate,
                    seed=seed, training_recipe=recipe.as_dict(),
                    checkpoint_policy=("online_metrics_and_latest_optimizer_state"
                                       if resume_path is not None
                                       else "online_metrics_only"),
                    evaluation_seconds=eval_seconds)
    finally:
        engine.model.load_state_dict(base_state)
        engine.model.eval()


def run_arm_training(
    engine,
    *,
    source_windows: np.ndarray,
    source_steps: int | None = None,
    source_epochs: int | None = None,
    budget_windows: int | None = None,
    checkpoint_dir: Path | None = None,
    training_seconds: float | None = None,
    batch_size: int,
    learning_rate: float,
    seed: int,
    test_windows: np.ndarray,
    test_entities: list[str],
    context_length: int,
    test_scales: np.ndarray | None = None,
    base_state: Mapping[str, torch.Tensor] | None = None,
    online_evaluation: bool = False,
    original_mase: float | None = None,
    recipe: TrainingRecipe | None = None,
) -> dict:
    """One cell: continuous AdamW training, save/evaluate every epoch, restore base.

    The final epoch is the main-table result, never the best test checkpoint.
    Explicit source_steps is retained only for low-level diagnostic callers.
    ``recipe`` selects the frozen production schedule: steps are derived as
    ceil(epochs * N / effective_batch), evaluation runs on the half-epoch grid
    plus the (possibly fractional) stop epoch, and the learning rate follows a
    schedule_epochs-long trajectory that does not end at the stop epoch.
    """
    model = engine.model
    # The main-table runner shares one immutable pretrained snapshot across
    # all cells. Keep the local fallback for standalone callers.
    if base_state is None:
        base_state = {name: value.detach().clone()
                      for name, value in model.state_dict().items()}
    if recipe is not None:
        if source_steps is not None:
            raise ValueError("recipe training derives steps from epochs; drop source_steps")
        if batch_size != recipe.batch_size:
            raise ValueError("batch_size must equal recipe.batch_size")
        if learning_rate != recipe.learning_rate:
            raise ValueError("learning_rate must equal recipe.learning_rate")
        if source_epochs is not None and float(source_epochs) != float(recipe.epochs):
            raise ValueError("source_epochs conflicts with recipe.epochs")
        source_epochs = float(recipe.epochs)
    if source_epochs is not None and source_steps is not None:
        raise ValueError("specify epochs or diagnostic steps, not both")
    if recipe is None and source_epochs is None and source_steps is None:
        source_epochs = 4
    if budget_windows is None:
        budget_windows = getattr(source_windows, "budget_windows", len(source_windows))
    from .resampling import ResampledWindows
    from .schedule import exposure_checkpoints
    if isinstance(source_windows, ResampledWindows):
        if budget_windows != source_windows.pool_n:
            raise ValueError("Training N must equal the shared full-pool manifest N")
        if recipe is not None:
            boundaries = [(row["epoch"], row["step"]) for row in exposure_checkpoints(
                budget_windows, recipe.batch_size, observation_epochs(recipe.epochs))]
        else:
            if source_epochs is not None and (source_epochs <= 0 or source_epochs * 2 % 1):
                raise ValueError("Full-pool observation schedule requires positive half/integer epochs")
            epochs = np.arange(.5, source_epochs + .25, .5) if source_epochs is not None else []
            boundaries = [(row["epoch"], row["step"])
                          for row in exposure_checkpoints(budget_windows, batch_size, epochs)]
    elif recipe is not None and len(source_windows):
        boundaries = [(row["epoch"], row["step"]) for row in exposure_checkpoints(
            budget_windows, recipe.batch_size, observation_epochs(recipe.epochs))]
    else:
        boundaries = (epoch_checkpoints(budget_windows, batch_size, source_epochs)
                      if source_epochs is not None and len(source_windows) else [])
    steps = boundaries[-1][1] if boundaries else int(source_steps or 0)
    schedule_total = (recipe.steps_for(budget_windows, recipe.schedule_epochs)
                      if recipe is not None else None)
    if online_evaluation:
        def assess():
            metrics = evaluate(engine, test_windows, test_entities, context_length=context_length,
                               test_scales=test_scales)
            if original_mase is not None:
                metrics["relative_improvement_pct"] = 100 * (1 - metrics["mase"] / original_mase)
            return metrics
        if recipe is not None:
            curve = _online_curve_recipe(engine, source_windows, base_state=base_state,
                recipe=recipe, seed=seed, checkpoints=boundaries,
                schedule_total_steps=schedule_total, evaluate_fn=assess,
                resume_path=checkpoint_dir / "resume.pt",
                observations_path=checkpoint_dir / "curve.json",
                on_record=lambda row: print("EPOCH_EVALUATED", row["epoch"], row["metrics"], flush=True))
        else:
            from .calibration import online_curve
            curve = online_curve(engine, source_windows, base_state=base_state,
                learning_rate=learning_rate, seed=seed, checkpoints=boundaries, batch_size=batch_size,
                evaluate_fn=assess, resume_path=checkpoint_dir / "resume.pt",
                observations_path=checkpoint_dir / "curve.json",
                on_record=lambda row: print("EPOCH_EVALUATED", row["epoch"], row["metrics"], flush=True))
        records = curve["curve"]
        metrics = dict(records[-1]["metrics"])
        curve["train"].update(target_epochs=source_epochs, budget_windows=budget_windows,
                              actual_epochs=curve["train"]["sample_exposures"] / budget_windows)
        metrics.update(train={"source": curve["train"]}, checkpoints=records, seed=seed,
                       final_epoch=records[-1]["epoch"], evaluation_elapsed_s=curve["evaluation_seconds"],
                       checkpoint_policy="rolling_latest_evaluated_optimizer_state")
        if recipe is not None:
            metrics["training_recipe"] = recipe.as_dict()
        if hasattr(source_windows, "coverage"):
            metrics["selection_coverage"] = source_windows.coverage()
        return metrics
    snapshots = {}
    records = []

    def save_snapshot(epoch, step):
        if checkpoint_dir is None:
            snapshots[epoch] = {k: v.detach().cpu().clone()
                                for k, v in model.state_dict().items()}
            path = None
        else:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            filename = ("epoch-%02d.pt" % epoch if isinstance(epoch, int)
                        else "epoch-%.3f.pt" % epoch)
            path = checkpoint_dir / filename
            torch.save(model.state_dict(), path)
            snapshots[epoch] = path
            print(f"checkpoint saved: {path} (step {step})", flush=True)
        records.append({"epoch": epoch, "step": step,
                        "sample_exposures": step * batch_size,
                        "actual_epoch": step * batch_size / budget_windows,
                        "checkpoint": str(path) if path else None,
                        "training_elapsed_s": time.time() - started})

    def save_epoch(step):
        for epoch, boundary in boundaries:
            if step == boundary:
                save_snapshot(epoch, step)

    started = time.time()
    model.load_state_dict(base_state, strict=True)
    try:
        training_started = time.monotonic()
        source_log = fine_tune(engine, source_windows, steps=steps,
                               batch_size=batch_size, learning_rate=learning_rate,
                               seed=seed, on_step=save_epoch, max_seconds=training_seconds,
                               resume_path=(checkpoint_dir / "resume.pt" if checkpoint_dir is not None
                                            and isinstance(source_windows, ResampledWindows) else None),
                               checkpoint_steps=[step for _, step in boundaries],
                               recipe=recipe, schedule_total_steps=schedule_total)
        completed = source_log["steps"]
        if checkpoint_dir is not None and source_log.get("resumed_from_step", 0):
            for epoch, step in boundaries:
                if step > completed or epoch in snapshots:
                    continue
                filename = ("epoch-%02d.pt" % epoch if isinstance(epoch, int) else "epoch-%.3f.pt" % epoch)
                path = checkpoint_dir / filename
                if not path.exists():
                    raise FileNotFoundError(f"Missing pre-resume epoch checkpoint: {path}")
                snapshots[epoch] = path
                records.append(dict(epoch=epoch, step=step, checkpoint=str(path),
                    sample_exposures=step * batch_size, actual_epoch=step * batch_size / budget_windows,
                    training_elapsed_s=None))
            records.sort(key=lambda record: record["epoch"])
        if training_seconds is not None and completed and not any(
                record["step"] == completed for record in records):
            save_snapshot(completed * batch_size / budget_windows, completed)
        training_elapsed = time.monotonic() - training_started
        evaluation_started = time.monotonic()
        for record in records:
            snapshot = snapshots[record["epoch"]]
            state = (torch.load(snapshot, map_location=engine.device, weights_only=True)
                     if isinstance(snapshot, Path) else snapshot)
            model.load_state_dict(state, strict=True)
            model.eval()
            record["metrics"] = evaluate(
                engine, test_windows, test_entities, context_length=context_length,
                test_scales=test_scales)
        metrics = (dict(records[-1]["metrics"]) if records else
                   evaluate(engine, test_windows, test_entities,
                            context_length=context_length, test_scales=test_scales))
        source_log.update(target_epochs=source_epochs,
                          epochs=(records[-1]["epoch"] if records else 0),
                          budget_windows=budget_windows,
                          selected_windows=(None if hasattr(source_windows, "regions") or
                                            isinstance(source_windows, ResampledWindows) else len(source_windows)),
                          legal_window_occurrences=len(source_windows),
                          sample_exposures=completed * batch_size)
        metrics.update(train={"source": source_log}, seed=seed,
                       training_elapsed_s=training_elapsed,
                       evaluation_elapsed_s=time.monotonic() - evaluation_started,
                       elapsed_s=round(time.time() - started, 1),
                       checkpoints=records, final_epoch=(records[-1]["epoch"] if records else 0),
                       checkpoint_policy="final_epoch_not_best_test",
                       forecast_protocol=getattr(engine, "forecast_protocol", "autoregressive_greedy_prefix_bin_cdf"))
        if recipe is not None:
            metrics["training_recipe"] = recipe.as_dict()
        if hasattr(source_windows, "coverage"):
            metrics["selection_coverage"] = source_windows.coverage()
        return metrics
    finally:
        model.load_state_dict(base_state, strict=True)
        model.eval()


__all__ = [
    "crb_gradient_computer",
    "fine_tune",
    "evaluate",
    "run_arm_training",
    "TrainingRecipe",
    "observation_epochs",
]
