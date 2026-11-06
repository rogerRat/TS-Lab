"""2026-09-15 production training-recipe regressions (handoff main-experiment-20260915).

Covers the frozen per-model recipe from configs/rage/shared_tasks_10.json:
micro-batch accumulation equivalence, schedule/stop decoupling (including the
fractional TimesFM stop epoch 2.625), bias/norm weight-decay exclusion, and
interrupted-resume equivalence under the recipe path.  All cases run on a tiny
CPU engine; no GPU and no real model weights are required.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rage.experiments.train_eval import (TrainingRecipe, _build_optimizer,
                                         fine_tune, observation_epochs,
                                         run_arm_training)


class Model(torch.nn.Module):
    """Same shape contract as the probing tests: C=2, H=2, vocab 3."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)

    def forward(self, input_ids, **kwargs):
        logits = self.linear(input_ids.float())[:, None, :].expand(-1, 2, -1)
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids, **kwargs):
        assert "labels" not in kwargs
        assert input_ids.shape[1] == 2
        logits = self(input_ids).logits
        return SimpleNamespace(scores=tuple(logits[:, i] for i in range(2)),
                               sequences=torch.cat([torch.zeros(len(logits), 1, dtype=torch.long),
                                                    logits.argmax(-1)], dim=1))


class NormModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.norm = torch.nn.LayerNorm(4)


def make_engine(seed=42):
    torch.manual_seed(seed)
    return SimpleNamespace(model=Model(), device="cpu",
        config=SimpleNamespace(context_length=2, prediction_length=2),
        batch_tensors=lambda w: {"input_ids": torch.tensor(w[:, :2]),
                                 "labels": torch.tensor(w[:, 2:]).long() % 3},
        tokenizer=SimpleNamespace(
            context_input_transform=lambda x: (x, torch.ones_like(x), torch.ones(len(x))),
            output_transform=lambda tokens, scale: (tokens.float() + 10) * scale[:, None, None]))


def make_windows(n=64, seed=7, scale=1.0):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 5, size=(n, 4)).astype(np.float32)) * scale


def base_recipe(**overrides):
    defaults = dict(learning_rate=1e-3, epochs=1.0, batch_size=8, micro_batch_size=8,
                    weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8, max_grad_norm=1.0,
                    warmup_ratio=0.0, schedule="linear", min_lr_ratio=0.0,
                    schedule_epochs=6.0, exclude_bias_norm=False)
    defaults.update(overrides)
    return TrainingRecipe(**defaults)


def snapshot(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


TIMEMOE_SETTINGS = {
    "batch_size": 64, "context_length": 512, "epochs": [1], "learning_rates": [0.0001],
    "optimizer": {"betas": [0.9, 0.95], "eps": 1e-08, "exclude_bias_norm": True,
                  "max_grad_norm": 1, "micro_batch_size": 16, "min_lr_ratio": 0.5,
                  "schedule": "cosine", "warmup_ratio": 0},
    "prediction_length": 64, "schedule_epochs": 6, "seeds": [0, 1, 2],
    "training_data": "safe_history_random", "weight_decay": 0.1,
}

TIMESFM_SETTINGS = {
    "batch_size": 32, "context_length": 512, "epochs": [2.625], "learning_rates": [3e-05],
    "optimizer": {"betas": [0.9, 0.999], "eps": 1e-08, "max_grad_norm": 1,
                  "micro_batch_size": 32, "min_lr_ratio": 0, "schedule": "cosine",
                  "warmup_ratio": 0},
    "prediction_length": 64, "schedule_epochs": 6, "seeds": [0, 1, 2],
    "training_data": "safe_history_random", "weight_decay": 0.01,
}


def test_recipe_from_settings_parses_shared_tasks_blocks():
    timemoe = TrainingRecipe.from_settings(TIMEMOE_SETTINGS)
    assert timemoe.accumulation == 4
    assert (timemoe.learning_rate, timemoe.epochs, timemoe.batch_size) == (1e-4, 1.0, 64)
    assert timemoe.betas == (0.9, 0.95) and timemoe.weight_decay == 0.1
    assert timemoe.schedule == "cosine" and timemoe.min_lr_ratio == 0.5
    assert timemoe.exclude_bias_norm and timemoe.schedule_epochs == 6.0
    timesfm = TrainingRecipe.from_settings(TIMESFM_SETTINGS)
    assert timesfm.epochs == 2.625 and timesfm.accumulation == 1
    assert timesfm.min_lr_ratio == 0.0 and timesfm.schedule == "cosine"


def test_recipe_validation_rejects_bad_ratios_and_micro_batch():
    with pytest.raises(ValueError):
        base_recipe(batch_size=8, micro_batch_size=3)
    with pytest.raises(ValueError):
        base_recipe(schedule="step")
    with pytest.raises(ValueError):
        base_recipe(min_lr_ratio=1.5)
    with pytest.raises(ValueError):
        base_recipe(epochs=0)


def test_lr_multiplier_schedule_and_stop_decoupling():
    total = 24  # 6-epoch schedule for N=32, batch=8
    linear = base_recipe(min_lr_ratio=0.0)
    assert linear.lr_multiplier(0, total) == pytest.approx(1.0)
    assert linear.lr_multiplier(total // 2, total) == pytest.approx(0.5)
    # Stopping at epoch 2 (step 8 of 24) must sit on the 6-epoch trajectory.
    assert linear.lr_multiplier(8, total) == pytest.approx(1.0 - 8 / total)
    assert linear.lr_multiplier(total + 5, total) == pytest.approx(0.0)
    cosine = base_recipe(schedule="cosine", min_lr_ratio=0.5)
    assert cosine.lr_multiplier(0, total) == pytest.approx(1.0)
    assert cosine.lr_multiplier(total, total) == pytest.approx(0.5)
    assert cosine.lr_multiplier(total // 2, total) == pytest.approx(0.75)
    warm = base_recipe(warmup_ratio=0.25)
    assert warm.lr_multiplier(0, total) == pytest.approx(1 / 6)
    assert warm.lr_multiplier(5, total) == pytest.approx(1.0)
    values = [warm.lr_multiplier(step, total) for step in range(total)]
    assert values.index(max(values)) == 5


def test_observation_epochs_and_fractional_step_accounting():
    assert observation_epochs(1.0) == [0.5, 1.0]
    assert observation_epochs(2.0) == [0.5, 1.0, 1.5, 2.0]
    assert observation_epochs(2.625) == [0.5, 1.0, 1.5, 2.0, 2.5, 2.625]
    timesfm = TrainingRecipe.from_settings(TIMESFM_SETTINGS)
    assert timesfm.steps_for(37) == int(np.ceil(2.625 * 37 / 32))
    assert timesfm.steps_for(37, timesfm.schedule_epochs) == int(np.ceil(6.0 * 37 / 32))


def test_micro_batch_accumulation_matches_effective_batch():
    windows = make_windows(64, scale=50.0)  # large grads exercise clipping path
    full = base_recipe(micro_batch_size=8)
    split = base_recipe(micro_batch_size=4)
    engine_a, engine_b = make_engine(), make_engine()
    log_a = fine_tune(engine_a, windows, steps=5, batch_size=8, learning_rate=1e-3,
                      seed=3, recipe=full, schedule_total_steps=5)
    log_b = fine_tune(engine_b, windows, steps=5, batch_size=8, learning_rate=1e-3,
                      seed=3, recipe=split, schedule_total_steps=5)
    for (name, a), (name_b, b) in zip(snapshot(engine_a.model).items(),
                                      snapshot(engine_b.model).items()):
        assert name == name_b
        assert torch.allclose(a, b, atol=1e-6), name
    assert log_a["final_loss"] == pytest.approx(log_b["final_loss"], rel=1e-5)


def test_clip_grad_norm_matches_manual_reference():
    windows = make_windows(64, scale=50.0)
    recipe = base_recipe()
    engine = make_engine()
    log = fine_tune(engine, windows, steps=1, batch_size=8, learning_rate=1e-3,
                    seed=11, recipe=recipe, schedule_total_steps=6)
    assert log["steps"] == 1
    # Manual reference: same frozen sample draw, same loss, clip at 1.0, AdamW.
    reference = make_engine()
    model, optimizer = reference.model, _build_optimizer(reference.model, recipe)
    rng = np.random.default_rng(11)
    indices = rng.integers(0, len(windows), size=8)
    from rage.engines import per_sample_cross_entropy
    tensors = reference.batch_tensors(windows[indices])
    loss = per_sample_cross_entropy(model(**tensors).logits, tensors["labels"]).mean()
    loss.backward()
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert float(total_norm) > 1.0, "fixture should actually exercise clipping"
    for group in optimizer.param_groups:
        group["lr"] *= recipe.lr_multiplier(0, 6)
    optimizer.step()
    for (name, a), (_, b) in zip(snapshot(engine.model).items(),
                                 snapshot(model).items()):
        assert torch.allclose(a, b, atol=1e-6), name


def test_exclude_bias_norm_optimizer_groups():
    recipe = base_recipe(exclude_bias_norm=True, weight_decay=0.1, betas=(0.9, 0.95))
    model = NormModel()
    optimizer = _build_optimizer(model, recipe)
    assert len(optimizer.param_groups) == 2
    decay, no_decay = optimizer.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    decay_names = {n for n, p in model.named_parameters() if id(p) in map(id, decay["params"])}
    no_decay_names = {n for n, p in model.named_parameters() if id(p) in map(id, no_decay["params"])}
    assert decay_names == {"linear.weight"}
    assert no_decay_names == {"linear.bias", "norm.weight", "norm.bias"}
    assert optimizer.defaults["betas"] == (0.9, 0.95)
    assert optimizer.defaults["eps"] == 1e-8


def test_interrupted_resume_reproduces_uninterrupted_run(tmp_path):
    windows = make_windows(48)
    recipe = base_recipe(epochs=1.0)
    uninterrupted = make_engine()
    fine_tune(uninterrupted, windows, steps=6, batch_size=8, learning_rate=1e-3,
              seed=5, recipe=recipe, schedule_total_steps=6)
    expected = snapshot(uninterrupted.model)

    def crash_after_checkpoint(step):
        if step == 5:
            raise RuntimeError("simulated crash after the step-3 checkpoint")

    crashed = make_engine()
    with pytest.raises(RuntimeError):
        fine_tune(crashed, windows, steps=6, batch_size=8, learning_rate=1e-3,
                  seed=5, recipe=recipe, schedule_total_steps=6,
                  resume_path=tmp_path / "resume.pt", checkpoint_steps=[3],
                  on_step=crash_after_checkpoint)
    resumed = make_engine()
    log = fine_tune(resumed, windows, steps=6, batch_size=8, learning_rate=1e-3,
                    seed=5, recipe=recipe, schedule_total_steps=6,
                    resume_path=tmp_path / "resume.pt", checkpoint_steps=[6])
    assert log["resumed_from_step"] == 3
    for (name, a), (_, b) in zip(snapshot(resumed.model).items(), expected.items()):
        assert torch.allclose(a, b, atol=1e-7), name


def test_run_arm_training_fractional_stop_epochs_and_boundaries(tmp_path):
    engine = make_engine()
    base = snapshot(engine.model)
    recipe = base_recipe(epochs=2.625, batch_size=8, micro_batch_size=4,
                         schedule="cosine", min_lr_ratio=0.5, learning_rate=1e-3,
                         schedule_epochs=6.0)
    metrics = run_arm_training(engine, source_windows=make_windows(37),
                               batch_size=8, learning_rate=1e-3, seed=0,
                               test_windows=make_windows(4, seed=9),
                               test_entities=["a"] * 4, context_length=2,
                               test_scales=np.ones(4),
                               checkpoint_dir=tmp_path, recipe=recipe)
    epochs = [record["epoch"] for record in metrics["checkpoints"]]
    assert epochs == [0.5, 1.0, 1.5, 2.0, 2.5, 2.625]
    assert metrics["final_epoch"] == 2.625
    assert metrics["train"]["source"]["steps"] == int(np.ceil(2.625 * 37 / 8))
    assert (tmp_path / "epoch-2.625.pt").exists()
    assert metrics["training_recipe"]["epochs"] == 2.625
    # The shared snapshot is restored whatever happened during the cell.
    for (name, a), (_, b) in zip(snapshot(engine.model).items(), base.items()):
        assert torch.equal(a, b), name


def test_run_arm_training_rejects_conflicting_arguments():
    engine = make_engine()
    recipe = base_recipe()
    with pytest.raises(ValueError):
        run_arm_training(engine, source_windows=make_windows(8), source_steps=3,
                         batch_size=8, learning_rate=1e-3, seed=0,
                         test_windows=make_windows(4), test_entities=["a"] * 4,
                         context_length=2, recipe=recipe)
    with pytest.raises(ValueError):
        run_arm_training(engine, source_windows=make_windows(8),
                         batch_size=4, learning_rate=1e-3, seed=0,
                         test_windows=make_windows(4), test_entities=["a"] * 4,
                         context_length=2, recipe=recipe)
    with pytest.raises(ValueError):
        run_arm_training(engine, source_windows=make_windows(8), source_epochs=1.5,
                         batch_size=8, learning_rate=1e-3, seed=0,
                         test_windows=make_windows(4), test_entities=["a"] * 4,
                         context_length=2, recipe=recipe)
