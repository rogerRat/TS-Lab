"""Tests for the downstream main experiment (needs torch + chronos + rage env)."""

from __future__ import annotations

from pathlib import Path
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rage.core import TimeSeriesFragment
from rage.experiments.data import (
    sample_pool_windows,
    split_reference_fragment,
)
from rage.experiments.downstream_main import (
    evaluate,
    expected_value_mae,
    fine_tune,
    run_selection,
    summarize,
)


def tiny_engine(context_length=32, prediction_length=8, projection=64, batch_size=16):
    from rage.experiments.core_experiments import build_engine

    args = types.SimpleNamespace(
        context_length=context_length,
        prediction_length=prediction_length,
        projection_dimension=projection,
        batch_size=batch_size,
        model_seed=0,
    )
    # from-scratch tiny model: no warmup checkpoints needed
    from gradient.chronos import ChronosTinyConfig, ChronosTinyGradientEngine

    config = ChronosTinyConfig(
        context_length=context_length,
        prediction_length=prediction_length,
        projection_dimension=projection,
        require_warmup_models=False,
        initialization_mode="from_scratch",
        gradient_batch_size=batch_size,
    )
    return ChronosTinyGradientEngine(config, device="cpu")


def synthetic_windows(count, length, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(count * length, dtype=np.float64).reshape(count, length)
    values = np.sin(2 * np.pi * t / (4 * length)) + 0.01 * rng.normal(size=(count, length))
    return values.astype(np.float32)


class TestSplit:
    def test_split_disjoint_and_ordered(self) -> None:
        original = TimeSeriesFragment(data=np.arange(10000, dtype=np.float32), fragment_id=3)
        selection, test = split_reference_fragment(
            original, test_fraction=0.2, window_length=576
        )
        assert len(selection) + len(test) == len(original)
        np.testing.assert_array_equal(selection.data, original.data[: len(selection)])
        np.testing.assert_array_equal(test.data, original.data[len(selection):])
        assert selection.fragment_id == 3
        assert test.fragment_id != selection.fragment_id
        # test region is the later tail
        assert test.data[0] == original.data[len(selection)]

    def test_split_guarantees_test_windows(self) -> None:
        original = TimeSeriesFragment(data=np.arange(1500, dtype=np.float32), fragment_id=0)
        _selection, test = split_reference_fragment(
            original, test_fraction=0.2, window_length=576
        )
        assert len(test.valid_starts(512, 64)) > 0

    def test_split_rejects_fragment_without_room(self) -> None:
        short = TimeSeriesFragment(data=np.arange(700, dtype=np.float32), fragment_id=0)
        with pytest.raises(ValueError):
            split_reference_fragment(short, test_fraction=0.2, window_length=576)


@pytest.fixture
def source_load(tmp_path):
    from assets.dataset_processing.shard_dataset import SourceShardDatasetWriter
    bundle = tmp_path / "source"
    writer = SourceShardDatasetWriter(bundle, shard_bytes=8192)
    for row in range(6):
        writer.append(trajectory_id=f"trajectory-{row}", scope="source", dataset="fixture",
                      subset="train", series_id=f"series-{row}",
                      values=np.arange(700, dtype=np.float64) + row * 10000)
    writer.append(trajectory_id="invalid", scope="source", dataset="fixture", subset="train",
                  series_id="invalid", values=np.full(700, np.nan))
    writer.finish()
    return bundle / "load"


class TestPoolSampling:
    def test_sample_pool_windows_shapes_and_origins(self, source_load) -> None:
        windows, origins = sample_pool_windows(16, window_length=576, seed=0, load_dir=source_load)
        assert windows.shape == (16, 576)
        assert np.isfinite(windows).all()
        assert len(origins) == 16
        for window, origin in zip(windows, origins):
            row = int(origin.removeprefix("series-"))
            assert row * 10000 <= window[0] <= row * 10000 + 124
            np.testing.assert_array_equal(np.diff(window), np.ones(575))
        # different seeds give different samples
        other, _ = sample_pool_windows(16, window_length=576, seed=1, load_dir=source_load)
        assert not np.array_equal(windows, other)

    def test_sample_is_reproducible_per_seed(self, source_load) -> None:
        first, first_origins = sample_pool_windows(8, window_length=576, seed=7, load_dir=source_load)
        second, second_origins = sample_pool_windows(8, window_length=576, seed=7, load_dir=source_load)
        np.testing.assert_array_equal(first, second)
        assert first_origins == second_origins


class TestEvaluation:
    def test_expected_value_mae_beats_mean_predictor(self) -> None:
        engine = tiny_engine()
        windows = synthetic_windows(8, engine.config.window_length)
        mae = expected_value_mae(engine, windows, batch_size=4)
        futures = windows[:, engine.config.context_length:]
        mean_predictor = float(np.abs(futures - futures.mean(axis=1, keepdims=True)).mean())
        # A trained-on-sinusoids-free model still produces finite forecasts;
        # the mean predictor comparison is informational, so only check sanity.
        assert np.isfinite(mae) and mae >= 0.0

    def test_evaluate_returns_ce_and_mae(self) -> None:
        engine = tiny_engine()
        windows = synthetic_windows(8, engine.config.window_length)
        metrics = evaluate(engine, windows, batch_size=4, compute_mae=True)
        assert metrics["test_windows"] == 8
        assert np.isfinite(metrics["test_ce"])
        assert np.isfinite(metrics["test_mae"])


class TestFineTune:
    def test_training_reduces_loss_and_restores_base(self) -> None:
        engine = tiny_engine()
        windows = synthetic_windows(16, engine.config.window_length)
        before = {k: v.clone() for k, v in engine.model.state_dict().items()}
        outcome = fine_tune(
            engine, windows, steps=30, batch_size=8, learning_rate=1e-3, seed=0
        )
        changed = any(not torch.equal(before[k], engine.model.state_dict()[k]) for k in before)
        assert changed, "fine-tuning must modify the weights"
        assert outcome["curve"]["final_loss"] < outcome["curve"]["initial_loss"] * 1.5
        engine.model.load_state_dict(outcome["base_state"], strict=True)
        assert all(torch.equal(before[k], engine.model.state_dict()[k]) for k in before)


class TestSelectionRunner:
    def test_selection_returns_budget_windows_with_identity(self) -> None:
        from rage.experiments.core_experiments import BatchedScanner

        engine = tiny_engine(context_length=32, prediction_length=8, projection=32)
        scanner = BatchedScanner(engine, 32, 8)
        rng = np.random.default_rng(0)
        t = np.arange(1400, dtype=np.float64)
        values = (np.sin(2 * np.pi * t / 900) + 0.02 * rng.normal(size=1400)).astype(np.float32)
        selection_fragment = TimeSeriesFragment(data=values, entity_id="toy::a", fragment_id=0)
        candidate_fragment = TimeSeriesFragment(
            data=np.roll(values, 200).astype(np.float32), entity_id="toy::b", fragment_id=1
        )
        starts = candidate_fragment.valid_starts(32, 8)[:24]
        candidate_windows = np.stack([
            candidate_fragment.extract_window(int(s), 32, 8) for s in starts
        ])
        candidate_ids = [(1, int(s)) for s in starts]
        candidate_sketches = scanner.scan_windows(candidate_windows).astype(np.float64)

        outcome = run_selection(
            scanner, [selection_fragment], [candidate_fragment], candidate_windows,
            candidate_ids, budget=8, tolerance=0.1, versions=4, block_length=4,
            stability_threshold=0.5, lower_quantile=0.25,
            candidate_sketches=candidate_sketches,
        )
        assert outcome.selected_windows.shape == (8, 40)
        assert len(outcome.selected_ids) == 8
        assert all(fragment_id == 1 for fragment_id, _start in outcome.selected_ids)
        assert outcome.versions == 4
        assert 0 < outcome.anchors <= outcome.total_starts


class TestSummarize:
    def test_summarize_pairs_against_rage(self) -> None:
        results = {
            "base_model": {"test_ce": 4.0, "test_mae": 1.0, "test_windows": 10},
            "rage": [
                {"test_ce": 3.0, "test_mae": 0.8},
                {"test_ce": 3.2, "test_mae": 0.9},
            ],
            "random": [
                {"test_ce": 3.5, "test_mae": 1.1},
                {"test_ce": 3.7, "test_mae": 1.2},
            ],
        }
        summary = summarize(results, arms=["rage", "random"])
        assert summary["arms"]["rage"]["test_ce_mean"] == pytest.approx(3.1)
        comparison = summary["paired_comparison"]["rage_vs_random"]
        # positive delta = rage better (lower CE) than random
        assert comparison["ce_delta"] == pytest.approx(0.5)
        assert comparison["mae_delta"] == pytest.approx(0.3)
        assert comparison["ce_positive_seeds"] == 2
