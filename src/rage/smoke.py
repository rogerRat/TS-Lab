"""Smoke run: RAGE pipeline with the real ChronosTiny gradient engine.

Builds a from-scratch ChronosTiny model (no downloads), synthesises
reference and candidate fragments with a transient regime change, and runs
the full two-stage pipeline. Prints a compact report; writes nothing.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def build_engine(context_length: int, prediction_length: int, projection_dimension: int, device: str):
    from gradient.chronos import ChronosTinyConfig, ChronosTinyGradientEngine

    config = ChronosTinyConfig(
        context_length=context_length,
        prediction_length=prediction_length,
        projection_dimension=projection_dimension,
        projection_repeats=1,
        token_chunk_size=8,
        gradient_batch_size=8,
        require_warmup_models=False,
        initialization_mode="from_scratch",
    )
    return ChronosTinyGradientEngine(config, device=device)


def synthetic_fragments(window_length: int, extra: int, seed: int):
    from rage.core import TimeSeriesFragment

    rng = np.random.default_rng(seed)
    length = window_length + extra

    def make(seed_i: int, regime_shift: float, spike_center: float | None) -> TimeSeriesFragment:
        local = np.random.default_rng(seed_i)
        t = np.arange(length, dtype=np.float64)
        slow = np.sin(2 * np.pi * t / (3 * length))
        values = 0.5 * slow + 0.02 * local.normal(size=length)
        values[int(len(values) * regime_shift):] += 1.0
        if spike_center is not None:
            center = int(len(values) * spike_center)
            values += 2.0 * np.exp(-((np.arange(length) - center) / 20.0) ** 2)
        return TimeSeriesFragment(data=values.astype(np.float32), fragment_id=seed_i)

    reference = [make(seed + 1, 0.5, 0.55), make(seed + 2, 0.0, None)]
    candidates = [make(seed + 3, 0.4, None), make(seed + 4, 0.6, 0.5), make(seed + 5, 0.2, None)]
    return reference, candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=64)
    parser.add_argument("--extra", type=int, default=40, help="valid starts per fragment = extra + 1")
    parser.add_argument("--projection-dimension", type=int, default=256)
    parser.add_argument("--versions", type=int, default=5)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    from rage.bridge import ChronosGradientComputer
    from rage.core import RAGEConfig
    from rage.pipeline import RAGEPipeline

    window_length = args.context_length + args.prediction_length
    engine = build_engine(
        args.context_length, args.prediction_length, args.projection_dimension, args.device
    )
    print(f"engine ready: layer={engine.layer_name} fast_path={engine.fast_lm_head} device={engine.device}")

    computer = ChronosGradientComputer(engine)
    reference, candidates = synthetic_fragments(window_length, args.extra, seed=0)

    config = RAGEConfig(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        direction_tolerance=0.05,
        magnitude_tolerance=0.05,
        min_interval_size=4,
        block_length=8,
        num_reference_versions=args.versions,
        selection_budget=args.budget,
        ranking_budget=args.budget,
        stability_threshold=0.4,
        lower_quantile=0.25,
        projection_dimension=args.projection_dimension,
    )
    report = RAGEPipeline(config, computer).run(reference, candidates)

    print(f"reference anchors: {report.reference_backward_count}/{report.total_start_positions} starts")
    print(f"||G|| = {report.reference_gradient_norm:.4f}")
    print(f"version norms: min={report.version_norms.min():.4f} max={report.version_norms.max():.4f}")
    print(f"candidates scored: {len(report.candidate_scores)}")
    print(f"selected: {len(report.selection.selected)} coverage={report.selection.coverage:.2f} "
          f"backfilled={report.selection.num_backfilled}")
    top = report.selection.selected[:5]
    for candidate in top:
        print(f"  fragment={candidate.fragment_id} start={candidate.position} "
              f"pi={candidate.stability_frequency:.2f} q={candidate.lower_quantile_score:.4f}")
    for result in report.quadrature_results:
        print(f"fragment {result.fragment_index}: {result.computed_gradients} anchors "
              f"over {result.total_positions} starts (compression {result.compression_ratio:.1f}x)")

    assert np.isfinite(report.estimated_reference_gradient).all()
    assert report.reference_backward_count < report.total_start_positions
    assert len(report.selection.selected) > 0
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
