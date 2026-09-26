"""End-to-end CRB block selection, training, and true-history evaluation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from rags.core import TimeSeriesFragment
from rags.crb import CRBConfig, CRBPipeline, CRBRunReport, GradientBlock, top_unique_candidate_blocks
from .core_experiments import BatchedScanner, build_engine
from .data import (
    build_candidate_fragments,
    build_reference_fragments,
    reference_load_dirs,
    split_reference_fragment,
)
from .downstream_main import evaluate, fine_tune, summarize


def _gradient_computer(scanner: BatchedScanner):
    def compute(fragment: TimeSeriesFragment, starts: np.ndarray) -> np.ndarray:
        return scanner.scan_starts(fragment, starts)
    return compute


def _windows_for_blocks(blocks: list[GradientBlock], fragments: list[TimeSeriesFragment],
                        context_length: int, prediction_length: int) -> np.ndarray:
    by_id = {fragment.fragment_id: fragment for fragment in fragments}
    rows = [
        by_id[block.fragment_id].extract_window(int(start), context_length, prediction_length)
        for block in blocks
        for start in block.start_positions
    ]
    if not rows:
        raise RuntimeError("block selection produced no trainable windows")
    return np.stack(rows).astype(np.float32, copy=False)


def _evaluation_windows(fragments: list[TimeSeriesFragment], context_length: int,
                        prediction_length: int, stride: int,
                        evaluation_mode: str) -> np.ndarray:
    """Build ground-truth contexts for one latest origin or every rolling origin."""
    rows = []
    for fragment in fragments:
        starts = fragment.valid_starts(context_length, prediction_length, stride)
        if evaluation_mode == "single_window":
            starts = starts[-1:]
        elif evaluation_mode != "rolling":
            raise ValueError("evaluation_mode must be single_window or rolling")
        rows.extend(fragment.extract_window(int(start), context_length, prediction_length)
                    for start in starts)
    if not rows:
        raise RuntimeError("evaluation split contains no legal windows")
    return np.stack(rows).astype(np.float32, copy=False)


def _diagnostics(report: CRBRunReport) -> dict:
    def side(results):
        return {
            "trajectories": len(results),
            "legal_positions": sum(len(result.valid_starts) for result in results),
            "backward_count": sum(result.computed_gradients for result in results),
            "blocks": sum(len(result.blocks) for result in results),
            "discarded_no_signal": sum(result.discarded_no_signal for result in results),
            "discarded_antagonistic": sum(result.discarded_antagonistic for result in results),
            "banned_budget": sum(result.banned_budget for result in results),
        }
    return {"reference": side(report.reference_results), "candidate": side(report.candidate_results)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["ETT_1H"])
    parser.add_argument("--series-cap", type=int, default=2)
    parser.add_argument("--slice-length", type=int, default=2048)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--candidate-series", type=int, default=8)
    parser.add_argument("--candidate-slice", type=int, default=2048)
    parser.add_argument("--block-budget", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=64)
    parser.add_argument("--window-stride", type=int, default=4)
    parser.add_argument("--cover-density", type=float, default=0.02)
    parser.add_argument("--error-density", type=float, default=0.01)
    parser.add_argument("--split-density", type=float, default=0.005)
    parser.add_argument("--mean-epsilon", type=float, default=1e-8)
    parser.add_argument("--endpoint-epsilon", type=float, default=1e-8)
    parser.add_argument("--curvature-cap", type=float, default=1.0)
    parser.add_argument("--projection-dimension", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--train-batch", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--evaluation-mode", choices=["single_window", "rolling"],
                        default="single_window")
    parser.add_argument("--out", default="outputs/crb_main")
    args = parser.parse_args(argv)
    if args.block_budget <= 0:
        parser.error("--block-budget must be positive")

    started = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    window_length = args.context_length + args.prediction_length

    full_references = build_reference_fragments(
        reference_load_dirs(keywords=args.datasets), window_length=window_length,
        slice_length=args.slice_length, series_cap=args.series_cap,
    )
    pairs = [split_reference_fragment(fragment, test_fraction=args.test_fraction,
                                      window_length=window_length)
             for fragment in full_references]
    references = [pair[0] for pair in pairs]
    test_fragments = [pair[1] for pair in pairs]
    candidates = build_candidate_fragments(
        window_length=window_length, slice_length=args.candidate_slice,
        num_series=args.candidate_series, seed=0,
    )
    test_windows = _evaluation_windows(
        test_fragments, args.context_length, args.prediction_length,
        args.window_stride, args.evaluation_mode,
    )

    engine = build_engine(args)
    scanner = BatchedScanner(engine, args.context_length, args.prediction_length, log_every=10000)
    config = CRBConfig(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        window_stride=args.window_stride,
        cover_density=args.cover_density,
        error_density=args.error_density,
        split_density=args.split_density,
        curvature_cap=args.curvature_cap,
        mean_epsilon=args.mean_epsilon,
        endpoint_epsilon=args.endpoint_epsilon,
    )
    report = CRBPipeline(config, _gradient_computer(scanner)).run(
        references, candidates, block_budget=args.block_budget,
    )
    selected_blocks = top_unique_candidate_blocks(report, args.block_budget)
    if len(selected_blocks) < args.block_budget:
        raise RuntimeError("only %d unique validated blocks available for budget %d" %
                           (len(selected_blocks), args.block_budget))
    crb_windows = _windows_for_blocks(
        selected_blocks, candidates, args.context_length, args.prediction_length,
    )
    rng = np.random.default_rng(1729)
    random_indices = rng.choice(len(report.candidate_blocks), size=args.block_budget, replace=False)
    random_blocks = [report.candidate_blocks[int(index)] for index in random_indices]
    random_windows = _windows_for_blocks(
        random_blocks, candidates, args.context_length, args.prediction_length,
    )

    results = {
        "protocol": vars(args),
        "crb_diagnostics": _diagnostics(report),
        "selected": {
            "blocks": len(selected_blocks),
            "train_windows": len(crb_windows),
            "random_train_windows": len(random_windows),
        },
    }
    results["base_model"] = evaluate(engine, test_windows)
    arms = {"crb": crb_windows, "random_blocks": random_windows}
    for arm_name, windows in arms.items():
        records = []
        for seed in args.seeds:
            outcome = fine_tune(
                engine, windows, steps=args.train_steps, batch_size=args.train_batch,
                learning_rate=args.learning_rate, seed=seed,
                eval_windows=test_windows, eval_every=args.eval_every,
            )
            metrics = evaluate(engine, test_windows)
            metrics.update({"train": outcome["curve"], "eval_curve": outcome["eval_curve"], "seed": seed})
            records.append(metrics)
            engine.model.load_state_dict(outcome["base_state"], strict=True)
            engine.model.eval()
        results[arm_name] = records
    # ``summarize`` predates CRB and uses ``rags`` as the focal arm name.
    compat = {**results, "rags": results["crb"]}
    summary = summarize(compat, arms=["rags", "random_blocks"])
    summary["arms"]["crb"] = summary["arms"].pop("rags")
    summary["paired_comparison"]["crb_vs_random_blocks"] = (
        summary["paired_comparison"].pop("rags_vs_random_blocks")
    )
    results["summary"] = summary
    results["elapsed_minutes"] = round((time.time() - started) / 60.0, 2)
    destination = out_dir / "crb_main.json"
    destination.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results["summary"], indent=2), flush=True)
    print("CRB MAIN DONE: %s" % destination, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
