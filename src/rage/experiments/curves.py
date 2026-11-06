"""Supplementary curves for the main experiment (experiment plan §2.2).

1. data-curve: N in {1024, 2048, 4096, 8192} = {5%, 10%, 20%, 40%} of the
   pool capacity (5N = 20480 window equivalents), one four-epoch schedule.
   Every epoch checkpoint is saved and evaluated. Every arm's selection is
   computed once at the largest N and smaller points reuse the same ranked
   list as prefixes (deployment order), so ordering effects cannot
   contaminate the comparison across N. Random always samples the whole
   feasible pool with replacement; N changes only its training budget.
2. time-curve: every arm receives the same total GPU budget; selection wall
   time is charged to the arm and the remainder is spent on single-stage
   training with a time-bounded loop.

Arms: rage, random, plus the trajectory-level granularity arm on the data
curve per §2.2. Cells are resumable.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from rage.engines import ChronosT5Config, ChronosT5Engine

from .arms import (
    ArmOutput,
    SelectionCost,
    WindowIndex,
    trajectory_level_select,
    streaming_random_select,
    RANDOM_SAMPLING_REVISION,
    rage_crb_select,
)
from baselines.runners import run_baseline
from .tasks import build_ratio_pool, build_task_data, build_task_specs
from .train_eval import crb_gradient_computer, evaluate, run_arm_training
from .schedule import epoch_checkpoints

ARMS = ("rage", "random", "trajectory_level",
        "less", "tsds", "lp_mirror", "ltsv")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("data-curve", "time-curve"))
    parser.add_argument("--dataset", default="ETT_1H")
    parser.add_argument("--arms", nargs="+", default=["rage", "random"],
                        choices=ARMS)
    parser.add_argument("--n-values", nargs="+", type=int,
                        default=[1024, 2048, 4096, 8192],
                        help="data-curve N points (absolute window counts)")
    parser.add_argument("--n-windows", type=int, default=4096,
                        help="main-table N (equal-epoch reference point)")
    parser.add_argument("--total-budget-seconds", type=float, default=1800.0,
                        help="time-curve: per-arm total GPU budget")
    parser.add_argument("--series-cap", type=int, default=None)
    parser.add_argument("--pool-equivalents", type=int, default=20480)
    parser.add_argument("--pool-seed", type=int, default=0)
    parser.add_argument("--pool-stride", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=64)
    parser.add_argument("--projection-dimension", type=int, default=4096)
    parser.add_argument("--cover-density", type=float, default=0.02)
    parser.add_argument("--error-density", type=float, default=0.02)
    parser.add_argument("--split-density", type=float, default=0.005)
    parser.add_argument("--block-budget", type=int, default=8)
    parser.add_argument("--mean-epsilon", type=float, default=1e-8)
    parser.add_argument("--endpoint-epsilon", type=float, default=1e-8)
    parser.add_argument("--curvature-cap", type=float, default=1.0)
    parser.add_argument("--source-epochs", type=int, default=4)
    parser.add_argument("--train-batch", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--out", default=None)
    return parser.parse_args(argv)


def fine_tune_time_bounded(engine, windows, *, max_seconds, batch_size,
                           learning_rate, seed, max_steps):
    """AdamW fine-tuning bounded by wall-clock seconds and a step ceiling."""
    from rage.engines import per_sample_cross_entropy

    model = engine.model
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=0.01)
    rng = np.random.default_rng(seed)
    windows = np.asarray(windows, dtype=np.float32)
    if len(windows) == 0 or max_seconds <= 0 or max_steps <= 0:
        return {"steps": 0, "final_loss": None}
    model.train()
    final_loss: torch.Tensor | None = None
    started = time.time()
    step = 0
    while step < max_steps:
        # Synchronize only at the coarse timing boundary. The old scalar
        # conversion synchronized every step, making this timing experiment
        # measure host/device stalls rather than useful training work.
        if step % 10 == 0:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if time.time() - started >= max_seconds:
                break
        indices = rng.integers(0, len(windows), size=min(batch_size, len(windows)))
        tensors = engine.batch_tensors(windows[indices])
        optimizer.zero_grad(set_to_none=True)
        output = model(**tensors, return_dict=True)
        loss = per_sample_cross_entropy(output.logits, tensors["labels"]).mean()
        loss.backward()
        optimizer.step()
        final_loss = loss.detach()
        step += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model.eval()
    return {"steps": step,
            "final_loss": float(final_loss.cpu()) if final_loss is not None else None,
            "seconds": round(time.time() - started, 1)}


def _load_or_compute_selection(out_dir: Path, arm: str, *, engine, task, pool,
                               index, pool_manifest, args, n_max: int):
    path = out_dir / "selection" / ("%s.npz" % arm)
    from .regions import load_selection, save_selection
    if path.exists():
        return load_selection(path)
    started = time.time()
    if arm == "rage":
        out = rage_crb_select(
            task.reference_fragments, pool, index,
            gradient_computer=crb_gradient_computer(
                engine, args.context_length, args.prediction_length),
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            window_stride=args.pool_stride,
            cover_density=args.cover_density,
            error_density=args.error_density,
            split_density=args.split_density,
            block_budget=args.block_budget,
            n_select=n_max,           # keep the full ranked (deployment) list
            mean_epsilon=args.mean_epsilon,
            endpoint_epsilon=args.endpoint_epsilon,
            tie_seed=5,
            curvature_cap=args.curvature_cap,
            pool_cache_dir=out_dir / "cache",
            pool_manifest=pool_manifest,
        )
    elif arm == "random":
        out = streaming_random_select(index, n_select=n_max, seed=101)
    elif arm == "trajectory_level":
        out = trajectory_level_select(
            task.reference_fragments, pool, index, engine=engine,
            n_select=n_max,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            cover_density=args.cover_density, error_density=args.error_density,
            split_density=args.split_density, cache_dir=out_dir / "cache", pool_manifest=pool_manifest,
        )
    else:
        out = run_baseline(
            arm, engine=engine, reference_fragments=task.reference_fragments,
            pool=pool, index=index,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            n_select=n_max, seed=7,
            cache_dir=out_dir / "cache", pool_manifest=pool_manifest,
            cover_density=args.cover_density,
            error_density=args.error_density,
            split_density=args.split_density,
            dataset_tag=args.dataset,
        )
    out.cost.wall_seconds = round(time.time() - started, 1)
    save_selection(path, out)
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.random_sampling_revision = RANDOM_SAMPLING_REVISION
    out_dir = Path(args.out or ("outputs/curve_%s" % args.mode.replace("-", "_")))
    from .regions import check_run_revision, REVISION, RegionWindows
    check_run_revision(out_dir, args)
    protocol_path = out_dir / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()).get("random_sampling_revision") != RANDOM_SAMPLING_REVISION:
        raise ValueError("Random sampling changed; use a new run directory")
    (out_dir / "cells").mkdir(parents=True, exist_ok=True)
    started = time.time()

    specs = build_task_specs()
    specs = [spec for spec in specs if spec.dataset == args.dataset]
    if not specs:
        raise SystemExit("unknown dataset: %s" % args.dataset)
    spec = specs[0]

    pool, pool_manifest = build_ratio_pool(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        equivalents=args.pool_equivalents, seed=args.pool_seed,
        cache_dir=out_dir / "cache")
    index = WindowIndex(pool, context_length=args.context_length,
                        prediction_length=args.prediction_length,
                        stride=args.pool_stride)
    n_max = max(args.n_values) if args.mode == "data-curve" else args.n_windows

    engine_config = ChronosT5Config(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        projection_dimension=args.projection_dimension,
    )
    if args.model_path:
        engine_config = replace(engine_config, model_path=args.model_path)
    engine = ChronosT5Engine(engine_config, device="cuda")
    base_state = {name: value.detach().clone()
                  for name, value in engine.model.state_dict().items()}

    protocol = vars(args)
    protocol["delivery_revision"] = REVISION
    protocol["pool_size"] = len(index)
    protocol["out"] = str(out_dir)
    protocol["pool"] = {k: v for k, v in pool_manifest.items() if k != "rows"}
    (out_dir / "protocol.json").write_text(json.dumps(protocol, indent=2))

    task = build_task_data(
        spec,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        series_cap=args.series_cap,
        pool_fragments=pool,
    )
    print("[%s] %d reference series, %d test windows" % (
        spec.dataset, len(task.reference_fragments), len(task.test_windows)),
        flush=True)

    for arm in args.arms:
        arm_out = _load_or_compute_selection(
            out_dir, arm, engine=engine, task=task, pool=pool,
            index=index, pool_manifest=pool_manifest, args=args, n_max=n_max)
        print("[%s] %s selection ready: %d legal window occurrences (cost %.1fs)" % (
            spec.dataset, arm, len(arm_out.selected),
            arm_out.cost.wall_seconds), flush=True)

        points = ([(n, "epoch") for n in sorted(args.n_values)]
                  if args.mode == "data-curve"
                  else [(args.n_windows, "time")])

        for n, variant in points:
            for seed in args.seeds:
                if args.mode == "data-curve":
                    tag = "n%04d-%s" % (n, variant)
                    cell = out_dir / "cells" / ("%s__%s__%s__seed%d.json" % (
                        spec.dataset, arm, tag, seed))
                    if cell.exists():
                        continue
                    metrics = run_arm_training(
                        engine,
                        source_windows=(arm_out.selected if arm == "random" else
                                        arm_out.selected.prefix(n) if isinstance(arm_out.selected, RegionWindows)
                                        else arm_out.selected[:n]),
                        source_epochs=args.source_epochs,
                        budget_windows=n,
                        checkpoint_dir=out_dir / "checkpoints" / arm / tag / ("seed%d" % seed),
                        batch_size=args.train_batch,
                        learning_rate=args.learning_rate,
                        seed=seed,
                        test_windows=task.test_windows,
                        test_entities=task.test_entities,
                        context_length=args.context_length,
                        test_scales=task.test_scales,
                        base_state=base_state,
                    )
                    metrics["n_windows"] = n
                    metrics["schedule_variant"] = variant
                    metrics["source_steps_used"] = metrics["train"]["source"]["steps"]
                else:
                    tag = "time"
                    cell = out_dir / "cells" / ("%s__%s__%s__seed%d.json" % (
                        spec.dataset, arm, tag, seed))
                    if cell.exists():
                        continue
                    model = engine.model
                    cell_started = time.time()
                    remaining = max(0.0, args.total_budget_seconds
                                    - arm_out.cost.wall_seconds)
                    metrics = run_arm_training(
                        engine, source_windows=arm_out.selected,
                        source_epochs=args.source_epochs, training_seconds=remaining,
                        budget_windows=n,
                        checkpoint_dir=out_dir / "checkpoints" / arm / "time" / ("seed%d" % seed),
                        batch_size=args.train_batch,
                        learning_rate=args.learning_rate, seed=seed,
                        test_windows=task.test_windows, test_entities=task.test_entities,
                        context_length=args.context_length, test_scales=task.test_scales,
                        base_state=base_state)
                    metrics["seed"] = seed
                    metrics["elapsed_s"] = round(time.time() - cell_started, 1)
                    metrics["selection_seconds"] = arm_out.cost.wall_seconds
                    metrics["total_seconds"] = (arm_out.cost.wall_seconds
                                                + metrics["elapsed_s"])
                    model.load_state_dict(base_state, strict=True)
                    model.eval()
                metrics["arm"] = arm
                metrics["dataset"] = spec.dataset
                metrics["selection_cost"] = vars(arm_out.cost)
                if not isinstance(arm_out.selected, RegionWindows):
                    from .deploy import selection_coverage
                    metrics["selection_coverage"] = selection_coverage(
                        arm_out.window_ids[:n], args.context_length + args.prediction_length)
                cell.write_text(json.dumps(metrics, indent=2))
                print("[%s] %-16s %s n=%d seed=%d MASE %.4f [%.0fs]" % (
                    spec.dataset, arm, tag, n, seed, metrics["mase"],
                    metrics["elapsed_s"]), flush=True)

    from .checkpoint_report import write_checkpoint_report
    write_checkpoint_report(out_dir)
    print("%s DONE in %.1f min -> %s" % (args.mode.upper(),
                                         (time.time() - started) / 60, out_dir),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
