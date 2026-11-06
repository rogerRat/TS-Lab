"""Main-table experiment: does selected data improve forecast accuracy?

Eight arms per dataset and three students (experiment decision 9):

  original          pretrained Chronos-T5-small, no training;
  random            online uniform replacement draws from the feasible pool;
  trajectory_level  one gradient per disjoint 576-point segment (F2);
  rage           CRB reconstruction + cancellation detection + block recall;
  less / tsds / lp_mirror / ltsv   paper baselines (adapted, disclosed).

Single-stage schedule (D1/D2): four epochs of fine-tuning on the selected
windows, then direct evaluation on the FEV-native exam; three training
seeds; MASE with full-history seasonal scales is the sole comparison metric
(per-series means, equal-weight dataset average).

The run is resumable: every finished (dataset, arm, seed) cell is written to
``cells/`` immediately and skipped on restart; selections are cached under
``selection/`` and pool-side artefacts (CRB blocks, LESS features, TSDS
embeddings, LP-Mirror proxy, LTSV scores) under ``cache/``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

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
from baselines.trajectory import REVISION as LANGUAGE_BASELINE_REVISION
from .tasks import build_ratio_pool, build_task_data, build_task_specs
from .time_bench import build_time_task, build_time_task_specs, verify_time_checkout
from .quito_bench import build_quito_task, build_quito_task_specs
from .train_eval import crb_gradient_computer
from .students import STUDENTS, deploy_students, key_values, atomic_json
from .selection_metrics import SelectionMetrics
from rage.execution import selection_execution, clear_execution_cache

BASELINE_ARMS = ("less", "tsds", "lp_mirror", "ltsv")
GRANULARITY_ARMS = ("trajectory_level",)
CORE_ARMS = ("original", "random", "rage")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="subset of dataset names (default: the six main datasets)")
    parser.add_argument("--benchmark", choices=["fev", "time", "quito"], default="fev",
                        help="task source: FEV, TIME or QuitoBench")
    parser.add_argument("--quito-root", default=None,
                        help="Quito snapshot root on the data disk")
    parser.add_argument("--term", choices=["short", "medium", "long"],
                        default="short",
                        help="TIME term whose prediction_length shapes the exam "
                             "(TIME benchmark only)")
    parser.add_argument("--series-cap", type=int, default=None,
                        help="cap on reference series per dataset (default: all, A3)")
    parser.add_argument("--n-windows", type=int, default=4096,
                        help="N: selected windows (native external baselines may overlap)")
    parser.add_argument("--pool-equivalents", type=int, default=20480,
                        help="pool capacity in non-overlapping window "
                             "equivalents (A5: 5N)")
    parser.add_argument("--pool-seed", type=int, default=0)
    parser.add_argument("--pool-stride", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=64)
    parser.add_argument("--projection-dimension", type=int, default=4096)
    parser.add_argument("--selection-gradient-batch", type=int, default=0)
    parser.add_argument("--selection-inference-batch", type=int, default=0)
    parser.add_argument("--selection-memory-fraction", type=float, default=0.7)
    parser.add_argument("--selection-cpu-threads", type=int, default=4)
    parser.add_argument("--selection-generation", choices=["eager", "cuda_graph"], default="cuda_graph")
    parser.add_argument("--cover-density", type=float, default=0.02)
    parser.add_argument("--error-density", type=float, default=0.02)
    parser.add_argument("--split-density", type=float, default=0.005)
    parser.add_argument("--block-budget", type=int, default=8)
    parser.add_argument("--mean-epsilon", type=float, default=1e-8,
                        help="epsilon_mu (C3: calibrated p10, frozen)")
    parser.add_argument("--endpoint-epsilon", type=float, default=1e-8,
                        help="epsilon_g (C3: same calibrated value)")
    parser.add_argument("--curvature-cap", type=float, default=1.0)
    parser.add_argument("--source-epochs", type=int, default=4,
                        help="D2: save/evaluate every epoch, use the final epoch")
    parser.add_argument("--train-batch", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--with-baselines", nargs="*", default=list(BASELINE_ARMS),
                        choices=list(BASELINE_ARMS),
                        help="baseline arms to enable in this run")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--students", nargs="+", choices=STUDENTS, default=list(STUDENTS))
    parser.add_argument("--student-environments", nargs="*", default=[], help="student=conda-env or student=/path/to/python")
    parser.add_argument("--student-model-paths", nargs="*", default=[], help="student=/local/checkpoint")
    parser.add_argument("--student-parallelism", type=int, default=1)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--out", default="outputs/main_table")
    return parser.parse_args(argv)


def _cell_path(out_dir: Path, dataset: str, arm: str, seed: int) -> Path:
    return out_dir / "cells" / ("%s__%s__seed%d.json" % (dataset, arm, seed))


def _selection_path(out_dir: Path, dataset: str, arm: str) -> Path:
    return out_dir / "selection" / ("%s__%s.npz" % (dataset, arm))


def _save_selection(path: Path, arm_out: ArmOutput) -> None:
    from .regions import save_selection
    save_selection(path, arm_out)


def _load_selection(path: Path) -> ArmOutput:
    from .regions import load_selection
    return load_selection(path)


def check_baseline_resume(out_dir, args):
    """One protocol check: old-method cells/selected sets need a fresh run."""
    path = out_dir / "protocol.json"
    if not path.exists():
        return
    from .regions import REVISION
    previous = json.loads(path.read_text())["protocol"]
    if previous.get("delivery_revision") != REVISION:
        raise ValueError("Delivery protocol changed; use a new --out run directory.")
    keys = ("language_baseline_revision", "cover_density", "error_density", "split_density", "random_sampling_revision")
    current = {**vars(args), "language_baseline_revision": LANGUAGE_BASELINE_REVISION,
               "random_sampling_revision": RANDOM_SAMPLING_REVISION}
    if any(previous.get(key) != current.get(key) for key in keys):
        raise ValueError("Baseline method or Random/sampling protocol changed; use a new --out run directory. "
                         "Old selected sets and student cells are not compatible.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.random_sampling_revision = RANDOM_SAMPLING_REVISION
    out_dir = Path(args.out)
    check_baseline_resume(out_dir, args)
    args.language_baseline_revision = LANGUAGE_BASELINE_REVISION
    from .regions import REVISION
    args.delivery_revision = REVISION
    (out_dir / "cells").mkdir(parents=True, exist_ok=True)
    (out_dir / "selection").mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    started = time.time()

    if args.benchmark == "quito":
        specs = build_quito_task_specs(args.datasets, root=args.quito_root)
    elif args.benchmark == "time":
        verify_time_checkout()
        specs = build_time_task_specs(args.datasets)
    else:
        specs = build_task_specs(series_cap=None)
    if args.datasets:
        wanted = set(args.datasets)
        specs = [spec for spec in specs if spec.dataset in wanted]
        if len(specs) != len(wanted):
            raise SystemExit("unknown dataset names: %s" % (
                sorted(wanted - {spec.dataset for spec in specs})))

    pool, pool_manifest = build_ratio_pool(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        equivalents=args.pool_equivalents, seed=args.pool_seed,
        cache_dir=cache_dir)
    index = WindowIndex(pool, context_length=args.context_length,
                        prediction_length=args.prediction_length,
                        stride=args.pool_stride)
    print("pool: %d trajectories, %d points, %d window equivalents, %d legal "
          "windows; N=%d" % (len(pool), pool_manifest["total_points"],
                             pool_manifest["total_equivalents"], len(index),
                             args.n_windows), flush=True)

    engine_config = ChronosT5Config(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        projection_dimension=args.projection_dimension,
        vmap_batch_size=args.selection_gradient_batch,
        inference_batch_size=args.selection_inference_batch,
        selection_memory_fraction=args.selection_memory_fraction,
        selection_cpu_threads=args.selection_cpu_threads,
        selection_generation=args.selection_generation,
    )
    if args.model_path:
        engine_config = replace(engine_config, model_path=args.model_path)
    engine = ChronosT5Engine(engine_config, device="cuda")
    print("engine ready: %d params on %s" % (engine.parameter_count, engine.device),
          flush=True)

    results = {"protocol": vars(args),
               "pool": {k: v for k, v in pool_manifest.items() if k != "rows"},
               "pool_size": len(index), "n_select": args.n_windows,
               "datasets": [spec.dataset for spec in specs]}
    (out_dir / "protocol.json").write_text(json.dumps(results, indent=2))
    selection_metrics = SelectionMetrics(out_dir, device=engine.device,
                                         protocol=results)

    baseline_names = list(args.with_baselines)
    arm_names = CORE_ARMS + GRANULARITY_ARMS + tuple(baseline_names)
    pool_block_cache = {}  # One immutable source pool for all tasks in this invocation.
    for spec in specs:
        dataset_started = time.time()
        if args.benchmark == "quito":
            task = build_quito_task(spec, context_length=args.context_length,
                                    prediction_length=args.prediction_length,
                                    series_cap=args.series_cap, pool_fragments=pool)
        elif args.benchmark == "time":
            task = build_time_task(spec, term=args.term,
                                   context_length=args.context_length,
                                   prediction_length=args.prediction_length,
                                   series_cap=args.series_cap,
                                   pool_fragments=pool)
        else:
            task = build_task_data(
                spec,
                context_length=args.context_length,
                prediction_length=args.prediction_length,
                series_cap=args.series_cap,
                pool_fragments=pool,
            )
        print("[%s] %d reference series, %d test windows, %d constant series" % (
            spec.dataset, len(task.reference_fragments), len(task.test_windows),
            len(task.constant_series)), flush=True)

        # ---- selection arms --------------------------------------------------
        arms: dict[str, ArmOutput] = {}
        selection_metrics.original(spec.dataset)

        def cached(arm: str, build) -> ArmOutput:
            path = _selection_path(out_dir, spec.dataset, arm)
            if path.exists():
                arm_out = selection_metrics.measure(
                    spec.dataset, arm, lambda: _load_selection(path), selection_cached=True)
            else:
                with selection_execution(engine):
                    arm_out = selection_metrics.measure(spec.dataset, arm, build)
                _save_selection(path, arm_out)
            arms[arm] = arm_out
            print("[%s] %-16s selection: %d legal window occurrences (cost %.1fs, %d gradients)" % (
                spec.dataset, arm, len(arm_out.selected),
                arm_out.cost.wall_seconds, arm_out.cost.gradient_computations),
                flush=True)
            return arm_out

        cached("rage", lambda: rage_crb_select(
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
            n_select=args.n_windows,
            mean_epsilon=args.mean_epsilon,
            endpoint_epsilon=args.endpoint_epsilon,
            tie_seed=5,
            curvature_cap=args.curvature_cap,
            pool_cache_dir=cache_dir,
            pool_manifest=pool_manifest,
            pool_block_cache=pool_block_cache,
        ))
        cached("random", lambda: streaming_random_select(
            index, n_select=args.n_windows, seed=101))
        for arm in GRANULARITY_ARMS:
            cached(arm, lambda: trajectory_level_select(
                task.reference_fragments, pool, index,
                engine=engine,
                n_select=args.n_windows,
                context_length=args.context_length,
                prediction_length=args.prediction_length,
                cover_density=args.cover_density, error_density=args.error_density,
                split_density=args.split_density, cache_dir=cache_dir, pool_manifest=pool_manifest,
            ))
        for name in baseline_names:
            cached(name, lambda n=name: run_baseline(
                n, engine=engine,
                reference_fragments=task.reference_fragments,
                pool=pool, index=index,
                context_length=args.context_length,
                prediction_length=args.prediction_length,
                n_select=args.n_windows, seed=7,
                cache_dir=cache_dir,
                pool_manifest=pool_manifest,
                cover_density=args.cover_density,
                error_density=args.error_density,
                split_density=args.split_density,
                dataset_tag=spec.dataset,
            ))

        # Selection/exam artifacts are written once; students cannot select.
        task_dir = out_dir / "tasks"
        task_dir.mkdir(exist_ok=True)
        exam_path = task_dir / (spec.dataset + ".npz")
        if not exam_path.exists():
            np.savez_compressed(exam_path, windows=task.test_windows,
                                entities=np.asarray(task.test_entities), scales=task.test_scales)
        model_paths = key_values(args.student_model_paths)
        if args.model_path:
            model_paths.setdefault("chronos-small", args.model_path)
        manifest = {
            "schema": "rage.shared-selection.v2", "delivery_revision": REVISION, "dataset": spec.dataset,
            "probe": "chronos-small", "context_length": args.context_length,
            "prediction_length": args.prediction_length,
            "model_paths": model_paths,
            "training": {"epochs": args.source_epochs, "batch_size": args.train_batch,
                         "budget_windows": args.n_windows,
                         "learning_rate": args.learning_rate, "weight_decay": .01,
                         "gradient_clipping": None, "dtype": "float32", "seeds": args.seeds},
            "exam": {"path": str(exam_path.resolve())},
            "selections": {arm: {"path": str(_selection_path(out_dir, spec.dataset, arm).resolve()),
                                  "cost": vars(output.cost)} for arm, output in arms.items()},
        }
        manifest_path = task_dir / (spec.dataset + ".json")
        atomic_json(manifest_path, manifest)
        if not args.selection_only:
            import torch
            clear_execution_cache(engine)
            engine.model.cpu()
            torch.cuda.empty_cache()
            deploy_students(args.students, manifest_path.resolve(), out_dir.resolve(),
                            environments=key_values(args.student_environments),
                            parallelism=args.student_parallelism)
            engine.model.to(engine.device)

        print("[%s] done in %.1f min" % (spec.dataset,
                                         (time.time() - dataset_started) / 60), flush=True)

    if not args.selection_only:
        from .student_report import summarize_students
        summarize_students(out_dir, specs, args.seeds, arm_names, args.students)
    print("MAIN TABLE DONE in %.1f min -> %s" % ((time.time() - started) / 60, out_dir),
          flush=True)
    return 0


def summarize_run(out_dir: Path, specs, seeds: list[int], arm_names: list[str]) -> dict:
    """Aggregate cells: per-dataset means, then equal-weight dataset average."""
    summary: dict[str, dict] = {}
    for arm in arm_names:
        per_dataset: dict[str, dict] = {}
        for spec in specs:
            cells_dir = out_dir / "cells"
            records = []
            for seed in seeds:
                path = cells_dir / ("%s__%s__seed%d.json" % (spec.dataset, arm, seed))
                if path.exists():
                    records.append(json.loads(path.read_text()))
            if not records:
                continue
            per_dataset[spec.dataset] = {
                "mase_mean": float(np.mean([r["mase"] for r in records])),
                "mase_std": float(np.std([r["mase"] for r in records])),
                "seeds": len(records),
            }
        if per_dataset:
            summary[arm] = {
                "per_dataset": per_dataset,
                "mase_avg": float(np.mean([v["mase_mean"] for v in per_dataset.values()])),
            }
    # relative improvement over random
    if "random" in summary and "rage" in summary:
        common = set(summary["random"]["per_dataset"]) & set(summary["rage"]["per_dataset"])
        summary["rage_vs_random"] = {
            dataset: {
                "mase_reduction": 1.0 - summary["rage"]["per_dataset"][dataset]["mase_mean"]
                / summary["random"]["per_dataset"][dataset]["mase_mean"],
            }
            for dataset in common
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    from .checkpoint_report import write_checkpoint_report
    write_checkpoint_report(out_dir)
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
