"""Ablation experiment: where do the gains come from (experiment plan §5)?

Seven variants plus a 2x2 grid (uniform/error-driven x none/standard
detection) at a fixed total budget, on the development dataset ETT_1H with
the main-table pool, N, schedule, and seeds:

  full            complete method (reference point);
  random_cover    keep endpoints, random interior cover anchors;
  no_error        error-stage budget spent by longest-interval bisection;
  no_split_detect detection and splitting removed, split budget -> error stage;
  drop_only       flagged intervals dropped, split budget -> error stage;
  fixed_blocks    same reconstructed gradients, equal-length blocks with the
                  block count matched to the anchor-interval count, detection
                  reapplied from in-block mean / endpoint signals;
  window_return   same reconstructed gradients, windows as index/return units.

Variants that share a CRB configuration (fixed_blocks / window_return reuse
the full method's anchors) share the pool-side CRB results through an
in-process cache.  Each cell is resumable and written to ``cells/``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from rags.core import TimeSeriesFragment
from rags.crb import CRBConfig, CRBEngine, CRBEngineMulti, CRBResult
from rags.engines import ChronosT5Config, ChronosT5Engine

from .arms import ArmOutput, SelectionCost, WindowIndex
from .deploy import cut_disjoint
from .main_table import summarize_run
from .tasks import build_ratio_pool, build_task_data, build_task_specs
from .train_eval import crb_gradient_computer, run_arm_training

VARIANTS = (
    "full", "random_cover", "no_error", "no_split_detect", "drop_only",
    "fixed_blocks", "window_return", "uniform_segments",
)
GRID_VARIANTS = ("grid_unif_nodet", "grid_unif_det", "grid_risk_nodet")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ETT_1H")
    parser.add_argument("--series-cap", type=int, default=None)
    parser.add_argument("--n-windows", type=int, default=4096,
                        help="N: non-overlapping training windows (A6)")
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
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS),
                        choices=VARIANTS + GRID_VARIANTS)
    parser.add_argument("--with-grid", action="store_true",
                        help="add the 2x2 uniform/error x detect/nodet grid")
    parser.add_argument("--window-entry-budget", type=int, default=None,
                        help="window-return comparison: B returned entries "
                             "(default: same as --block-budget)")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--out", default="outputs/ablation")
    return parser.parse_args(argv)


def variant_config(base: CRBConfig, variant: str) -> CRBConfig:
    """Config override for one ablation variant (budgets stay additive)."""
    if variant in ("full", "fixed_blocks", "window_return"):
        return base
    if variant == "grid_risk_nodet":
        return replace(base, detection="none")
    if variant == "grid_unif_nodet":
        return replace(base, error_placement="uniform", detection="none")
    if variant == "grid_unif_det":
        return replace(base, error_placement="uniform")
    if variant == "random_cover":
        return replace(base, initial_placement="random")
    if variant == "no_error":
        return replace(base, error_placement="uniform")
    if variant == "no_split_detect":
        # split budget reassigned to the error stage, detection disabled
        return replace(base, detection="none",
                       error_density=base.error_density + base.split_density,
                       split_density=0.0)
    if variant == "drop_only":
        return replace(base, detection="drop_only",
                       error_density=base.error_density + base.split_density,
                       split_density=0.0)
    raise ValueError("unknown variant %s" % variant)


def reconstruct_gradients(result: CRBResult) -> np.ndarray:
    """Linear interpolation of anchor gradients over valid positions [T, d].

    Anchor keys are 0-based row indices into ``valid_starts``; interpolation
    happens in that row space, not in physical start coordinates.
    """
    keys = np.asarray(sorted(result.anchors), dtype=np.int64)
    anchor_matrix = np.stack([result.anchors[int(k)] for k in keys])
    rows = np.arange(len(result.valid_starts), dtype=np.int64)
    index = np.searchsorted(keys, rows)
    left = np.clip(index - 1, 0, len(keys) - 1)
    right = np.clip(index, 0, len(keys) - 1)
    span = (keys[right] - keys[left]).astype(np.float64)
    alpha = np.where(span > 0,
                     (rows - keys[left]) / np.maximum(span, 1.0),
                     0.0)
    alpha = np.clip(alpha, 0.0, 1.0)[:, None]
    return ((1.0 - alpha) * anchor_matrix[left]
            + alpha * anchor_matrix[right]).astype(np.float32)


@dataclass
class RecallUnit:
    """One recall unit: window positions sharing a unit direction."""

    fragment_id: int
    starts: np.ndarray          # 0-based window positions (index row space)
    direction: np.ndarray


def equal_length_units(result: CRBResult, *, mean_epsilon: float,
                       endpoint_epsilon: float) -> tuple[list[RecallUnit], int]:
    """Fixed-length re-blocking of the reconstructed gradients.

    Block count matches the anchor-interval count; the representative is the
    in-block mean of reconstructed gradients and detection uses the same
    mean/endpoint-weakness rule on reconstructed endpoint gradients.  Equal-
    length blocks carry no split budget, so both no-signal and antagonistic
    blocks are dropped.
    """
    gradients = reconstruct_gradients(result)
    total = len(result.valid_starts)
    anchor_intervals = max(len(result.anchors) - 1, 1)
    n_blocks = min(anchor_intervals, total)
    edges = np.unique(np.linspace(0, total, n_blocks + 1).round().astype(int))
    units: list[RecallUnit] = []
    dropped = 0
    for k in range(len(edges) - 1):
        lo, hi = int(edges[k]), int(edges[k + 1])
        if hi <= lo:
            continue
        block = gradients[lo:hi]
        mean = block.mean(axis=0)
        mean_norm = float(np.linalg.norm(mean))
        if mean_norm < mean_epsilon:
            left_weak = np.linalg.norm(gradients[lo]) < endpoint_epsilon
            right_weak = np.linalg.norm(gradients[hi - 1]) < endpoint_epsilon
            dropped += 1
            continue
        units.append(RecallUnit(result.fragment_id,
                                np.arange(lo, hi, dtype=np.int64),
                                (mean / max(mean_norm, 1e-8)).astype(np.float32)))
    return units, dropped


def window_units(result: CRBResult, *, mean_epsilon: float) -> list[RecallUnit]:
    """One recall unit per window, from the same reconstructed gradients."""
    gradients = reconstruct_gradients(result)
    norms = np.linalg.norm(gradients, axis=1)
    keep = np.nonzero(norms >= mean_epsilon)[0]
    return [RecallUnit(result.fragment_id,
                       np.asarray([int(row)], dtype=np.int64),
                       (gradients[row] / norms[row]).astype(np.float32))
            for row in keep]


def _streamed_top_b(reference_matrix: np.ndarray,
                    candidate_units: list[RecallUnit], take: int,
                    chunk: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Per-query top-B candidate-unit indices over the full candidate set."""
    queries = len(reference_matrix)
    best_values = np.full((queries, take), -np.inf, dtype=np.float32)
    best_indices = np.full((queries, take), -1, dtype=np.int64)
    for lo in range(0, len(candidate_units), chunk):
        hi = min(lo + chunk, len(candidate_units))
        candidate_matrix = np.stack([u.direction for u in candidate_units[lo:hi]])
        scores = reference_matrix @ candidate_matrix.T
        merged_values = np.concatenate([best_values, scores], axis=1)
        merged_indices = np.concatenate(
            [best_indices, np.arange(lo, hi, dtype=np.int64)[None, :].repeat(queries, 0)],
            axis=1)
        order = np.argsort(-merged_values, axis=1, kind="stable")[:, :take]
        best_values = np.take_along_axis(merged_values, order, axis=1)
        best_indices = np.take_along_axis(merged_indices, order, axis=1)
    return best_values, best_indices


def adjacent_duplicate_fraction(index: WindowIndex,
                                positions: np.ndarray) -> float:
    """Share of adjacent returned windows on one trajectory overlapping >=90%.

    Windows are ordered by position within each fragment; consecutive pairs
    with input+target interval overlap ratio >= 0.9 count as duplicates
    (threshold pre-fixed at 90% per experiment plan §5).
    """
    if len(positions) == 0:
        return 0.0
    window_length = index.context_length + index.prediction_length
    fids, starts = index.fragment_arrays(positions)
    duplicates = 0
    pairs = 0
    order = np.lexsort((starts, fids))
    fids, starts = fids[order], starts[order]
    for i in range(len(fids) - 1):
        if fids[i] != fids[i + 1]:
            continue
        pairs += 1
        delta = abs(int(starts[i + 1]) - int(starts[i]))
        if 1.0 - delta / window_length >= 0.9:
            duplicates += 1
    return duplicates / max(pairs, 1)


def select_uniform_segments(reference_fragments, pool, index, *, gradient_computer, args):
    """Historical 576-point segment-head scorer, now explicitly an ablation."""
    from .regions import deliver_regions
    width = args.context_length + args.prediction_length

    def gradients(fragments):
        regions, values = [], []
        for fragment in fragments:
            starts = np.arange(1, fragment.length - width + 2, width)
            if len(starts):
                values.extend(gradient_computer(fragment, starts[lo:lo + 256])
                              for lo in range(0, len(starts), 256))
                regions.extend((fragment.fragment_id, int(s - 1), int(s - 1 + width)) for s in starts)
        return regions, np.concatenate(values)

    _, reference = gradients(reference_fragments)
    regions, source = gradients(pool)
    aggregate = reference.mean(axis=0)
    scores = source / np.maximum(np.linalg.norm(source, axis=1, keepdims=True), 1e-12) @ (
        aggregate / max(float(np.linalg.norm(aggregate)), 1e-12))
    order = np.lexsort((np.random.default_rng(5).random(len(scores)), -scores))
    selected = deliver_regions(index, np.asarray(regions)[order], n=args.n_windows)
    return ArmOutput("uniform_segments", selected, [], SelectionCost(
        gradient_computations=len(source) + len(reference), index_bytes=source.nbytes),
        {"granularity": "uniform_576_segments", "delivery": selected.coverage(),
         "adaptation": "historical segment-head estimate; no adaptive allocation or detection"})


def select_variant(
    variant: str,
    reference_fragments: list[TimeSeriesFragment],
    pool: list[TimeSeriesFragment],
    index: WindowIndex,
    *,
    gradient_computer,
    args,
    crb_cache: dict | None = None,
) -> ArmOutput:
    """Shared selection path: CRB anchors, then variant-specific recall units.

    ``crb_cache`` maps a config key to (reference_results, candidate_results)
    so variants sharing a configuration reuse the same anchors.
    """
    if variant == "uniform_segments":
        return select_uniform_segments(reference_fragments, pool, index,
                                       gradient_computer=gradient_computer, args=args)
    base = CRBConfig(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        window_stride=args.pool_stride,
        cover_density=args.cover_density,
        error_density=args.error_density,
        split_density=args.split_density,
        mean_epsilon=args.mean_epsilon,
        endpoint_epsilon=args.endpoint_epsilon,
        curvature_cap=args.curvature_cap,
    )
    config = variant_config(base, variant)
    cache_key = (config.cover_density, config.error_density, config.split_density,
                 config.initial_placement, config.error_placement, config.detection)
    engine = CRBEngine(config, gradient_computer)
    if crb_cache is not None and cache_key in crb_cache:
        reference_results, candidate_results = crb_cache[cache_key]
    else:
        multi = CRBEngineMulti(config, gradient_computer)
        reference_results = multi.process_fragments(reference_fragments)
        candidate_results = multi.process_fragments(pool)
        if crb_cache is not None:
            crb_cache[cache_key] = (reference_results, candidate_results)

    def units_for(results: list[CRBResult], *, source=False) -> tuple[list[RecallUnit], int]:
        dropped = 0
        units: list[RecallUnit] = []
        if variant == "fixed_blocks":
            for result in results:
                frag_units, frag_dropped = equal_length_units(
                    result, mean_epsilon=args.mean_epsilon,
                    endpoint_epsilon=args.endpoint_epsilon)
                units.extend(frag_units)
                if source:
                    offset = index.bounds(result.fragment_id)[0]
                    for unit in frag_units:
                        unit.starts = unit.starts + offset
                dropped += frag_dropped
        elif variant == "window_return":
            for result in results:
                frag_units = window_units(result, mean_epsilon=args.mean_epsilon)
                if source:
                    offset = index.bounds(result.fragment_id)[0]
                    for unit in frag_units:
                        unit.starts = unit.starts + offset
                units.extend(frag_units)
        else:
            for result in results:
                for block in result.blocks:
                    if source:
                        lo = index.bounds(block.fragment_id)[0]
                        positions = lo + (block.start_positions - 1) // index.stride
                    else:
                        positions = block.start_positions.astype(np.int64)
                    units.append(RecallUnit(block.fragment_id, positions,
                                            block.direction))
                dropped += (result.discarded_no_signal
                            + result.discarded_antagonistic
                            + result.banned_budget)
        return units, dropped

    reference_units, _ref_dropped = units_for(reference_results)
    candidate_units, cand_dropped = units_for(candidate_results, source=True)
    if not reference_units or not candidate_units:
        raise RuntimeError("variant %s produced no recall units" % variant)

    entry_budget = (args.window_entry_budget if variant == "window_return"
                    and args.window_entry_budget is not None else args.block_budget)
    take = min(entry_budget, len(candidate_units))

    reference_matrix = np.stack([u.direction for u in reference_units])
    top_values, top_indices = _streamed_top_b(reference_matrix, candidate_units, take)

    # window -> best cosine over all returning (query, unit) pairs
    best_score: dict[int, float] = {}
    unit_scores = {}
    for query in range(len(top_indices)):
        for rank, unit_index in enumerate(top_indices[query]):
            if unit_index < 0:
                continue
            unit = candidate_units[int(unit_index)]
            score = float(top_values[query, rank])
            unit_scores[int(unit_index)] = max(score, unit_scores.get(int(unit_index), -np.inf))
            for position in unit.starts:
                if score > best_score.get(int(position), -np.inf):
                    best_score[int(position)] = score

    if variant == "window_return":
        returned_positions = np.asarray(sorted(best_score), dtype=np.int64)
        duplicate_fraction = adjacent_duplicate_fraction(index, returned_positions)
    else:
        duplicate_fraction = None

    positions = np.asarray(sorted(best_score), dtype=np.int64)
    scores = np.asarray([best_score[int(p)] for p in positions])
    fids, sts = index.fragment_arrays(positions)
    rng = np.random.default_rng(5)
    if variant == "window_return":
        chosen_local = cut_disjoint(fids, sts, scores,
                                    window_length=args.context_length + args.prediction_length,
                                    n=args.n_windows, tiebreak=rng.random(len(positions)))
        chosen = positions[chosen_local]
        windows = index.windows_at(chosen)
        ids = [(int(f), int(s)) for f, s in zip(*index.fragment_arrays(chosen))]
        delivery = {}
    else:
        from .regions import block_regions, deliver_regions
        keys = np.asarray(sorted(unit_scores))
        order = np.lexsort((rng.random(len(keys)), -np.asarray([unit_scores[k] for k in keys])))
        windows = deliver_regions(index, block_regions(index, [
            candidate_units[int(k)].starts for k in keys[order]]), n=args.n_windows)
        ids = []
        delivery = windows.coverage()

    gradients = sum(r.computed_gradients for r in reference_results + candidate_results)
    candidate_windows = sum(len(u.starts) for u in candidate_units)
    dropped_windows = sum(len(r.valid_starts) for r in candidate_results) - candidate_windows
    cost = SelectionCost(
        gradient_computations=gradients,
        index_bytes=int(len(candidate_units) * candidate_units[0].direction.nbytes),
        wall_seconds=0.0,
        notes={
            "reference_units": len(reference_units),
            "candidate_units": len(candidate_units),
            "dropped_units": cand_dropped,
            "dropped_window_fraction": (dropped_windows /
                                        max(1, dropped_windows + candidate_windows)),
            "returned_windows": len(best_score),
            "adjacent_duplicate_fraction": duplicate_fraction,
        },
    )
    return ArmOutput(
        name=variant,
        selected=windows,
        window_ids=ids,
        cost=cost,
        diagnostics={
            "selected": delivery.get("delivered_units", len(windows)),
            "delivery": delivery,
            "entry_budget": entry_budget,
            "score_stats": {
                "max": float(max(best_score.values())),
                "mean": float(np.mean(list(best_score.values()))),
            } if best_score else {},
            **cost.notes,
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out)
    from .regions import check_run_revision, REVISION, load_selection, save_selection
    check_run_revision(out_dir, args)
    args.delivery_revision = REVISION
    (out_dir / "cells").mkdir(parents=True, exist_ok=True)
    (out_dir / "selection").mkdir(parents=True, exist_ok=True)
    started = time.time()

    variants = list(args.variants)
    if args.with_grid:
        variants += [v for v in GRID_VARIANTS if v not in variants]

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
    print("pool: %d trajectories, %d points, %d legal windows; N=%d" % (
        len(pool), pool_manifest["total_points"], len(index), args.n_windows),
        flush=True)

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
    print("engine ready: %d params on %s" % (engine.parameter_count, engine.device),
          flush=True)

    protocol = vars(args)
    protocol["n_select"] = args.n_windows
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

    crb_cache: dict = {}
    for variant in variants:
        selection_path = out_dir / "selection" / ("%s.npz" % variant)
        if selection_path.exists():
            arm_out = load_selection(selection_path)
        else:
            selection_started = time.time()
            arm_out = select_variant(
                variant, task.reference_fragments, pool, index,
                gradient_computer=crb_gradient_computer(
                    engine, args.context_length, args.prediction_length),
                args=args, crb_cache=crb_cache,
            )
            arm_out.cost.wall_seconds = round(time.time() - selection_started, 1)
            save_selection(selection_path, arm_out)
        print("[%s] %-16s selection: %d windows (cost %.1fs, %d gradients)" % (
            spec.dataset, variant, len(arm_out.selected),
            arm_out.cost.wall_seconds, arm_out.cost.gradient_computations),
            flush=True)

        for seed in args.seeds:
            cell = out_dir / "cells" / ("%s__%s__seed%d.json" % (
                spec.dataset, variant, seed))
            if cell.exists():
                continue
            metrics = run_arm_training(
                engine,
                source_windows=arm_out.selected,
                source_epochs=args.source_epochs,
                budget_windows=args.n_windows,
                checkpoint_dir=out_dir / "checkpoints" / variant / ("seed%d" % seed),
                batch_size=args.train_batch,
                learning_rate=args.learning_rate,
                seed=seed,
                test_windows=task.test_windows,
                test_entities=task.test_entities,
                context_length=args.context_length,
                test_scales=task.test_scales,
                base_state=base_state,
            )
            metrics["arm"] = variant
            metrics["dataset"] = spec.dataset
            metrics["selection_cost"] = vars(arm_out.cost)
            metrics["selection_diagnostics"] = arm_out.diagnostics
            if "selection_coverage" not in metrics:
                from .deploy import selection_coverage
                metrics["selection_coverage"] = selection_coverage(
                    arm_out.window_ids, args.context_length + args.prediction_length)
            cell.write_text(json.dumps(metrics, indent=2))
            print("[%s] %-16s seed=%d MASE %.4f [%.0fs]" % (
                spec.dataset, variant, seed, metrics["mase"],
                metrics["elapsed_s"]), flush=True)

    class _Spec:
        dataset = spec.dataset
    summary = summarize_run(out_dir, [_Spec()], args.seeds, variants)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("ABLATION DONE in %.1f min -> %s" % ((time.time() - started) / 60, out_dir),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
