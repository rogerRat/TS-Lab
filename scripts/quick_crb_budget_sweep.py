"""Zero-BP exhaustive CRB phase allocations on the cached quick REF truth."""

import argparse
from dataclasses import replace
from itertools import product
import json
from math import gcd, sqrt
from pathlib import Path

import numpy as np

from assets.gradient.cache import atomic_json
from rags.crb import CRBEngine
from rags.experiments.ablation import reconstruct_gradients
from rags.experiments.v7_protocol import crb_for
from rags.experiments.v7_scores import interpolated_chunks, uniform_anchors
from scripts.quick_crb_ratio_ref import TruthComputer, cached_truth, phase_allocation
from scripts.quick_crb_uniform_ref import FIDS, STUDENT, TASK, make_context


def feasible_allocations(total):
    """Coverage pays for both endpoints; keep ≥1 probe in each later phase."""
    return [(cover, error, total-cover-error)
        for cover in range(2, total-1)
        for error in range(1, total-cover)
        if total-cover-error >= 1]


def reduced_ratio(ratio):
    divisor = gcd(gcd(ratio[0], ratio[1]), ratio[2])
    return tuple(int(v // divisor) for v in ratio)


def constrained_phase_allocation(total, ratio):
    """Closest feasible integer allocation to the ratio's fractional target.

    Unlike naive rounding, this preserves two coverage endpoints and at least
    one error/split probe even for very coverage-heavy ratios at K=4.
    """
    if len(ratio) != 3 or any(value <= 0 for value in ratio):
        raise ValueError("CRB ratio must have three positive weights")
    target = np.asarray(ratio, dtype=np.float64) * total / sum(ratio)
    return min(feasible_allocations(total), key=lambda counts: (
        sum((value - desired) ** 2 for value, desired in zip(counts, target)),
        -counts[0], -counts[1], -counts[2]))


def score_allocations(index, fragments, old_results, truth):
    by_fid = {}
    computer = TruthComputer(index, truth)
    for fragment, old in zip(fragments, old_results):
        fid = fragment.fragment_id
        lo, hi = index.bounds(fid)
        exact = truth[lo:hi].astype(np.float64)
        denominator = float(np.einsum("ij,ij->", exact, exact))
        if denominator <= 0:
            raise ValueError("Zero full-truth gradient energy")
        results = {}
        for cover, error, split in feasible_allocations(old.computed_gradients):
            config = replace(crb_for(STUDENT), cover_density=cover/len(exact),
                error_density=error/len(exact), split_density=split/len(exact))
            rebuilt = CRBEngine(config, computer).process_fragment(fragment)
            actual = (rebuilt.cover_computations, rebuilt.error_computations,
                      rebuilt.split_computations)
            if actual != (cover, error, split) or len(rebuilt.anchors) != old.computed_gradients:
                raise ValueError(f"CRB actual BP differs for {fid} {cover,error,split}: {actual}")
            estimated = reconstruct_gradients(rebuilt).astype(np.float64)
            residual = exact - estimated
            sse = float(np.einsum("ij,ij->", residual, residual))
            results[(cover, error, split)] = dict(relative_rmse=sqrt(sse/denominator),
                squared_error=sse, gradient_energy=denominator)
        by_fid[fid] = results
    return by_fid


def evaluate_ratio_grid(by_fid, totals, *, max_weight=12, constrained=False):
    ratios = {}
    for raw in product(range(1, max_weight+1), repeat=3):
        ratio = reduced_ratio(raw)
        if ratio in ratios:
            continue
        if constrained:
            allocations = tuple(constrained_phase_allocation(totals[fid], ratio) for fid in FIDS)
        else:
            try:
                allocations = tuple(phase_allocation(totals[fid], ratio) for fid in FIDS)
            except ValueError:
                continue
        if any(allocation not in by_fid[fid] for fid, allocation in zip(FIDS, allocations)):
            continue
        entries = [by_fid[fid][allocation]
                   for fid, allocation in zip(FIDS, allocations)]
        rows = [entry["relative_rmse"] for entry in entries]
        pooled = sqrt(sum(entry["squared_error"] for entry in entries) /
                      sum(entry["gradient_energy"] for entry in entries))
        ratios[ratio] = dict(ratio=list(ratio), allocations=[list(x) for x in allocations],
            equal_trajectory_mean_rmse=float(np.mean(rows)), pooled_rmse=pooled,
            worst_trajectory_rmse=max(rows), per_trajectory_rmse=rows)
    return ratios


def reference_mean_check(context, truth, ratios=((3,2,1), (4,1,1), (9,1,1))):
    """Compare an equal-trajectory reference mean, used by gradient baselines."""
    index = context["index"]
    truth_means = []
    estimated = {"native_111": [], "uniform": []}
    estimated.update({":".join(map(str, ratio)): [] for ratio in ratios})
    computer = TruthComputer(index, truth)
    for fragment, native in zip(context["refs"], context["results"]):
        lo, hi = index.bounds(fragment.fragment_id)
        exact = truth[lo:hi]
        n, k = len(exact), native.computed_gradients
        truth_means.append(exact.astype(np.float64).mean(axis=0))
        estimated["native_111"].append(reconstruct_gradients(native).astype(np.float64).mean(axis=0))
        anchors = uniform_anchors(n, k)
        uniform = np.concatenate([piece for _, piece in interpolated_chunks(
            n, anchors, exact[anchors])])
        estimated["uniform"].append(uniform.mean(axis=0))
        for ratio in ratios:
            cover, error, split = constrained_phase_allocation(k, ratio)
            config = replace(crb_for(STUDENT), cover_density=cover/n,
                error_density=error/n, split_density=split/n)
            result = CRBEngine(config, computer).process_fragment(fragment)
            estimated[":".join(map(str, ratio))].append(
                reconstruct_gradients(result).astype(np.float64).mean(axis=0))
    target = np.mean(truth_means, axis=0)
    answer = {}
    for label, means in estimated.items():
        vector = np.mean(means, axis=0)
        scale = np.linalg.norm(target) * np.linalg.norm(vector)
        answer[label] = dict(relative_error=float(np.linalg.norm(target-vector) /
            max(np.linalg.norm(target), 1e-30)),
            cosine=float(np.dot(target, vector) / scale) if scale else 0.)
    return answer


def run(context, truth, original, ratio_result):
    totals = {result.fragment_id: result.computed_gradients
              for result in context["results"]}
    by_fid = score_allocations(context["index"], context["refs"], context["results"], truth)
    grid = evaluate_ratio_grid(by_fid, totals)
    constrained_grid = evaluate_ratio_grid(by_fid, totals, max_weight=24, constrained=True)
    for ratio, name in (((1, 1, 1), "native_111"), ((3, 1, 1), "crb_311")):
        row = grid[ratio]
        expected = ratio_result["pooled"][name]["relative_frobenius_rmse"]
        if not np.isclose(row["pooled_rmse"], expected, atol=1e-12):
            raise ValueError(f"Cached sweep does not reproduce {ratio}: {row['pooled_rmse']} != {expected}")
        constrained_row = constrained_grid[ratio]
        if not np.isclose(constrained_row["pooled_rmse"], expected, atol=1e-12):
            raise ValueError(f"Constrained allocator does not reproduce {ratio}")
    top_equal = sorted(grid.values(), key=lambda r: (r["equal_trajectory_mean_rmse"],
        r["pooled_rmse"], sum(r["ratio"])))[:20]
    top_pooled = sorted(grid.values(), key=lambda r: (r["pooled_rmse"],
        r["equal_trajectory_mean_rmse"], sum(r["ratio"])))[:20]
    constrained_top_equal = sorted(constrained_grid.values(), key=lambda r: (
        r["equal_trajectory_mean_rmse"], r["pooled_rmse"], sum(r["ratio"])))[:20]
    constrained_top_pooled = sorted(constrained_grid.values(), key=lambda r: (
        r["pooled_rmse"], r["equal_trajectory_mean_rmse"], sum(r["ratio"])))[:20]
    best_per_trajectory = {str(fid): sorted([dict(phase_bp=list(counts), **metrics)
        for counts, metrics in scores.items()], key=lambda r: r["relative_rmse"])[:10]
        for fid, scores in by_fid.items()}
    return dict(schema="quick-crb-budget-sweep-v1", model=STUDENT, task=TASK,
        fragment_ids=list(FIDS), total_bp_by_fragment={str(fid): totals[fid] for fid in FIDS},
        feasible_allocations_by_fragment={str(fid): len(by_fid[fid]) for fid in FIDS},
        ratio_grid="positive integer cover:error:split weights in [1,12], gcd-reduced; largest-remainder apportionment; discard allocations with cover<2 or later phase<1",
        distinct_ratios=len(grid), distinct_allocation_pairs=len({tuple(map(tuple, row["allocations"]))
            for row in grid.values()}),
        constrained_ratio_grid="positive integer weights in [1,24], gcd-reduced; choose nearest feasible integer allocation by squared distance to fractional ratio target, with minimum (2,1,1) and cover/error/split tie order",
        constrained_distinct_ratios=len(constrained_grid),
        constrained_distinct_allocation_pairs=len({tuple(map(tuple, row["allocations"]))
            for row in constrained_grid.values()}),
        new_backward_passes=0, source_truth=str(context["output"] / "result.json"),
        baseline_pooled_rmse=dict(native_111=original["pooled"]["native_crb"]["relative_frobenius_rmse"],
            crb_311=ratio_result["pooled"]["crb_311"]["relative_frobenius_rmse"],
            uniform=original["pooled"]["uniform"]["relative_frobenius_rmse"]),
        top_equal_trajectory_ratios=top_equal, top_pooled_ratios=top_pooled,
        constrained_top_equal_trajectory_ratios=constrained_top_equal,
        constrained_top_pooled_ratios=constrained_top_pooled,
        constrained_named_ratios={":".join(map(str, ratio)): constrained_grid[ratio]
            for ratio in ((1,1,1), (3,1,1), (3,2,1), (4,1,1),
                          (8,1,1), (12,2,1), (15,2,1), (16,2,1), (20,1,1))},
        equal_trajectory_reference_mean=reference_mean_check(context, truth),
        best_allocations_per_trajectory=best_per_trajectory,
        all_ratio_results=[grid[key] for key in sorted(grid)],
        limitation="Exploratory in-sample fit on only three preselected short REF trajectories; use a held-out reference/task set before freezing a new V7 protocol or claiming downstream MASE improvement.")


def report(result):
    lines = ["# Exploratory CRB phase-budget sweep", "",
        "Same 3 Chronos/time_cphl_h REF trajectories and cached full gradients; **zero additional BP**. This grid tunes and evaluates on the same trajectories, so its optimum is descriptive, not validated.", "",
        f"Feasible integer allocations: {result['feasible_allocations_by_fragment']}; "
        f"{result['distinct_ratios']} reduced ratio candidates, "
        f"{result['distinct_allocation_pairs']} distinct realized allocation combinations.", "",
        "| Ratio | Phase BP for fragments 1 / 3 / 5 | Mean trajectory RMSE | Pooled RMSE |",
        "|---:|---|---:|---:|"]
    selected = [next(r for r in result["all_ratio_results"] if r["ratio"] == list(ratio))
        for ratio in ((1, 1, 1), (2, 1, 1), (3, 1, 1), (4, 1, 1))
        if any(r["ratio"] == list(ratio) for r in result["all_ratio_results"])]
    selected += result["top_equal_trajectory_ratios"][:5]
    seen = set()
    for row in selected:
        key = tuple(row["ratio"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"| {':'.join(map(str,key))} | {row['allocations']} | "
                     f"{row['equal_trajectory_mean_rmse']:.4f} | {row['pooled_rmse']:.4f} |")
    lines += ["", "Best feasible phase allocations by trajectory:", ""]
    for fid in FIDS:
        best = result["best_allocations_per_trajectory"][str(fid)][0]
        lines.append(f"- Fragment {fid}: {best['phase_bp']}, relative RMSE {best['relative_rmse']:.4f}.")
    lines += ["", "With explicit minimum (cover,error,split)=(2,1,1), the broader [1,24] weight grid also admits coverage-heavy ratios:", "",
              "| Ratio | Allocations | Mean trajectory RMSE | Pooled RMSE |",
              "|---:|---|---:|---:|"]
    for ratio, row in result["constrained_named_ratios"].items():
        lines.append(f"| {ratio} | {row['allocations']} | {row['equal_trajectory_mean_rmse']:.4f} | {row['pooled_rmse']:.4f} |")
    winner = result["constrained_top_equal_trajectory_ratios"][0]
    lines += ["", f"Lowest in-sample equal-trajectory mean among this constrained grid: {winner['ratio']} → {winner['allocations']}, mean {winner['equal_trajectory_mean_rmse']:.4f}, pooled {winner['pooled_rmse']:.4f}."]
    lines += ["", "For the equal-trajectory **reference mean gradient** (a complementary target used by V7 gradient baselines, not identical to RAGS block matching), relative vector errors are:", ""]
    for label, row in result["equal_trajectory_reference_mean"].items():
        lines.append(f"- {label}: relative error {row['relative_error']:.4f}, cosine {row['cosine']:.4f}.")
    lines += ["", result["limitation"], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--first-output", type=Path, required=True)
    parser.add_argument("--ratio-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    context = make_context(args.run, args.first_output)
    original = json.loads((args.first_output / "result.json").read_text())
    ratio = json.loads((args.ratio_output / "result.json").read_text())
    truth = cached_truth(context, original)
    result = run(context, truth, original, ratio)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "result.json", result)
    (args.output / "report.md").write_text(report(result))
    print("QUICK_CRB_SWEEP", json.dumps(dict(top_equal=result["top_equal_trajectory_ratios"][:3],
        constrained_top_equal=result["constrained_top_equal_trajectory_ratios"][:3],
        constrained_top_pooled=result["constrained_top_pooled_ratios"][:3],
        top_pooled=result["top_pooled_ratios"][:3], distinct_ratios=result["distinct_ratios"],
        distinct_allocation_pairs=result["distinct_allocation_pairs"], new_bp=0)), flush=True)


if __name__ == "__main__":
    main()
