"""Read-only 3:1:1 CRB phase-budget ablation on cached quick REF truth."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from assets.gradient.cache import atomic_json
from assets.gradient.dataset import GradientDataset
from rags.crb import CRBEngine
from rags.experiments.ablation import reconstruct_gradients
from rags.experiments.v7_protocol import crb_for
from rags.experiments.v7_scores import interpolated_chunks, uniform_anchors

from scripts.quick_crb_uniform_ref import (
    DATASET, FIDS, STUDENT, TASK, accuracy, make_context,
)


def phase_allocation(total, ratio=(3, 1, 1)):
    """Largest-remainder integer apportionment, ties: cover, error, split."""
    if total < 3 or len(ratio) != 3 or any(x <= 0 for x in ratio):
        raise ValueError("A positive three-phase ratio needs at least 3 BP")
    target = np.asarray(ratio, dtype=np.float64) * total / sum(ratio)
    counts = np.floor(target).astype(int)
    for phase in np.argsort(-(target - counts), kind="stable")[:total - int(counts.sum())]:
        counts[phase] += 1
    if counts[0] < 2 or np.any(counts < 1) or int(counts.sum()) != total:
        raise ValueError("Rounded CRB phase budget cannot cover endpoints and all phases")
    return tuple(int(x) for x in counts)


def cached_truth(context, first_result):
    """Join old production rows and isolated full-truth rows, with no model/BP."""
    audit, index = context["audit"], context["index"]
    if first_result["schema"] != "quick-crb-uniform-ref-v1" or first_result["model"] != STUDENT \
            or first_result["task"] != TASK or first_result["selected_fragment_ids"] != list(FIDS) \
            or first_result["production_run"] != str(context["run"]):
        raise ValueError("Original quick audit identity differs")
    private = GradientDataset(first_result["full_truth_cost"]["private_cache"])
    fields = private.manifest["data"]
    if fields["production_gradient_manifest"] != audit["gradient_manifest"] \
            or fields["reference_manifest"] != context["ref_manifest"] \
            or fields["selected_fragment_ids"] != list(FIDS) \
            or fields["task"] != TASK or private.manifest["model"] != audit["model"]:
        raise ValueError("Private full-truth cache identity differs")
    present = context["present"]
    old_positions, new_positions = np.flatnonzero(present), np.flatnonzero(~present)
    expected_private = {(DATASET, int(index.fragment_ids[p]), int(index.starts[p])) for p in new_positions}
    if set(private._positions) != expected_private:
        raise ValueError("Full-truth cache is incomplete or contains unexpected windows")
    truth = np.empty((len(index), 4096), dtype=np.float32)
    for begin in range(0, len(old_positions), 1024):
        positions = old_positions[begin:begin+1024]
        truth[positions] = context["old"].for_index(index, positions, dataset=DATASET)
    for begin in range(0, len(new_positions), 1024):
        positions = new_positions[begin:begin+1024]
        truth[positions] = private.for_index(index, positions, dataset=DATASET)
    if not np.isfinite(truth).all():
        raise ValueError("Nonfinite cached full-truth gradient")
    return truth


class TruthComputer:
    def __init__(self, index, truth):
        self.index, self.truth = index, truth

    def __call__(self, fragment, physical_starts):
        # The CRB algorithm receives only its requested anchor rows. Full
        # truth is held behind this lookup and cannot affect phase decisions.
        positions = [self.index.locate(fragment.fragment_id, int(start))
                     for start in physical_starts]
        return self.truth[positions]


def evaluate(context, truth, original):
    index = context["index"]
    rows = []
    aggregate = {name: [] for name in ("truth", "native_111", "crb_311", "uniform", "common_off_anchor")}
    for fragment, native in zip(context["refs"], context["results"]):
        fid = fragment.fragment_id
        lo, hi = index.bounds(fid)
        exact = truth[lo:hi]
        total = native.computed_gradients
        cover, error, split = phase_allocation(total)
        n = len(exact)
        config = replace(crb_for(STUDENT),
            cover_density=cover/n, error_density=error/n, split_density=split/n)
        ratio_result = CRBEngine(config, TruthComputer(index, truth)).process_fragment(fragment)
        actual = (ratio_result.cover_computations, ratio_result.error_computations,
                  ratio_result.split_computations)
        if actual != (cover, error, split) or ratio_result.computed_gradients != total \
                or len(ratio_result.anchors) != total:
            raise ValueError("3:1:1 CRB did not spend exactly the original BP budget")
        p_uniform = uniform_anchors(n, total)
        uniform = np.concatenate([part for _, part in interpolated_chunks(
            n, p_uniform, exact[p_uniform])]).astype(np.float32)
        native_curve = reconstruct_gradients(native)
        ratio_curve = reconstruct_gradients(ratio_result)
        off = np.ones(n, dtype=bool)
        off[np.union1d(np.union1d(np.array(list(native.anchors)),
            np.array(list(ratio_result.anchors))), p_uniform)] = False
        row = dict(fragment_id=fid, legal_windows=n, total_bp_each=total,
            native_111_phase_bp=[native.cover_computations, native.error_computations,
                                  native.split_computations],
            crb_311_phase_bp=list(actual),
            crb_311_anchor_positions=sorted(int(p) for p in ratio_result.anchors),
            common_off_anchor_windows=int(off.sum()),
            native_111=accuracy(exact, native_curve),
            crb_311=accuracy(exact, ratio_curve), uniform=accuracy(exact, uniform),
            native_111_common_off_anchor=accuracy(exact, native_curve, off),
            crb_311_common_off_anchor=accuracy(exact, ratio_curve, off),
            uniform_common_off_anchor=accuracy(exact, uniform, off))
        rows.append(row)
        for name, values in (("truth", exact), ("native_111", native_curve),
                             ("crb_311", ratio_curve), ("uniform", uniform),
                             ("common_off_anchor", off)):
            aggregate[name].append(values)
    exact = np.concatenate(aggregate["truth"])
    off = np.concatenate(aggregate["common_off_anchor"])
    pooled = {name: accuracy(exact, np.concatenate(aggregate[name]))
              for name in ("native_111", "crb_311", "uniform")}
    pooled["common_off_anchor"] = {name: accuracy(exact,
        np.concatenate(aggregate[name]), off) for name in ("native_111", "crb_311", "uniform")}
    for name, original_name in (("native_111", "native_crb"), ("uniform", "uniform")):
        if not np.isclose(pooled[name]["relative_frobenius_rmse"],
                          original["pooled"][original_name]["relative_frobenius_rmse"], atol=1e-12):
            raise ValueError("Cached-truth replay does not reproduce original result")
    return dict(schema="quick-crb-phase-ratio-v1", parent_result=str(context["output"] / "result.json"),
        model=STUDENT, task=TASK, fragment_ids=list(FIDS), ratio=[3, 1, 1],
        apportionment="largest remainder; tie order cover, error, split; totals fixed per trajectory",
        new_backward_passes=0, total_legal_windows=len(index),
        total_bp_per_method=sum(row["total_bp_each"] for row in rows),
        per_trajectory=rows, pooled=pooled,
        limitation="Projected-gradient reconstruction on three short REF trajectories, not downstream MASE.")


def report(result):
    lines = ["# Quick REF CRB phase-ratio ablation", "",
        "Same Chronos/time_cphl_h full-gradient truth as the original quick audit. Each method has the same per-trajectory total BP; 3:1:1 is rounded by largest remainder. No new BP was computed.", "",
        "| Fragment | Windows | Total BP | Native 1:1:1 phase BP | New 3:1:1 phase BP | Native RMSE | 3:1:1 RMSE | Uniform RMSE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["per_trajectory"]:
        lines.append(f"| {row['fragment_id']} | {row['legal_windows']} | {row['total_bp_each']} | {row['native_111_phase_bp']} | {row['crb_311_phase_bp']} | {row['native_111']['relative_frobenius_rmse']:.4f} | {row['crb_311']['relative_frobenius_rmse']:.4f} | {row['uniform']['relative_frobenius_rmse']:.4f} |")
    p = result["pooled"]
    lines += ["", "Pooled relative RMSE: native 1:1:1 **%.4f**, new 3:1:1 **%.4f**, uniform **%.4f**." % (
        p["native_111"]["relative_frobenius_rmse"], p["crb_311"]["relative_frobenius_rmse"],
        p["uniform"]["relative_frobenius_rmse"]),
        "Common off-anchor relative RMSE: native %.4f, 3:1:1 %.4f, uniform %.4f." % (
        p["common_off_anchor"]["native_111"]["relative_frobenius_rmse"],
        p["common_off_anchor"]["crb_311"]["relative_frobenius_rmse"],
        p["common_off_anchor"]["uniform"]["relative_frobenius_rmse"]),
        "", result["limitation"], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--first-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    context = make_context(args.run, args.first_output)
    first = json.loads((args.first_output / "result.json").read_text())
    truth = cached_truth(context, first)
    result = evaluate(context, truth, first)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "result.json", result)
    (args.output / "report.md").write_text(report(result))
    print("QUICK_CRB_RATIO_RESULT", json.dumps(dict(pooled=result["pooled"],
        new_backward_passes=0)), flush=True)


if __name__ == "__main__":
    main()
