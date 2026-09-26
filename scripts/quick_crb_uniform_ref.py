"""Budget-matched CRB vs uniform reference-gradient reconstruction audit.

Only the missing full-truth rows are backpropagated, in an isolated cache.
The production V7 cache, selections, and training workers are read-only here.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np

from assets.gradient.cache import GradientCache, atomic_json, index_request
from assets.gradient.dataset import GradientDataset
from rags.crb import CRBEngineMulti
from rags.execution import SelectionExecution, selection_execution
from rags.experiments.ablation import reconstruct_gradients
from rags.experiments.arms import WindowIndex
from rags.experiments.development_inputs import load_fragments
from rags.experiments.full_pool_models import build_model
from rags.experiments.v7_protocol import crb_for
from rags.experiments.v7_scores import interpolated_chunks, uniform_anchors


STUDENT = "chronos-base"
TASK = "time_cphl_h"
FIDS = (1, 3, 5)
DATASET = f"reference:{TASK}"


class ReadOnlyCachedComputer:
    def __init__(self, dataset, name):
        self.dataset, self.name = dataset, name

    def compute_many(self, entries):
        ids = np.concatenate([np.column_stack((
            np.full(len(starts), fragment.fragment_id, dtype=np.int64), starts))
            for fragment, starts in entries])
        keys = [(self.name, int(fid), int(start)) for fid, start in ids]
        missing = [key for key in keys if key not in self.dataset._positions]
        if missing:
            raise RuntimeError(f"Native CRB requests {len(missing)} missing production rows: {missing[:3]}")
        return self.dataset.lookup(self.name, ids)

    def __call__(self, fragment, starts):
        return self.compute_many([(fragment, starts)])


def accuracy(truth, estimate, mask=None):
    truth = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if mask is not None:
        truth, estimate = truth[mask], estimate[mask]
    if not len(truth):
        return None
    residual = truth - estimate
    tn = np.linalg.norm(truth, axis=1)
    en = np.linalg.norm(estimate, axis=1)
    cosine = np.divide(np.einsum("ij,ij->i", truth, estimate), tn * en,
                       out=np.zeros(len(truth)), where=(tn * en) > 0)
    tmean, emean = truth.mean(axis=0), estimate.mean(axis=0)
    denominator = np.linalg.norm(tmean) * np.linalg.norm(emean)
    return dict(windows=len(truth), relative_frobenius_rmse=float(np.linalg.norm(residual) /
        max(np.linalg.norm(truth), 1e-30)), mean_window_cosine=float(cosine.mean()),
        median_window_cosine=float(np.median(cosine)),
        mean_gradient_relative_error=float(np.linalg.norm(tmean-emean) /
            max(np.linalg.norm(tmean), 1e-30)),
        mean_gradient_cosine=float(np.dot(tmean, emean) / denominator) if denominator else 0.)


def make_context(run, output):
    receipt = json.loads((run / "seeds/seed0" / STUDENT / "assets" / f"{TASK}.json").read_text())
    audit = receipt["audit"]
    if not receipt["complete"] or not audit["complete"] or audit["task"] != TASK:
        raise ValueError("Production assets are incomplete or the task differs")
    old = GradientDataset(audit["path"])
    if old.manifest != audit["gradient_manifest"]:
        raise ValueError("Production gradient manifest differs from its asset receipt")
    native_input = run / "inputs" / STUDENT / f"{TASK}__reference.npz"
    ref_manifest = json.loads((run / "inputs" / STUDENT / f"{TASK}__manifest.json").read_text())
    marker = json.loads((Path(audit["path"]) / f"reference-{TASK}.json").read_text())
    all_fragments = load_fragments(native_input)
    by_fid = {f.fragment_id: f for f in all_fragments}
    if len(by_fid) != len(all_fragments) or set(FIDS) - set(by_fid):
        raise ValueError("Reference fragment identities are incomplete")
    for record in marker["fragments"]:
        fragment = by_fid[int(record["fragment_id"])]
        if fragment.entity_id != record["entity_id"] or fragment.length != record["length"]:
            raise ValueError("Native reference marker and input disagree")
    refs = [by_fid[fid] for fid in FIDS]
    index = WindowIndex(refs, context_length=512, prediction_length=64)
    planned = {int(row["fragment_id"]): row for row in audit["plans"]["reference"]}
    crb = crb_for(STUDENT)
    if audit["crb"] != asdict(crb):
        raise ValueError("Native CRB scientific configuration differs")
    for fid in FIDS:
        row = planned[fid]
        if row["legal_windows"] != index.fragment_lengths[fid] or row["gradients"] != sum(
                crb.budgets(index.fragment_lengths[fid])):
            raise ValueError("Native per-trajectory BP budget differs")
    results = CRBEngineMulti(crb, ReadOnlyCachedComputer(old, DATASET)).process_fragments(refs)
    for result in results:
        row = planned[result.fragment_id]
        if result.computed_gradients != row["gradients"] or len(result.valid_starts) != row["legal_windows"]:
            raise ValueError("Replayed CRB BP/starts differ from the production audit")
    present = np.array([(DATASET, int(fid), int(start)) in old._positions
        for fid, start in zip(index.fragment_ids, index.starts)], dtype=bool)
    return dict(old=old, results=results, refs=refs, index=index, present=present,
                audit=audit, receipt=receipt, ref_manifest=ref_manifest, input=native_input,
                output=output, run=run)


def make_truth(context, *, config):
    index, old, present = context["index"], context["old"], context["present"]
    output = context["output"]
    # An explicit private namespace prevents a V7 model lock collision and
    # prevents an audit-only full scan from altering production cache identity.
    private_root = output / "gradient_cache"
    os.environ["TS3_GRADIENT_CACHE_ROOT"] = str(private_root)
    _, probe, identity = build_model(STUDENT, model_path=config["model_paths"].get(STUDENT),
                                     context_length=512, prediction_length=64)
    if identity != context["audit"]["model"]:
        raise ValueError("Current model identity differs from the production checkpoint")
    execution = dict(config["selection_execution"])
    execution.update(json.loads((context["run"] / "execution.json").read_text()).get(STUDENT, {}))
    probe.execution = SelectionExecution(**execution)
    data_fields = dict(experiment="quick-crb-uniform-ref-v1", task=TASK,
        reference_manifest=context["ref_manifest"], selected_fragment_ids=list(FIDS),
        production_gradient_manifest=context["audit"]["gradient_manifest"])
    cache = GradientCache(output, probe, model_identity=identity, data_fields=data_fields,
                          context_length=512, prediction_length=64)
    missing = np.flatnonzero(~present)
    started = time.perf_counter()
    try:
        with selection_execution(probe):
            for begin in range(0, len(missing), 1024):
                cache.prepare([index_request(DATASET, index, missing[begin:begin+1024])])
        computed = cache.computed_windows
        cache.freeze()
        truth = np.empty((len(index), 4096), dtype=np.float32)
        for begin in range(0, len(index), 1024):
            positions = np.arange(begin, min(begin+1024, len(index)))
            old_positions = positions[present[positions]]
            new_positions = positions[~present[positions]]
            if len(old_positions):
                truth[old_positions] = old.for_index(index, old_positions, dataset=DATASET)
            if len(new_positions):
                truth[new_positions] = cache.get(index_request(DATASET, index, new_positions))
    finally:
        cache.close()
    if not np.isfinite(truth).all() or truth.shape != (len(index), 4096):
        raise ValueError("Full reference truth has a wrong shape or nonfinite rows")
    return truth, dict(production_reused_windows=int(present.sum()), private_existing_windows=int(
        len(missing)-computed), new_full_truth_bp=int(computed), private_cache=str(cache.path),
        elapsed_seconds=time.perf_counter()-started)


def compare(context, truth, truth_cost):
    index = context["index"]
    rows = []
    pooled = {"uniform": [], "native_crb": [], "truth": [], "off_union": []}
    for result in context["results"]:
        fid = result.fragment_id
        lo, hi = index.bounds(fid)
        exact = truth[lo:hi]
        k = result.computed_gradients
        uniform_positions = uniform_anchors(len(exact), k)
        uniform = np.concatenate([v for _, v in interpolated_chunks(len(exact),
            uniform_positions, exact[uniform_positions], chunk=1024)]).astype(np.float32)
        native = reconstruct_gradients(result)
        native_positions = np.array(sorted(result.anchors), dtype=np.int64)
        if len(native_positions) != k or len(uniform_positions) != k:
            raise ValueError("Actual BP is not exactly matched")
        off_union = np.ones(len(exact), dtype=bool)
        off_union[np.union1d(native_positions, uniform_positions)] = False
        row = dict(fragment_id=fid, entity_id=index.by_fragment[fid].entity_id,
            legal_windows=len(exact), bp_each=k, bp_density=k/len(exact),
            native_anchor_positions=native_positions.tolist(),
            uniform_anchor_positions=uniform_positions.tolist(),
            off_union_windows=int(off_union.sum()),
            native_crb=accuracy(exact, native), uniform=accuracy(exact, uniform),
            native_crb_off_union=accuracy(exact, native, off_union),
            uniform_off_union=accuracy(exact, uniform, off_union))
        row["uniform_minus_native_rmse"] = row["uniform"]["relative_frobenius_rmse"] - row["native_crb"]["relative_frobenius_rmse"]
        rows.append(row)
        pooled["truth"].append(exact)
        pooled["native_crb"].append(native)
        pooled["uniform"].append(uniform)
        pooled["off_union"].append(off_union)
    exact = np.concatenate(pooled["truth"])
    mask = np.concatenate(pooled["off_union"])
    payload = dict(schema="quick-crb-uniform-ref-v1", model=STUDENT, task=TASK,
        selected_fragment_ids=list(FIDS), selection_basis="three short reference trajectories; Chronos local cached-gradient direction relatively stable among three V7 models on this task",
        production_run=str(context["run"]), production_cache=context["audit"]["path"],
        scientific_identity=dict(model=context["audit"]["model"], crb=context["audit"]["crb"],
            reference_manifest=context["ref_manifest"], dataset=DATASET,
            physical_start_base=1, gradient_storage="BF16 decoded to FP32"),
        full_truth_cost=truth_cost, total_legal_windows=len(index),
        total_bp_each=sum(row["bp_each"] for row in rows),
        per_trajectory=rows,
        pooled=dict(native_crb=accuracy(exact, np.concatenate(pooled["native_crb"])),
            uniform=accuracy(exact, np.concatenate(pooled["uniform"])),
            native_crb_off_union=accuracy(exact, np.concatenate(pooled["native_crb"]), mask),
            uniform_off_union=accuracy(exact, np.concatenate(pooled["uniform"]), mask)),
        interpretation_limit="This tests projected-gradient reconstruction fidelity, not downstream selection probability or final MASE; only three short reference trajectories.")
    return payload


def report(payload):
    rows = payload["per_trajectory"]
    lines = ["# Quick reference CRB reconstruction check", "",
        "Chronos, time_cphl_h REF, fragment IDs 1/3/5. Same actual per-trajectory BP in both methods; independent full-gradient truth is not counted as either method's cost.", "",
        "| REF fragment | legal windows | BP each | native CRB rel. RMSE | uniform rel. RMSE | uniform − native |",
        "|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['fragment_id']} | {row['legal_windows']} | {row['bp_each']} | {row['native_crb']['relative_frobenius_rmse']:.4f} | {row['uniform']['relative_frobenius_rmse']:.4f} | {row['uniform_minus_native_rmse']:+.4f} |")
    p = payload["pooled"]
    lines += ["", f"Pooled relative RMSE: native CRB {p['native_crb']['relative_frobenius_rmse']:.4f}; uniform {p['uniform']['relative_frobenius_rmse']:.4f}.",
        f"Off-union-of-anchors relative RMSE: native CRB {p['native_crb_off_union']['relative_frobenius_rmse']:.4f}; uniform {p['uniform_off_union']['relative_frobenius_rmse']:.4f}.",
        "", f"Full-truth audit: {payload['total_legal_windows']} windows; {payload['full_truth_cost']['production_reused_windows']} old production rows reused; {payload['full_truth_cost']['new_full_truth_bp']} new BP in a private cache.",
        "", payload["interpretation_limit"], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assess-only", action="store_true")
    args = parser.parse_args()
    context = make_context(args.run, args.output)
    summary = dict(fragments=[dict(fid=r.fragment_id, legal_windows=len(r.valid_starts),
        native_crb_bp=r.computed_gradients) for r in context["results"]],
        old_rows=int(context["present"].sum()), missing_rows=int((~context["present"]).sum()))
    print("QUICK_CRB_ASSESS", json.dumps(summary), flush=True)
    if args.assess_only:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    truth, cost = make_truth(context, config=json.loads(args.config.read_text()))
    result = compare(context, truth, cost)
    atomic_json(args.output / "result.json", result)
    (args.output / "report.md").write_text(report(result))
    print("QUICK_CRB_RESULT", json.dumps(dict(pooled=result["pooled"], cost=cost)), flush=True)


if __name__ == "__main__":
    main()
