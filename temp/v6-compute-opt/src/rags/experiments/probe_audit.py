"""Audit native gradients and initial source/reference support without test data."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .calibration import build_student
from .native_probe import NativeProbe
from .regions import load_selection
from .students import atomic_json


def balanced_reference_rows(ids, count, rng):
    """Equal reference-window counts per dataset; no performance-based sampling."""
    groups = {}
    for row, identity in enumerate(ids):
        groups.setdefault(identity[0].split("::", 1)[0], []).append(row)
    if count % len(groups):
        raise ValueError("Balanced reference count must be divisible by the dataset count")
    return np.concatenate([rng.choice(groups[key], count // len(groups), replace=False)
                           for key in sorted(groups)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", choices=["timesfm-2.5", "timemoe-200m"], required=True)
    parser.add_argument("--source-selection", required=True)
    parser.add_argument("--reference-probes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-count", type=int, default=64)
    parser.add_argument("--reference-count", type=int, default=16)
    parser.add_argument("--balance-reference-datasets", action="store_true")
    parser.add_argument("--gpu-memory-fraction", type=float, default=.18)
    parser.add_argument("--finite-difference-steps", type=float, nargs="+",
                        default=[1e-2, 3e-3, 1e-3],
                        help="Perturbation sizes; smaller steps help diagnose discrete MoE routing")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    started = time.monotonic()
    source = load_selection(args.source_selection)
    with np.load(args.reference_probes) as data:
        refs, ref_ids = data["windows"], json.loads(str(data["ids"]))
    rng = np.random.default_rng(20260913)
    src_rows = rng.choice(len(source.selected), args.source_count, replace=False)
    ref_rows = (balanced_reference_rows(ref_ids, args.reference_count, rng)
                if args.balance_reference_datasets else
                rng.choice(len(refs), args.reference_count, replace=False))
    probe = NativeProbe(build_student(args.student), model_key=args.student)
    audit = probe.finite_difference(source.selected[src_rows[0]],
                                    steps=args.finite_difference_steps)
    print("finite difference", json.dumps(audit), flush=True)
    _, reference = probe.batch_gradient_sketch(refs[ref_rows])
    _, candidates = probe.batch_gradient_sketch(source.selected[src_rows])
    ref_unit = reference / np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-12)
    src_unit = candidates / np.maximum(np.linalg.norm(candidates, axis=1, keepdims=True), 1e-12)
    cosine = src_unit @ ref_unit.T
    labels = np.array([ref_ids[int(i)][0].split("::", 1)[0] for i in ref_rows])
    by_dataset = {}
    for label in sorted(set(labels)):
        values = cosine[:, labels == label]
        by_dataset[label] = dict(reference_count=values.shape[1],
            positive_pair_fraction=float(np.mean(values > 0)),
            cosine_quantiles=np.quantile(values, [0, .1, .5, .9, 1]).tolist(),
            best_per_reference_quantiles=np.quantile(values.max(axis=0), [0, .5, 1]).tolist())
    record = dict(arguments=vars(args), native_gradient_audit=audit,
        by_dataset=by_dataset,
        elapsed_seconds=time.monotonic() - started, peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
        source_ids=[source.window_ids[int(i)] for i in src_rows],
        reference_ids=[ref_ids[int(i)] for i in ref_rows],
        source_reference_cosines=cosine.tolist(),
        cosine_quantiles=np.quantile(cosine, [0, .1, .5, .9, 1]).tolist(),
        positive_pair_fraction=float(np.mean(cosine > 0)),
        per_reference_best_cosine=cosine.max(axis=0).tolist(),
        warning="Small development support audit, not proof that positive cosine improves downstream MASE")
    atomic_json(Path(args.out), record)
    print("NATIVE_PROBE_AUDIT_DONE", flush=True)
    return 0
