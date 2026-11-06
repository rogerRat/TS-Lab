"""Recalibrate original p10 CRB thresholds on real, pre-exam development data."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from rage.core import TimeSeriesFragment
from rage.execution import SelectionExecution, selection_execution
from .development_inputs import load_fragments, save_fragments
from .full_pool_models import build_model
from .students import STUDENTS, atomic_json
from assets.gradient import GradientCache, FragmentGradients


def import_inputs(folder, payload):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for side in ("reference", "source"):
        fragments = [TimeSeriesFragment(np.asarray(row["values"], dtype=np.float32),
            entity_id=row["entity"], fragment_id=row["fragment_id"]) for row in payload[side]]
        save_fragments(folder / f"{side}.npz", fragments)
    atomic_json(folder / "provenance.json", payload["provenance"])


def calibrate(student, inputs, out, *, model_path=None):
    inputs, out = Path(inputs), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    engine, probe, model = build_model(student, model_path=model_path)
    probe.execution = SelectionExecution(gradient_batch=8, gradient_workers=4, inference_batch=2, generation="eager")
    provenance = json.loads((inputs / "provenance.json").read_text())
    identity = dict(model=model, inputs=provenance, rule="p10_full_scan_both_sides")
    identity_path = out / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Calibration model or development inputs changed; use a new directory")
    atomic_json(identity_path, identity)
    all_norms = []
    cache = GradientCache(out, probe, model_identity=model, data_fields=provenance,
                          context_length=512, prediction_length=64)
    try:
        with selection_execution(probe):
            for side in ("reference", "source"):
                for fragment in load_fragments(inputs / f"{side}.npz"):
                    path = out / f"{side}-{fragment.fragment_id}.npz"
                    starts = fragment.valid_starts(512, 64, 1)
                    if path.exists():
                        with np.load(path, allow_pickle=False) as saved:
                            norms = saved["norms"]
                    else:
                        parts = []
                        reader = FragmentGradients(cache, side)
                        for lo in range(0, len(starts), 256):
                            gradients = reader(fragment, starts[lo:lo+256])
                            parts.append(np.linalg.norm(gradients.astype(np.float64), axis=1))
                            progress = dict(side=side, fragment_id=fragment.fragment_id,
                                fragment_windows_done=min(lo+256,len(starts)), fragment_windows=len(starts),
                                cached_windows=len(cache.rows), gradient_batch=8, gradient_workers=4)
                            atomic_json(out / 'progress.json', progress)
                            print('CALIBRATION_PROGRESS', progress, flush=True)
                        norms = np.concatenate(parts)
                        if not np.all(np.isfinite(norms)):
                            raise ValueError("Nonfinite real gradient norm in threshold calibration")
                        temporary = path.with_suffix(".tmp")
                        with temporary.open("wb") as stream:
                            np.savez_compressed(stream, starts=starts, norms=norms)
                        temporary.replace(path)
                    all_norms.append(norms)
    finally:
        cache.close()
    norms = np.concatenate(all_norms)
    report = dict(model=model, inputs=provenance, count=len(norms), epsilon=float(np.percentile(norms, 10)),
        percentile=10, distribution={str(p): float(np.percentile(norms, p)) for p in (1, 5, 10, 25, 50, 75, 90, 99)},
        rule="p10 over every source/reference development window gradient norm; epsilon_mu=epsilon_g",
        norm_precision="float64 CPU norm of BF16 cached gradient sketches")
    atomic_json(out / "epsilon.json", report)
    print("REAL_THRESHOLD_CALIBRATION_COMPLETE", student, report["epsilon"], flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out")
    parser.add_argument("--student", choices=STUDENTS)
    parser.add_argument("--model-path")
    parser.add_argument("--import-json", action="store_true")
    args = parser.parse_args(argv)
    if args.import_json:
        import_inputs(args.inputs, json.load(sys.stdin))
        return
    if not args.student or not args.out:
        parser.error("calibration requires --student and --out")
    torch.set_num_threads(4)
    calibrate(args.student, args.inputs, args.out, model_path=args.model_path)


if __name__ == "__main__":
    main()
