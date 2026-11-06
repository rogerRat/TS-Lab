"""Prepare per-window BF16 gradient assets for Chronos, TimesFM or TimeMoE."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from assets.gradient import GradientCache, WindowRequest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("chronos-base", "timesfm-2.5", "timemoe-200m"), required=True)
    parser.add_argument("--ckpt", "--model-path", dest="ckpt")
    parser.add_argument("--windows", type=Path, required=True,
                        help="NPZ with windows[N,C+H] and window_ids[N,2]")
    parser.add_argument("--run-dir", "--output-dir", dest="run_dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True, help="immutable dataset/version identifier")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gradient-batch-size", type=int, default=0, help="0: memory-aware automatic batch")
    parser.add_argument("--gradient-workers", type=int, default=2)
    args = parser.parse_args(argv)
    with np.load(args.windows, allow_pickle=False) as payload:
        windows = payload["windows"]
        ids = payload["window_ids"]
    if windows.ndim != 2 or ids.shape != (len(windows), 2):
        raise ValueError("windows and window_ids must align")
    from rage.experiments.full_pool_models import build_model
    from rage.execution import SelectionExecution, selection_execution
    _, probe, identity = build_model(args.model, model_path=args.ckpt, device=args.device)
    probe.execution = SelectionExecution(gradient_batch=args.gradient_batch_size,
                                         gradient_workers=args.gradient_workers)
    cache = GradientCache(args.run_dir, probe, model_identity=identity,
        data_fields=dict(dataset=args.dataset, path=str(args.windows.resolve()), rows=len(windows)),
        context_length=probe.config.context_length, prediction_length=probe.config.prediction_length)
    try:
        with selection_execution(probe):
            cache.prepare([WindowRequest(args.dataset, ids, lambda rows: windows[rows])])
        print(json.dumps(dict(path=str(cache.path), computed_windows=cache.computed_windows)))
    finally:
        cache.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
