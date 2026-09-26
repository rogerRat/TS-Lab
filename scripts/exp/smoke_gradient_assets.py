"""Bounded real-weight check: two cached windows and one BF16 training step."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from assets.gradient import GradientCache, WindowRequest, GradientDataset
from rags.execution import SelectionExecution, selection_execution
from rags.experiments.full_pool_models import build_model
from rags.experiments.train_eval import fine_tune


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gradient-batch", type=int, default=1)
    parser.add_argument("--gradient-workers", type=int, default=2)
    parser.add_argument("--incremental-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    engine, probe, identity = build_model(args.model, model_path=args.model_path)
    assert {p.dtype for p in engine.model.parameters()} == {torch.bfloat16}
    probe.execution = SelectionExecution(gradient_batch=args.gradient_batch,
        gradient_workers=args.gradient_workers, inference_batch=1, generation="eager")
    if args.incremental_only:
        windows = np.stack([np.sin(np.arange(576) / 15 + i * .3) for i in range(5)]).astype(np.float32)
        def requirement(rows):
            rows = np.asarray(rows)
            return WindowRequest("synthetic", np.column_stack([np.ones(len(rows)), rows + 1]),
                                 lambda positions: windows[rows[positions]])
        counts = []
        for rows in ([0, 1, 2], [2, 3, 4, 3]):
            cache = GradientCache(args.run_dir, probe, model_identity=identity,
                data_fields={"purpose": "incremental-five-windows-v1"},
                context_length=512, prediction_length=64)
            try:
                cache.prepare([requirement(rows)])
                counts.append(cache.computed_windows)
            finally:
                cache.close()
            if len(counts) == 1:
                first = GradientDataset(cache.path).lookup("synthetic", requirement(rows).ids)
        dataset = GradientDataset(cache.path)
        assert len(dataset) == 5
        assert counts in ([3, 2], [0, 0]), counts
        np.testing.assert_array_equal(dataset.lookup("synthetic", requirement([0, 1, 2]).ids), first)
        assert getattr(probe, "_gradient_pool", None) is None
        print(json.dumps(dict(model=args.model, computed_windows=counts, merged_windows=len(dataset),
            cache=str(cache.path), execution=getattr(probe, "last_gradient_execution", {}))), flush=True)
        return
    windows = np.stack([np.sin(np.arange(576) / 15 + shift) for shift in (0., .3)]).astype(np.float32)
    cache = GradientCache(args.run_dir, probe, model_identity=identity,
        data_fields={"purpose": "two-synthetic-windows-smoke-v1"}, context_length=512, prediction_length=64)
    request = WindowRequest("synthetic", np.array([[1, 1], [1, 2]]), lambda rows: windows[rows])
    try:
        with selection_execution(probe):
            cache.prepare([request])
            first = cache.get(request)
            count = cache.computed_windows
            np.testing.assert_array_equal(first, cache.get(request))
            assert cache.computed_windows == count
        assert np.isfinite(first).all() and np.linalg.norm(first) > 0
    finally:
        cache.close()
    np.testing.assert_array_equal(GradientDataset(cache.path).lookup("synthetic", request.ids), first)
    result = fine_tune(engine, windows, steps=1, batch_size=1, learning_rate=1e-4, seed=42)
    assert all(torch.isfinite(p).all() for p in engine.model.parameters())
    print(json.dumps(dict(model=args.model, dtype="bfloat16", computed_windows=count,
        cache=str(cache.path), shape=first.shape, training=result,
        peak_gpu_mib=torch.cuda.max_memory_allocated() / 2**20)), flush=True)


if __name__ == "__main__":
    main()
