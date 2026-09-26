"""Synthetic all-arm native-model deployment smoke, not efficacy evidence."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from rags.core import TimeSeriesFragment
from rags.execution import SelectionExecution, selection_execution
from .arms import WindowIndex
from .calibration import build_student
from .native_probe import NativeProbe
from .native_selection import ARMS, select_arm
from .students import atomic_json
from .train_eval import fine_tune


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True, choices=["timesfm-2.5", "timemoe-200m"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.24)
    rng = np.random.default_rng(19)
    fragments = [TimeSeriesFragment((np.sin(np.arange(640) / (10 + i)) +
        rng.normal(0, .05, 640)).astype(np.float32), entity_id=f"synthetic::{i}", fragment_id=i)
        for i in range(44)]
    # Upstream caps retrieval k at pool_size/10; 40 ensures the retrieved
    # union has at least four rows, required by its unmodified IVF KDE.
    pool, reference = fragments[:40], fragments[40:]
    index = WindowIndex(pool, context_length=512, prediction_length=64, stride=1)
    engine = build_student(args.student)
    probe = NativeProbe(engine, model_key=args.student)
    probe.execution = SelectionExecution(gradient_batch=1, inference_batch=4)
    base = {name: value.detach().cpu().clone() for name, value in engine.model.state_dict().items()}
    record = dict(student=args.student, purpose="synthetic deployment smoke, not efficacy",
                  source_seed=19, source_trajectories=40, reference_trajectories=4, n=4, arms={})
    for arm in ARMS:
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        with selection_execution(probe):
            selected = select_arm(arm, probe, reference, pool, index, n_select=4,
                                  context_windows=4)
        selection_seconds = time.monotonic() - started
        if arm == "ltsv":
            for name, value in engine.model.state_dict().items():
                assert torch.equal(value.cpu(), base[name]), name
        training = fine_tune(engine, selected.selected, steps=1, batch_size=2, learning_rate=1e-6, seed=0)
        engine.model.load_state_dict(base)
        engine.model.eval()
        record["arms"][arm] = dict(selection_seconds=selection_seconds,
            peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
            legal_window_occurrences=len(selected.selected), diagnostics=selected.diagnostics,
            training=training)
        atomic_json(Path(args.out), record)
        print(arm, "SELECTION_AND_TRAIN_OK", selection_seconds, flush=True)
    print("ALL_NATIVE_ARMS_SMOKE_DONE", flush=True)


if __name__ == "__main__":
    main()
