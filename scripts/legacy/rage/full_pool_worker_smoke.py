"""Real-data, tiny production-worker acceptance; not a scientific experiment."""
import argparse
from pathlib import Path

import numpy as np
import torch

from rage.core import TimeSeriesFragment
from rage.experiments.development_inputs import load_fragments, save_fragments
from rage.experiments.full_pool_worker import run_worker
from rage.experiments.metrics import history_seasonal_scale, seasonality_for
from rage.experiments.resampling import REVISION
from rage.experiments.students import STUDENTS, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", choices=STUDENTS, required=True)
    parser.add_argument("--calibration-inputs", required=True)
    parser.add_argument("--epsilon", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--out", required=True)
    parser.add_argument("--arms", nargs="+", default=["random", "rage"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = Path(args.out).resolve()
    inputs = out / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    source = load_fragments(Path(args.calibration_inputs) / "source.npz")[-1:]
    reference = load_fragments(Path(args.calibration_inputs) / "reference.npz")[:1]
    ref = reference[0]
    history = TimeSeriesFragment(ref.data[:-576].copy(), entity_id=ref.entity_id,
                                 fragment_id=ref.fragment_id)
    save_fragments(inputs / "pool.npz", source)
    save_fragments(inputs / "real__reference.npz", [history])
    atomic_json(inputs / "pool_manifest.json", dict(total_equivalents=sum(len(f.data) // 576 for f in source),
        purpose="real_development_acceptance_not_formal_pool", source_entities=[f.entity_id for f in source]))
    np.savez_compressed(inputs / "real__exam.npz", windows=ref.data[-576:][None],
        entities=np.array([ref.entity_id]),
        scales=np.array([history_seasonal_scale(ref.data[:-64], seasonality_for(ref.entity_id))]))
    config = dict(purpose="real_production_entry_acceptance_not_formal_results", resampling_revision=REVISION,
        students=[args.student], datasets=["real"], arms=args.arms, seeds=[0], context_length=512,
        prediction_length=64, train_batch=2, eval_batch=1, inputs=str(inputs),
        model_paths={args.student: args.model_path},
        crb=dict(cover_density=.02, error_density=.02, split_density=.005, curvature_cap=1.),
        settings={args.student: dict(epsilon_calibration=str(Path(args.epsilon).resolve()),
            epochs=.5, learning_rate=1e-6, temperature=.1)},
        selection_execution=dict(gradient_batch=1, inference_batch=1, cpu_threads=4, generation="eager"))
    atomic_json(out / "protocol.json", config)
    run_worker(out, args.student)
    print("REAL_PRODUCTION_WORKER_SMOKE_COMPLETE", args.student, flush=True)


if __name__ == "__main__":
    main()
