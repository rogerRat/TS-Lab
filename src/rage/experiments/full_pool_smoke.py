"""Real pretrained models, synthetic source/ref/exam: execution, not efficacy."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from rage.core import TimeSeriesFragment
from rage.crb import CRBConfig
from rage.execution import SelectionExecution, selection_execution
from .arms import WindowIndex
from .full_pool import ARMS, ARM_BASE, select_full_pool
from .ranking import DEFAULT_TEMPERATURE
from assets.gradient.prepare import prepare_gradients
from .full_pool_models import build_model
from .regions import load_selection, save_selection
from .students import STUDENTS, atomic_json
from .train_eval import evaluate, fine_tune


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", choices=STUDENTS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=[a for a in ARMS if a != "orig"])
    parser.add_argument("--source-trajectories",type=int,default=20)
    parser.add_argument("--reference-trajectories",type=int,default=2)
    parser.add_argument("--trajectory-length",type=int,default=632)
    args = parser.parse_args(argv)
    if args.source_trajectories<1 or args.reference_trajectories<1 or args.trajectory_length<576:
        parser.error("Need positive source/ref counts and length >= C+H")
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    engine, probe, identity = build_model(args.student, model_path=args.model_path)
    probe.execution = SelectionExecution(gradient_batch=1, inference_batch=2, generation="eager")
    rng = np.random.default_rng(914)
    fragments = [TimeSeriesFragment((np.sin(np.arange(args.trajectory_length) / (10 + i % 5)) +
        rng.normal(0, .03, args.trajectory_length)).astype(np.float32), entity_id=f"synthetic::{i}", fragment_id=i)
        for i in range(args.source_trajectories+args.reference_trajectories+2)]
    cut=args.source_trajectories;end=cut+args.reference_trajectories
    pool,refs,exam=fragments[:cut],fragments[cut:end],fragments[end:]
    index = WindowIndex(pool, context_length=512, prediction_length=64)
    manifest = dict(total_equivalents=len(pool)*(args.trajectory_length//576), total_points=len(pool)*args.trajectory_length, lengths=[args.trajectory_length]*len(pool),
                    seed=914, source="synthetic_smoke_not_formal")
    # Frozen V5 has no threshold or P10 calibration.
    config = CRBConfig(context_length=512,prediction_length=64,cover_density=.02,error_density=.01,split_density=.01)
    eval_kwargs = dict(test_windows=np.stack([f.data[:576] for f in exam]),
        test_entities=[f.entity_id for f in exam], test_scales=np.ones(2), context_length=512, batch_size=1)
    base = {key: value.detach().cpu().clone() for key, value in engine.model.state_dict().items()}
    record = dict(purpose="real pretrained model synthetic full-pool smoke; not efficacy or calibration",
                  model=identity, manifest=manifest, arms={})
    protocol_path=args.out/"protocol.json"
    protocol=dict(model=identity,manifest=manifest,arms=args.arms,crb=asdict(config),reference_count=len(refs))
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:
        raise ValueError("Smoke protocol differs; use a new output directory")
    atomic_json(protocol_path,protocol)
    atomic_json(args.out / "model.json", identity)
    record["original"] = evaluate(engine, **eval_kwargs)
    with selection_execution(probe):
        assets=prepare_gradients([ARM_BASE[a] for a in args.arms if a != "orig"],probe,
            run_dir=args.out,reference_fragments=refs,pool=pool,index=index,pool_manifest=manifest,
            model_identity=identity,crb_config=config,dataset_tag="synthetic")
    for name in args.arms:
        if name == "orig":continue
        completed = args.out / f"{name}.json"
        if completed.exists():
            continue
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        with selection_execution(probe):
            output = select_full_pool(name, probe, reference_fragments=refs, pool=pool, index=index,
                pool_manifest=manifest, model_identity=identity, crb_config=config,
                cache_dir=args.out / "method_cache", run_dir=args.out, dataset_tag="synthetic",gradient_assets=assets,
                temperature=.05 if name=="lp-mirror" else DEFAULT_TEMPERATURE)
        artifact = args.out / f"{name}.npz"
        save_selection(artifact, output)
        restored = load_selection(artifact)
        np.testing.assert_array_equal(restored.selected.probabilities, output.selected.probabilities)
        # LTSV must restore the real model too, not just a tiny fixture.
        for key, value in engine.model.state_dict().items():
            if not torch.equal(value.cpu(), base[key]):
                raise ValueError(f"Selection changed pretrained weights: {name}/{key}")
        training = fine_tune(engine, restored.selected, steps=1, batch_size=1,
                             learning_rate=1e-6, seed=0)
        metrics = evaluate(engine, **eval_kwargs)
        result = dict(arm=name, train=training, metrics=metrics, cost=vars(output.cost),
                      diagnostics=output.diagnostics, elapsed_seconds=time.perf_counter() - started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated())
        atomic_json(completed, result)
        record["arms"][name] = result
        atomic_json(args.out / "smoke.json", record)
        print(name, "REAL_FULL_POOL_TRAIN_EVAL_OK", flush=True)
        engine.model.load_state_dict(base)
        engine.model.eval()
    print("REAL_FULL_POOL_SMOKE_COMPLETE", args.student, flush=True)


if __name__ == "__main__":
    main()
