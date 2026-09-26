"""Development-history schedule calibration with online metrics.

This is NOT a replacement test table. Never use RAGS-vs-Random test gains to
select a schedule. The initial calibration uses only the unconditioned Random
selection and validation windows strictly preceding the formal exam.
The full-pool exploration entry optionally persists the latest optimizer state.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
import random
from pathlib import Path
import time

import numpy as np
import torch

from .students import STUDENTS, ENVIRONMENTS, adapter_engine, atomic_json
from .train_eval import evaluate, fine_tune


def validation_from_history(fragments, *, context_length=512, prediction_length=64, count=8,
                            gap_windows=0, seasonality=None):
    from .metrics import history_seasonal_scale, seasonality_for
    windows, entities, scales, boundaries = [], [], [], []
    for fragment in fragments:
        stop = len(fragment.data) - gap_windows * prediction_length
        values = fragment.data[:stop]
        first = len(values) - count * prediction_length - context_length
        if first < context_length + prediction_length:
            raise ValueError(f"Insufficient development history: {fragment.entity_id}")
        for start in range(first, len(values) - context_length, prediction_length):
            windows.append(values[start:start + context_length + prediction_length])
            entities.append(fragment.entity_id)
            scales.append(history_seasonal_scale(values[:start + context_length],
                                                   seasonality if seasonality is not None else seasonality_for(fragment.entity_id)))
        boundaries.append(dict(entity=fragment.entity_id, reference_stop_exclusive=first,
            validation_first_window=first, validation_first_target=first + context_length,
            validation_stop_exclusive=len(values)))
    return dict(test_windows=np.stack(windows), test_entities=entities,
                test_scales=np.asarray(scales), context_length=context_length), boundaries


def online_curve(engine, windows, *, base_state, learning_rate, seed, checkpoints,
                 batch_size, evaluate_fn, on_record=None, sampling_probabilities=None,
                 resume_path=None, observations_path=None, weight_decay=0.01,
                 training_options=None):
    """Keep one continuous optimizer; evaluation cannot consume training RNG."""
    observations_path = Path(observations_path) if observations_path is not None else None
    records = (json.loads(observations_path.read_text()) if observations_path is not None
               and observations_path.exists() else [])
    started = time.monotonic()
    eval_seconds = 0.
    step_epochs = {}
    for epoch, step in checkpoints:
        step_epochs.setdefault(step, []).append(epoch)
    device = torch.device(engine.device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []

    def observe(step):
        nonlocal eval_seconds
        if step not in step_epochs:
            return
        was_training = engine.model.training
        eval_started = time.monotonic()
        numpy_state, python_state = np.random.get_state(), random.getstate()
        try:
            with torch.random.fork_rng(devices=devices):
                metrics = evaluate_fn()
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            engine.model.train(was_training)
        eval_seconds += time.monotonic() - eval_started
        for epoch in step_epochs[step]:
            record = dict(epoch=epoch, step=step, metrics=metrics,
                          sample_exposures=step * batch_size,
                          training_seconds=time.monotonic() - started - eval_seconds,
                          evaluation_seconds=eval_seconds)
            if hasattr(windows, "pool_n"):
                record["actual_epoch"] = step * batch_size / windows.pool_n
            records[:] = [previous for previous in records if previous["epoch"] != epoch]
            records.append(record)
            records.sort(key=lambda item: item["epoch"])
            if observations_path is not None:
                atomic_json(observations_path, records)
            if on_record:
                on_record(record)

    engine.model.load_state_dict(base_state)
    try:
        log = fine_tune(engine, windows, steps=max(step_epochs), batch_size=batch_size,
                        learning_rate=learning_rate, seed=seed, on_step=observe,
                        sampling_probabilities=sampling_probabilities,
                        resume_path=resume_path, checkpoint_steps=list(step_epochs),
                        weight_decay=weight_decay, training_options=training_options)
        return dict(curve=records, train=log, learning_rate=learning_rate, seed=seed,
                    checkpoint_policy=("online_metrics_and_latest_optimizer_state" if resume_path is not None
                                       else "online_metrics_only"), evaluation_seconds=eval_seconds)
    finally:
        engine.model.load_state_dict(base_state)
        engine.model.eval()


def build_student(student, model_path=None):
    if student in ("chronos-base", "chronos-small"):
        from rags.engines import ChronosT5Config, ChronosT5Engine, DEFAULT_CHRONOS_T5_SMALL, DEFAULT_CHRONOS_T5_BASE
        default = DEFAULT_CHRONOS_T5_BASE if student == "chronos-base" else DEFAULT_CHRONOS_T5_SMALL
        return ChronosT5Engine(ChronosT5Config(context_length=512, prediction_length=64,
            model_path=model_path or default), device="cuda")
    adapter = _foundation_adapter(student, model_path)
    return adapter_engine(adapter, 512, 64)


def _foundation_adapter(student, model_path):
    # Foundation adapter honoring the pinned environment's HF cache.
    import subprocess
    from adapters.foundation import build_adapter
    adapter = build_adapter(student, device="cuda")
    if model_path:
        adapter.spec = replace(adapter.spec, model_id=model_path)
        return adapter
    environment = ENVIRONMENTS.get(student)
    if "/" in environment:
        prefix = [environment]
    else:
        import shutil
        conda = shutil.which("conda") or os.environ.get("CONDA_EXE")
        if conda is None:
            # Frozen envs may have no conda shim on PATH; fall back to the
            # per-model HF cache convention before the data disk finishes.
            fallback = Path("/root/autodl-tmp/hf-integration/hub") / adapter.spec.model_id
            if fallback.is_dir():
                adapter.spec = replace(adapter.spec, model_id=str(fallback))
            return adapter
        prefix = [conda, "run", "--no-capture-output", "-n", environment]
    result = subprocess.run(prefix + [
        "-c", "import json, os, sys; print(json.dumps([os.environ.get('HF_HUB_CACHE'), sys.executable]))"],
        capture_output=True, text=True, check=True)
    cache, executable = json.loads(result.stdout.strip().splitlines()[-1])
    if cache and Path(cache).is_dir():
        cache_root = Path(cache)
    else:
        env_root = Path(executable).resolve().parents[1]
        cache_root = env_root / "share" / "hf-integration" / "hub"
    if (cache_root / adapter.spec.model_id).is_dir():
        adapter.spec = replace(adapter.spec, model_id=str(cache_root / adapter.spec.model_id))
    return adapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", choices=STUDENTS, required=True)
    parser.add_argument("--random-selection", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--datasets", nargs="+", default=["boomlet_1676", "boomlet_619"])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[1e-6, 3e-6, 1e-5])
    parser.add_argument("--epochs", nargs="+", type=float, default=[.25, .5, 1., 2., 4.])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--validation-windows", type=int, default=8)
    parser.add_argument("--validation-gap-windows", type=int, default=0,
                        help="Move validation earlier by this many horizons; use a new validation directory")
    parser.add_argument("--eval-batch", type=int, default=32)
    parser.add_argument("--validation-dir", help="small temporary prepared inputs for isolated model environments")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepare-probes", help="write small pre-validation gradient-audit inputs, then exit")
    parser.add_argument("--gpu-memory-fraction", type=float, help="allocator cap for concurrent jobs; no batch change")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="recompute activations; never save training weights")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    if args.gpu_memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Fixed source selection independent of the target. Reuse identities, no feature cache.
    from .regions import load_selection
    selected = load_selection(args.random_selection)
    if selected.name != "random":
        raise ValueError("Neutral schedule calibration requires the Random selection")
    windows = selected.selected
    n = getattr(windows, "budget_windows", len(windows))
    checkpoints = [(epoch, math.ceil(epoch * n / 32)) for epoch in args.epochs]
    if args.prepare_probes:
        from .tasks import build_task_specs, build_task_data
        probe_windows, probe_ids = [], []
        for spec in build_task_specs(keywords=tuple(args.datasets)):
            task = build_task_data(spec, context_length=512, prediction_length=64, pool_fragments=[])
            for fragment in task.reference_fragments:
                stop = len(fragment.data) - (args.validation_windows + args.validation_gap_windows) * 64 - 512
                for start in np.linspace(0, stop - 576, 2).astype(int):
                    probe_windows.append(fragment.data[start:start + 576])
                    probe_ids.append([fragment.entity_id, int(start), int(start + 576)])
        np.savez_compressed(args.prepare_probes, windows=np.stack(probe_windows), ids=json.dumps(probe_ids))
        print("PROBE_INPUTS_PREPARED", flush=True)
        return 0
    validation, boundaries = {}, {}
    prepared = Path(args.validation_dir) if args.validation_dir else None
    if prepared is not None and (prepared / "boundaries.json").exists():
        boundaries = json.loads((prepared / "boundaries.json").read_text())
        for dataset in args.datasets:
            with np.load(prepared / (dataset + ".npz")) as payload:
                validation[dataset] = dict(test_windows=payload["windows"],
                    test_entities=payload["entities"].tolist(), test_scales=payload["scales"], context_length=512)
    else:
        from .tasks import build_task_specs, build_task_data
        for spec in build_task_specs(keywords=tuple(args.datasets)):
            task = build_task_data(spec, context_length=512, prediction_length=64, pool_fragments=[])
            validation[spec.dataset], boundaries[spec.dataset] = validation_from_history(
                task.reference_fragments, count=args.validation_windows,
                gap_windows=args.validation_gap_windows)
        if prepared is not None:
            prepared.mkdir(parents=True, exist_ok=True)
            for dataset, values in validation.items():
                np.savez_compressed(prepared / (dataset + ".npz"), windows=values["test_windows"],
                    entities=np.asarray(values["test_entities"]), scales=values["test_scales"])
            atomic_json(prepared / "boundaries.json", boundaries)
    if args.prepare_only:
        if prepared is None:
            parser.error("--prepare-only requires --validation-dir")
        print("VALIDATION_PREPARED", flush=True)
        return 0
    protocol = dict(arguments=vars(args), validation_boundaries=boundaries,
        source_window_ids=selected.window_ids, budget_windows=n,
        purpose="development_random_schedule_calibration_not_test_results",
        selection_reference_rule="exclude full validation windows from all future target-guided selection")
    atomic_json(out / f"invocation-{time.time_ns()}.json", protocol)
    if not (out / "protocol.json").exists():
        atomic_json(out / "protocol.json", protocol)
    engine = build_student(args.student, args.model_path)
    if args.gradient_checkpointing:
        engine.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base = {key: value.detach().cpu().clone() for key, value in engine.model.state_dict().items()}

    def assess():
        scores = {name: evaluate(engine, batch_size=args.eval_batch, **kwargs)
                  for name, kwargs in validation.items()}
        return dict(by_dataset=scores, mean_mase=float(np.mean([s["mase"] for s in scores.values()])))

    original = assess()
    atomic_json(out / "original.json", original)
    for lr in args.learning_rates:
        for seed in args.seeds:
            name = f"lr-{lr:g}-seed-{seed}"
            path = out / (name + ".json")
            if path.exists():
                continue
            torch.cuda.reset_peak_memory_stats()
            def record(value):
                with (out / "observations.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(run=name, **value)) + "\n")
                print(f"{name} epoch={value['epoch']} dev={value['metrics']['mean_mase']:.6f}", flush=True)
            result = online_curve(engine, windows, base_state=base, learning_rate=lr, seed=seed,
                checkpoints=checkpoints, batch_size=32, evaluate_fn=assess, on_record=record)
            result.update(student=args.student, peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30)
            atomic_json(path, result)
    print("CALIBRATION_DONE", flush=True)
    return 0
