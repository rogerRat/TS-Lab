"""One student's production full-pool task loop in its isolated environment."""
import argparse
import fcntl
import gc
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from rags.crb import CRBConfig
from rags.execution import SelectionExecution, selection_execution
from .arms import WindowIndex
from .checkpoint_retention import cleanup_completed_weights
from .development_inputs import load_fragments
from .full_pool import select_full_pool, ARM_BASE, validate_cached_selection
from .full_pool_models import build_model
from .protocol import EXPERIMENT_VERSIONS
from .regions import load_selection, save_selection
from .reference_opinions import selection_options, weight_placement_for
from .resampling import REVISION
from .selection_metrics import SelectionMetrics
from .students import STUDENTS, atomic_json
from .train_eval import evaluate, run_arm_training
from assets.gradient.prepare import prepare_gradients


def _assert_training_enabled(run_dir, *, experiment_version=None):
    run = Path(run_dir)
    control_path = run/'execution_control.json'
    control = json.loads(control_path.read_text()) if control_path.exists() else None
    if (control is not None and (control.get('mode') in ('assets_only', 'prepared')
            or control.get('training_enabled') is False or control.get('evaluation_enabled') is False)):
        raise RuntimeError('Run is '+str(control.get('mode'))+': training and evaluation are paused')
    protocol_path = run/'protocol.json'
    if experiment_version is None and protocol_path.exists():
        experiment_version = json.loads(protocol_path.read_text()).get('experiment_version')
    if experiment_version in EXPERIMENT_VERSIONS and (control is None
            or control.get('training_enabled') is not True or control.get('evaluation_enabled') is not True
            or control.get('selection_enabled') is not True):
        raise RuntimeError('V5 execution controls must enable selection, training and evaluation')


def _validate_completed_cell(path,*,student,dataset,arm,seed,experiment_version='v5'):
    record=json.loads(Path(path).read_text())
    expected=dict(student=student,dataset=dataset,arm=arm,seed=seed,
        experiment_version=experiment_version,resampling_revision=REVISION)
    if any(record.get(k)!=v for k,v in expected.items()):
        raise ValueError('Completed cell identity differs from '+experiment_version+'; use a fresh run directory')


def run_worker(run_dir, student, *, model_bundle=None, original_only=False, datasets=None):
    """Serialize launchers before model loading, including supervisor/watchdog races."""
    _assert_training_enabled(run_dir)
    out = Path(run_dir) / "students" / student
    out.mkdir(parents=True, exist_ok=True)
    # Tagged workers (RAGS_WORKER_TAG) hold distinct locks so deliberate
    # task-disjoint helper workers can overlap the supervisor serial worker;
    # completed-cell identity validation makes overlaps safe skips.
    tag = os.environ.get("RAGS_WORKER_TAG", "")
    with (out / f"worker{tag}.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(student, "WAITING_FOR_WORKER_LOCK", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        _assert_training_enabled(run_dir)
        return _run_worker(run_dir, student, model_bundle=model_bundle,
                           original_only=original_only, datasets=datasets)


def _run_worker(run_dir, student, *, model_bundle=None, original_only=False, datasets=None):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "protocol.json").read_text())
    if config["resampling_revision"] != REVISION or student not in config["students"]:
        raise ValueError("Wrong full-pool protocol/student")
    from .protocol import validate_config
    validate_config(config)
    original_arm = "orig"
    training_arms = [arm for arm in config["arms"] if arm != "orig"]
    settings = config["settings"][student]
    recipe = None
    if 'optimizer' in settings:
        from .train_eval import TrainingRecipe
        recipe = TrainingRecipe.from_settings(dict(settings,
            learning_rates=[settings['learning_rate']],epochs=[settings['epochs']]))
    elif config.get('task_protocol') == 'shared-ten-formal-20260915':
        raise ValueError('Formal shared-ten worker requires the complete frozen training recipe')
    engine, probe, identity = model_bundle or build_model(student,
        model_path=config.get("model_paths", {}).get(student), device=config.get("device", "cuda"),
        context_length=config["context_length"], prediction_length=config["prediction_length"])
    execution = dict(config.get("selection_execution", {}))
    execution.update(config.get("student_selection_execution", {}).get(student, {}))
    execution_path = run_dir / "execution.json"
    if execution_path.exists():
        execution.update(json.loads(execution_path.read_text()).get("student_selection_execution", {}).get(student, {}))
    probe.execution = SelectionExecution(**execution)
    out = run_dir / "students" / student
    out.mkdir(parents=True, exist_ok=True)
    identity_path = out / "model.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Loaded model identity changed; use a new run")
    atomic_json(identity_path, identity)
    crb = CRBConfig(context_length=config['context_length'],
        prediction_length=config['prediction_length'], **config['crb'])
    c, h = crb.context_length, crb.prediction_length
    inputs = Path(config["inputs"])
    pool = load_fragments(inputs / "pool.npz")
    manifest = json.loads((inputs / "pool_manifest.json").read_text())
    index = WindowIndex(pool, context_length=c, prediction_length=h)
    base = {name: value.detach().cpu().clone() for name, value in engine.model.state_dict().items()}
    costs = SelectionMetrics(out, device=engine.device, protocol=dict(revision=REVISION, student=student,
        selection_execution=asdict(probe.execution), timing_context="shared_gpu_parallel_workers" if execution_path.exists()
        else "serial_worker"))
    state_path = out / "state.json"

    def train_dataset(dataset, exam_kwargs, original, selection_paths):
        """Train one task after its static selections are published."""
        if identity.get("training_parameter_dtype") == "torch.bfloat16":
            engine.model.to(dtype=torch.bfloat16)
        for arm, path in selection_paths.items():
            selected = load_selection(path)
            for seed in config["seeds"]:
                cell = cell_dir / f"{dataset}__{arm}__seed{seed}.json"
                checkpoint_root = Path(config.get("checkpoint_root") or out / "checkpoints")
                checkpoint_dir = checkpoint_root / student / dataset / arm / f"seed{seed}"
                clean_completed = config.get("checkpoint_retention") == "completed_cell_cleanup"
                if cell.exists():
                    _validate_completed_cell(cell,student=student,dataset=dataset,arm=arm,seed=seed,experiment_version=config.get('experiment_version'))
                    if clean_completed:
                        cleanup_completed_weights(cell, checkpoint_dir)
                    continue
                _assert_training_enabled(run_dir)
                atomic_json(state_path, dict(stage="training", dataset=dataset, arm=arm, seed=seed, status="running"))
                gc.collect()
                if torch.device(engine.device).type == "cuda":
                    torch.cuda.empty_cache()
                started = time.perf_counter()
                try:
                    result = run_arm_training(engine, source_windows=selected.selected,
                        source_epochs=settings["epochs"], batch_size=(recipe.batch_size if recipe else config["train_batch"]),
                        recipe=recipe,
                        learning_rate=settings["learning_rate"], seed=seed, base_state=base,
                        checkpoint_dir=checkpoint_dir, online_evaluation=config.get("online_evaluation", False),
                        original_mase=json.loads(original.read_text())["mase"], **exam_kwargs)
                    atomic_json(cell, dict(result, student=student, dataset=dataset, arm=arm, seed=seed,
                        pool_equivalents=manifest["total_equivalents"], selection_artifact=str(path),
                        experiment_version=config.get("experiment_version"),
                        crb_split_policy=crb.split_policy, resampling_revision=REVISION, elapsed_seconds=time.perf_counter() - started))
                    if clean_completed:
                        cleanup_completed_weights(cell, checkpoint_dir)
                    print(student, dataset, arm, seed, "CELL_COMPLETE", flush=True)
                except Exception as error:
                    atomic_json(state_path, dict(stage="training", dataset=dataset, arm=arm, seed=seed,
                                                 status="failed", error=repr(error)))
                    raise
            del selected
    for dataset in (datasets if datasets is not None else config["datasets"]):
        _assert_training_enabled(run_dir)
        if identity.get("training_parameter_dtype") == "torch.bfloat16":
            engine.model.to(dtype=torch.bfloat16)
        cell_dir = out / "cells"
        cell_dir.mkdir(exist_ok=True)
        exam_path = inputs / f"{dataset}__exam.npz"
        with np.load(exam_path, allow_pickle=False) as exam:
            exam_kwargs = dict(test_windows=exam["windows"], test_entities=exam["entities"].tolist(),
                               test_scales=exam["scales"], context_length=c)
        original = cell_dir / f"{dataset}__{original_arm}__seed0.json"
        if original.exists():
            _validate_completed_cell(original,student=student,dataset=dataset,arm=original_arm,seed=0,experiment_version=config.get('experiment_version'))
        if not original.exists():
            costs.original(dataset, arm=original_arm)
            metrics = evaluate(engine, batch_size=config.get("eval_batch", 32), **exam_kwargs)
            atomic_json(original, dict(student=student, dataset=dataset, arm=original_arm, seed=0,
                experiment_version=config.get("experiment_version"), resampling_revision=REVISION, **metrics))
        if original_only:
            continue
        # Complete selection before training. Training never opens reference data.
        if identity.get("scoring_parameter_dtype") == "torch.bfloat16":
            engine.model.to(dtype=torch.bfloat16)
        engine.model.load_state_dict(base)
        refs = load_fragments(inputs / f"{dataset}__reference.npz")
        atomic_json(state_path, dict(stage="assets", dataset=dataset, status="running",
                                    epsilon_policy=crb.epsilon_policy))
        needed = [arm for arm in training_arms if not
                  (out / "selections" / f"{dataset}__{arm}.npz").exists()]
        method_names = [ARM_BASE.get(arm, arm) for arm in needed]
        assets = None
        if any(arm in ("rags", "less", "rags_traj") for arm in method_names):
            # Runtime-only throughput tuning for the asset-first queue.
            override_path = run_dir / "asset_execution_override.json"
            if override_path.exists():
                override = json.loads(override_path.read_text())
                probe.execution = SelectionExecution(**dict(asdict(probe.execution), **override))
            preparation_started = time.perf_counter()
            with selection_execution(probe):
                assets = prepare_gradients(method_names, probe, run_dir=run_dir,
                    reference_fragments=refs, pool=pool, index=index, pool_manifest=manifest,
                    model_identity=identity, crb_config=crb, dataset_tag=dataset)
            atomic_json(out / f"{dataset}__gradient_assets.json", dict(
                path=str(assets.cache.path), computed_windows=assets.cache.computed_windows,
                total_cached_windows=len(assets.cache.rows), model=identity,
                preparation_seconds=time.perf_counter()-preparation_started,
                preparation_gradient_seconds=assets.cache.computed_gradient_seconds,
                preparation_scope="shared_method_union",method_costs=assets.method_costs))
            # Compute cosine tiles once per representation, only for missing
            # arms in this task. No cross-task, model, or checkpoint reuse.
            for arm in needed:
                family = ARM_BASE.get(arm, arm)
                if family in ("rags", "rags_traj"):
                    modes = assets.selection_aggregations.setdefault(family, [])
                    mode = selection_options(arm, config)["aggregation"]
                    if mode not in modes:
                        modes.append(mode)
        selection_paths = {}
        for arm in training_arms:
            temperatures = settings["temperature"]
            temperature = (temperatures.get(arm, 1.) if isinstance(temperatures, dict) else temperatures)
            path = out / "selections" / f"{dataset}__{arm}.npz"
            atomic_json(state_path, dict(stage="selection", dataset=dataset, arm=arm, status="running"))
            method_name = ARM_BASE.get(arm, arm)
            gradient_cost = (dict(assets.method_costs[method_name], benchmark=assets.cache.gradient_benchmark())
                             if assets is not None and method_name in assets.method_costs else None)
            options = selection_options(arm, config)
            weight_placement = weight_placement_for(arm, config)
            try:
                if path.exists():
                    output = costs.measure(dataset, arm, lambda: load_selection(path), selection_cached=True, gradient_cost=gradient_cost)
                    validate_cached_selection(output.selected.metadata, arm=arm, identity=identity,
                        manifest=manifest, crb=crb, temperature=temperature)
                else:
                    with selection_execution(probe):
                        output = costs.measure(dataset, arm, lambda: select_full_pool(arm, probe,
                            reference_fragments=refs, pool=pool, index=index, pool_manifest=manifest,
                            model_identity=identity, crb_config=crb, cache_dir=out / "method_cache",
                            run_dir=run_dir, gradient_assets=assets,
                            dataset_tag=dataset, temperature=temperature,
                            reference_aggregation=options["aggregation"], competition=options["competition"],
                            reference_weight_placement=weight_placement), gradient_cost=gradient_cost)
                    save_selection(path, output)
                selection_paths[arm] = path
                del output
            except Exception as error:
                atomic_json(state_path, dict(stage="selection", dataset=dataset, arm=arm,
                                             status="failed", error=repr(error)))
                raise
        del refs
        del assets
        # Operational-only preparation barrier: it does not alter scientific
        # configuration or saved selections.  It lets every task materialize
        # its reusable gradient assets before any arm begins training.
        if (run_dir / "asset_preparation_only").exists():
            gc.collect()
            if torch.device(engine.device).type == "cuda":
                torch.cuda.empty_cache()
            continue
        gc.collect()
        if torch.device(engine.device).type == "cuda":
            torch.cuda.empty_cache()
        train_dataset(dataset, exam_kwargs, original, selection_paths)
    atomic_json(state_path, dict(status="original_complete" if original_only else "complete", student=student))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--student", choices=STUDENTS, required=True)
    parser.add_argument("--original-only", action="store_true")
    parser.add_argument("--tasks", default=None,
                        help="Comma-separated dataset subset; protocol order is preserved")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    tasks = args.tasks.split(",") if args.tasks else None
    run_worker(args.run, args.student, original_only=args.original_only, datasets=tasks)


if __name__ == "__main__":
    main()
