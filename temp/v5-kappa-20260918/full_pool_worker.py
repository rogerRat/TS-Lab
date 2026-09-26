"""One student's production full-pool task loop in its isolated environment."""
import argparse
import fcntl
import gc
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from rags.crb import CRBConfig
from rags.execution import SelectionExecution, selection_execution
from .arms import WindowIndex
from .checkpoint_retention import cleanup_completed_weights
from .development_inputs import load_fragments
from .full_pool import RAGS_PROBABILITY_REVISION, COSINE_NUMERICS, select_full_pool, TEMPERATURE_ARM_BASE
from .full_pool_models import build_model
from .regions import load_selection, save_selection
from .reference_opinions import selection_options, probability_revision, weight_placement_for
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
    if experiment_version in ('v3', 'v4') and (control is None
            or control.get('training_enabled') is not True or control.get('evaluation_enabled') is not True
            or control.get('selection_enabled') is not True):
        message=('V3 is prepared only: user authorization is required before experiment launch'
                 if experiment_version=='v3' else 'V4 is prepared only: experiment execution controls must be enabled')
        raise RuntimeError(message)


def run_worker(run_dir, student, *, model_bundle=None, original_only=False, datasets=None):
    """Serialize launchers before model loading, including supervisor/watchdog races."""
    _assert_training_enabled(run_dir)
    out = Path(run_dir) / "students" / student
    out.mkdir(parents=True, exist_ok=True)
    with (out / "worker.lock").open("a+") as lock:
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
    if config.get("experiment_version") == "v4":
        from .v4_protocol import validate_v4_config
        validate_v4_config(config)
    if config.get("experiment_version") == "v3":
        from .v3_protocol import validate_v3_config
        validate_v3_config(config)
    original_arm = "orig" if config.get("experiment_version") in ("v3", "v4") else "original"
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
        model_path=config.get("model_paths", {}).get(student), device=config.get("device", "cuda"))
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
    from rags.epsilon_protocol import load_p10_calibration
    local_p10 = config["crb"].get("epsilon_policy") == "trajectory_p10"
    no_threshold = config["crb"].get("epsilon_policy") == "none"
    calibration = None if (local_p10 or no_threshold) else load_p10_calibration(settings["epsilon_calibration"], identity)
    c, h = config["context_length"], config["prediction_length"]
    crb = CRBConfig(context_length=c, prediction_length=h,
        **({} if (local_p10 or no_threshold) else dict(mean_epsilon=calibration["epsilon"], endpoint_epsilon=calibration["epsilon"])),
        **config["crb"])
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
        method_names = [TEMPERATURE_ARM_BASE.get(arm, arm) for arm in needed]
        assets = None
        if any(arm in ("rags", "less", "rags_traj", "rags_uniform") for arm in method_names):
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
        selection_paths = {}
        for arm in training_arms:
            temperatures = settings["temperature"]
            temperature = (temperatures.get(arm, 1.) if isinstance(temperatures, dict) else temperatures)
            path = out / "selections" / f"{dataset}__{arm}.npz"
            atomic_json(state_path, dict(stage="selection", dataset=dataset, arm=arm, status="running"))
            method_name = TEMPERATURE_ARM_BASE.get(arm, arm)
            gradient_cost = (dict(assets.method_costs[method_name], benchmark=assets.cache.gradient_benchmark())
                             if assets is not None and method_name in assets.method_costs else None)
            options = selection_options(arm, config)
            weight_placement = weight_placement_for(arm, config)
            try:
                if path.exists():
                    output = costs.measure(dataset, arm, lambda: load_selection(path), selection_cached=True, gradient_cost=gradient_cost)
                    if (output.selected.metadata["pool_manifest"] != manifest or
                            output.selected.metadata.get("model", identity) != identity):
                        raise ValueError("Cached probability artifact belongs to another pool/model")
                    if TEMPERATURE_ARM_BASE.get(arm, arm) == "less":
                        from baselines.less.runner import assert_probability_revision
                        assert_probability_revision(output.selected.metadata, **options, weight_placement=weight_placement)
                    if TEMPERATURE_ARM_BASE.get(arm, arm) in ("rags", "rags_uniform", "rags_traj"):
                        expected_weighting = ("equal_trajectory" if TEMPERATURE_ARM_BASE.get(arm, arm) == "rags_traj"
                                              else "equal_block")
                        if (output.selected.metadata.get("rags_probability_revision") != probability_revision(
                                RAGS_PROBABILITY_REVISION, **options, weight_placement=weight_placement)
                                or output.selected.metadata.get("source_weighting") != expected_weighting):
                            raise ValueError("RAGS reference/trajectory probability changed; use a new selection/run directory")
                    if method_name in ("rags", "rags_traj", "rags_uniform", "less"):
                        recorded = dict(aggregation=output.selected.metadata.get("opinion_aggregation", "linearSum"),
                                        competition=output.selected.metadata.get("competition", "global"))
                        if (recorded != options or output.selected.metadata.get("opinion_weight_placement", "after_transform") != weight_placement):
                            raise ValueError("Cached reference aggregation differs; use a new selection/run directory")
                    recorded_temperature = output.selected.metadata.get("temperature")
                    if recorded_temperature is not None and recorded_temperature != temperature:
                        raise ValueError(f"Cached selection temperature {recorded_temperature} does not"
                                         f" match configured {temperature} for arm {arm}")
                    if TEMPERATURE_ARM_BASE.get(arm, arm) in ("rags", "less", "rags_traj", "rags_uniform") and output.selected.metadata.get("cosine_numerics") != COSINE_NUMERICS:
                        raise ValueError("Gradient cosine normalization changed; use a new selection/run directory")
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
