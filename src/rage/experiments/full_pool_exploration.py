"""Finite development LR x Epoch trials; never open formal exam arrays."""
import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch

from rage.core import TimeSeriesFragment
from .arms import WindowIndex
from .calibration import online_curve, validation_from_history
from .development_inputs import load_fragments, save_fragments
from .full_pool_models import build_model
from .resampling import REVISION, random_distribution
from .schedule import exposure_checkpoints
from .students import STUDENTS, atomic_json
from .train_eval import evaluate


def development_probability(config, out, probe, identity, pool, manifest, history, boundaries):
    """Use only pre-validation references for the same native scoring kernels."""
    from rage.crb import CRBConfig
    from rage.execution import SelectionExecution, selection_execution
    from .full_pool import COSINE_NUMERICS, select_full_pool
    from .regions import load_selection, save_selection
    from .selection_metrics import SelectionMetrics
    if len(config["datasets"]) != 1:
        raise ValueError("Target-guided temperature trials use one dataset per run")
    from rage.epsilon_protocol import load_p10_calibration
    calibration = load_p10_calibration(config["epsilon_calibration"], identity)
    c, h = config.get("context_length", 512), config.get("prediction_length", 64)
    crb = CRBConfig(context_length=c, prediction_length=h, mean_epsilon=calibration["epsilon"],
        endpoint_epsilon=calibration["epsilon"], **config["crb"])
    probe.execution = SelectionExecution(**config.get("selection_execution", {}))
    dataset, arm = config["datasets"][0], config["arm"]
    costs = SelectionMetrics(out, device=probe.device, protocol=dict(revision=REVISION,
        purpose="prevalidation_probability_development", arm=arm, temperature=config["temperature"]))
    cache = out / "method_cache"
    cache.mkdir(parents=True, exist_ok=True)
    cache_boundaries = cache / "development_boundaries.json"
    if cache_boundaries.exists() and json.loads(cache_boundaries.read_text()) != boundaries:
        raise ValueError("Shared development score cache uses another validation split")
    atomic_json(cache_boundaries, boundaries)
    path = out / "selection.npz"
    if path.exists():
        output = costs.measure(dataset, arm, lambda: load_selection(path), selection_cached=True)
        if (output.selected.metadata["development_boundaries"] != boundaries or
                output.selected.metadata["pool_manifest"] != manifest or
                output.selected.metadata["model"] != identity):
            raise ValueError("Development probability provenance changed")
        if arm in ("rage", "less", "trajectory_level") and output.selected.metadata.get("cosine_numerics") != COSINE_NUMERICS:
            raise ValueError("Gradient cosine normalization changed; use a new development directory")
    else:
        with selection_execution(probe):
            output = costs.measure(dataset, arm, lambda: select_full_pool(arm, probe,
                reference_fragments=history, pool=pool,
                index=WindowIndex(pool, context_length=c, prediction_length=h), pool_manifest=manifest,
                model_identity=identity, crb_config=crb, temperature=config["temperature"],
                cache_dir=cache, run_dir=out, dataset_tag=dataset + "__prevalidation"))
        output.selected.metadata.update(development_boundaries=boundaries, model=identity)
        save_selection(path, output)
    return output.selected


def development_split(inputs, datasets, *, context=512, horizon=64, count=8):
    """Reference files already exclude exams; reserve whole validation windows."""
    validation, boundaries, history = {}, {}, []
    for dataset in datasets:
        refs = load_fragments(Path(inputs) / f"{dataset}__reference.npz")
        manifest_path = Path(inputs) / f"{dataset}__manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        validation[dataset], bounds = validation_from_history(refs,
            context_length=context, prediction_length=horizon, count=count,
            seasonality=manifest.get("seasonality"))
        boundaries[dataset] = bounds
        for fragment, bound in zip(refs, bounds):
            history.append(TimeSeriesFragment(
                fragment.data[:bound["reference_stop_exclusive"]].copy(),
                entity_id=fragment.entity_id, fragment_id=len(history)))
    return validation, boundaries, history


def run_exploration(config, out, *, model_bundle=None):
    from .checkpoint_retention import cleanup_completed_weights
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = out / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != config:
        raise ValueError("Exploration configuration changed; use a new output directory")
    if config["training_data"] not in ("source_random", "safe_history_random", "source_probability"):
        raise ValueError("Label source-pool and safe-history exploration separately")
    if not config["epochs"] or any(not math.isfinite(epoch) or epoch <= 0 for epoch in config["epochs"]):
        raise ValueError("Use positive finite Epoch observations")
    atomic_json(protocol_path, config)
    c, h = config.get("context_length", 512), config.get("prediction_length", 64)
    validation, boundaries, history = development_split(config["inputs"], config["datasets"],
        context=c, horizon=h, count=config.get("validation_windows", 8))
    atomic_json(out / "validation_boundaries.json", boundaries)
    # Export only the prefix for later target-guided probability calibration.
    save_fragments(out / "prevalidation_history.npz", history)
    if config["training_data"] != "safe_history_random":
        pool = load_fragments(Path(config["inputs"]) / "pool.npz")
        manifest = json.loads((Path(config["inputs"]) / "pool_manifest.json").read_text())
    else:
        pool = history
        manifest = dict(total_equivalents=sum(len(f.data) // (c + h) for f in pool),
            purpose="safe_history_development_only", boundaries=boundaries)
    engine, probe, identity = model_bundle or build_model(config["student"],
        model_path=config.get("model_path"), device=config.get("device", "cuda"),
        context_length=c, prediction_length=h)

    def assess():
        scores = {name: evaluate(engine, batch_size=config.get("eval_batch", 32), **kwargs)
                  for name, kwargs in validation.items()}
        return dict(by_dataset=scores, mean_mase=float(np.mean([s["mase"] for s in scores.values()])))

    original_path = out / "original.json"
    if not original_path.exists():
        atomic_json(original_path, assess())
    original = json.loads(original_path.read_text())
    if config["training_data"] == "source_probability":
        atomic_json(out / "selection_state.json", dict(status="running", arm=config["arm"]))
        try:
            windows = development_probability(config, out, probe, identity, pool, manifest, history, boundaries)
            atomic_json(out / "selection_state.json", dict(status="complete", arm=config["arm"]))
        except Exception as error:
            atomic_json(out / "selection_state.json", dict(status="failed", arm=config["arm"], error=repr(error)))
            raise
    else:
        windows = random_distribution(WindowIndex(pool, context_length=c, prediction_length=h),
                                      pool_manifest=manifest)
    del probe, history
    if identity.get("training_parameter_dtype") == "torch.bfloat16":
        engine.model.to(dtype=torch.bfloat16)
    batch = config.get("batch_size", 32)
    schedule = exposure_checkpoints(windows.pool_n, batch, config["epochs"])
    training_options = config.get("optimizer")
    if training_options is not None:
        training_options = dict(training_options)
        horizon = config.get("schedule_epochs", max(config["epochs"]))
        training_options["schedule_steps"] = math.ceil(horizon * windows.pool_n / batch)
    atomic_json(out / "sampling.json", dict(revision=REVISION, pool_manifest=manifest,
        pool_n=windows.pool_n, legal_windows=len(windows), schedule=schedule,
        training_data=config["training_data"], formal_exam_used=False))
    identity_path = out / "model.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Exploration model changed")
    atomic_json(identity_path, identity)
    if not (out / "provenance.json").exists():
        atomic_json(out / "provenance.json", dict(git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()))
    base = {key: value.detach().cpu().clone() for key, value in engine.model.state_dict().items()}

    results = []
    for lr in config["learning_rates"]:
        for seed in config["seeds"]:
            trial = out / f"lr-{lr:g}-seed-{seed}"
            trial.mkdir(exist_ok=True)
            result_path = trial / "result.json"
            if not result_path.exists():
                atomic_json(trial / "state.json", dict(status="running"))
                cuda = torch.device(engine.device).type == "cuda"
                if cuda:
                    torch.cuda.reset_peak_memory_stats()
                try:
                    result = online_curve(engine, windows, base_state=base, learning_rate=lr,
                        seed=seed, checkpoints=[(s["epoch"], s["step"]) for s in schedule],
                        batch_size=batch, evaluate_fn=assess,
                        weight_decay=config.get("weight_decay", 0.01), training_options=training_options,
                        resume_path=trial / "resume.pt", observations_path=trial / "curve.json",
                        on_record=lambda r: print(config["student"], trial.name,
                            r["epoch"], r["metrics"]["mean_mase"], flush=True))
                    result.update(peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.,
                        training_data=config["training_data"], pool_n=windows.pool_n)
                    for row in result["curve"]:
                        row["relative_improvement_pct"] = 100 * (1 - row["metrics"]["mean_mase"] / original["mean_mase"])
                        row["relative_improvement_by_dataset_pct"] = {
                            name: 100 * (1 - score["mase"] / original["by_dataset"][name]["mase"])
                            for name, score in row["metrics"]["by_dataset"].items()}
                    atomic_json(result_path, result)
                    atomic_json(trial / "state.json", dict(status="complete"))
                except Exception as error:
                    atomic_json(trial / "state.json", dict(status="failed", error=repr(error)))
                    raise
            if config.get("checkpoint_retention") == "completed_cell_cleanup":
                cleanup_completed_weights(result_path, trial)
            results.append(json.loads(result_path.read_text()))
    report = dict(original=original, trials=results, formal_exam_used=False,
        parameter_freeze="not automatic; inspect per-task curves and independent-seed replication")
    atomic_json(out / "results.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON development trial configuration")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    config = json.loads(Path(args.config).read_text())
    if config["student"] not in STUDENTS:
        parser.error("Use one of the three current ~200M students")
    run_exploration(config, args.out)


if __name__ == "__main__":
    main()
