"""Shared-selection deployment contracts; no optional model imports here."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

STUDENTS = ("chronos-base", "timesfm-2.5", "timemoe-200m", "timerxl-84m", "sundial-base-128m")
# Kept in the historical registry for offline compatibility, but these models
# have no supported student-training deployment contract.
NON_DEPLOYABLE_STUDENTS = frozenset(("sundial-base-128m", "timesfm-3.0"))
HISTORICAL_STUDENTS = ("chronos-small",)
ENVIRONMENTS = dict(zip(STUDENTS, ("ts3", "ts3-timesfm", "ts3-timemoe", "ts3-timerxl", "ts3-sundial")))
PARAMETER_COUNTS = dict(zip(STUDENTS, (201_374_976, 231_289_280, 453_196_800, 84_142_080, 128_329_680)))
ENVIRONMENTS["chronos-small"] = "ts3"
PARAMETER_COUNTS["chronos-small"] = 46_154_240


def provenance(*, include_lock=False):
    from configs.paths import PROJECT_ROOT
    from importlib.metadata import version, PackageNotFoundError
    packages = {}
    for name in ("torch", "transformers", "numpy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not-installed"
    result = {"environment": Path(sys.prefix).name,
              "python": sys.executable, "packages": packages}
    if include_lock:
        result["model_integrations_lock_sha256"] = hashlib.sha256(
            (PROJECT_ROOT / "requ/model_integrations.lock").read_bytes()).hexdigest()
    return result


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def key_values(items):
    result = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or key not in STUDENTS + HISTORICAL_STUDENTS or not value or key in result:
            raise ValueError("expected a unique student=value: " + item)
        result[key] = value
    return result


def worker_command(student, manifest, out_dir, *, environments=None):
    if student not in STUDENTS + HISTORICAL_STUDENTS or student in NON_DEPLOYABLE_STUDENTS:
        raise ValueError("model is excluded from experiments: " + student)
    environment = (environments or {}).get(student, ENVIRONMENTS[student])
    if "/" in environment:
        command = [environment]
    elif student in ("chronos-base", "chronos-small") and environment == "ts3":
        command = [sys.executable]
    else:
        command = ["conda", "run", "--no-capture-output", "-n", environment, "python"]
    return command + ["-m", "scripts.exp.model_train", "--model", student,
                       "--task-manifest", str(manifest), "--output", str(out_dir)]


def deploy_students(students, manifest, out_dir, *, environments=None, parallelism=1):
    """One task barrier: bounded concurrent model workers, then next task.

    Workers never run selection; each consumes the same shared artifacts.
    On failure stop and reap sibling processes (including conda descendants).
    """
    import signal
    import time
    if parallelism < 1:
        raise ValueError("student parallelism must be positive")
    commands = [(s, worker_command(s, manifest, out_dir / "students" / s,
                                  environments=environments)) for s in students]
    active = []
    try:
        while commands or active:
            while commands and len(active) < parallelism:
                student, command = commands.pop(0)
                print("student worker:", command, flush=True)
                active.append((student, subprocess.Popen(command, start_new_session=True)))
            for student, process in list(active):
                code = process.poll()
                if code is not None:
                    active.remove((student, process))
                    if code:
                        raise RuntimeError(f"student {student} failed with exit code {code}")
            if active:
                time.sleep(.1)
    finally:
        for _, process in active:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for _, process in active:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def adapter_engine(adapter, context_length, prediction_length):
    from types import SimpleNamespace
    model = adapter.load()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return SimpleNamespace(
        model=model, device=str(next(model.parameters()).device),
        config=SimpleNamespace(context_length=context_length, prediction_length=prediction_length),
        training_loss=lambda windows: adapter.training_loss(windows, context_length=context_length,
                                                           prediction_length=prediction_length),
        forecast=lambda contexts: adapter.forecast(contexts, prediction_length=prediction_length),
        context_embeddings=getattr(adapter, "context_embeddings", None),
        forecast_protocol={"timesfm-2.5": "native_mean",
                           "timemoe-200m": "native_autoregressive",
                           "timerxl-84m": "native_autoregressive_revin",
                           "sundial-base-128m": "native_sampling_median"}[adapter.spec.key],
    )


def run_student_task(student, manifest_path, out_dir, *, engine=None):
    """Isolated runtime worker; injectable tiny engine supports offline tests."""
    import numpy as np
    from .train_eval import evaluate, run_arm_training
    from .deploy import selection_coverage
    if student not in STUDENTS + HISTORICAL_STUDENTS:
        raise ValueError("excluded experiment student: " + student)
    manifest_path, out_dir = Path(manifest_path), Path(out_dir)
    manifest = json.loads(manifest_path.read_text())
    config = manifest["training"]
    c, h = manifest["context_length"], manifest["prediction_length"]
    if engine is None:
        model_path = manifest.get("model_paths", {}).get(student)
        if student in ("chronos-base", "chronos-small"):
            from rags.engines import ChronosT5Config, ChronosT5Engine, DEFAULT_CHRONOS_T5_SMALL, DEFAULT_CHRONOS_T5_BASE
            default = DEFAULT_CHRONOS_T5_BASE if student == "chronos-base" else DEFAULT_CHRONOS_T5_SMALL
            engine = ChronosT5Engine(ChronosT5Config(context_length=c, prediction_length=h,
                model_path=model_path or default), device=manifest.get("device", "cuda"))
        else:
            from adapters.foundation import build_adapter
            from dataclasses import replace
            adapter = build_adapter(student, device=manifest.get("device", "cuda"))
            if model_path:
                adapter.spec = replace(adapter.spec, model_id=model_path)
            engine = adapter_engine(adapter, c, h)
    runtime = provenance()
    dataset = manifest["dataset"]
    out_dir.mkdir(parents=True, exist_ok=True)
    run_doc = out_dir / "RUN.md"
    if not run_doc.exists():
        run_doc.write_text("# Student runtime\n\n```json\n" + json.dumps(
            {"student": student, **runtime, "training": config}, indent=2) + "\n```\n")
    exam = np.load(manifest["exam"]["path"], allow_pickle=False)
    exam_kwargs = dict(test_windows=exam["windows"], test_entities=exam["entities"].tolist(),
                       context_length=c, test_scales=exam["scales"])
    base = {key: value.detach().cpu().clone() for key, value in engine.model.state_dict().items()}
    (out_dir / "cells").mkdir(parents=True, exist_ok=True)
    original = out_dir / "cells" / f"{dataset}__original__seed0.json"
    if not original.exists():
        import time
        started = time.monotonic()
        metrics = evaluate(engine, **exam_kwargs)
        metrics.update(student=student, dataset=dataset, arm="original", seed=0,
                       runtime=runtime,
                       forecast_protocol=getattr(engine, "forecast_protocol", "autoregressive_greedy_prefix_bin_cdf"),
                       evaluation_elapsed_s=time.monotonic() - started)
        atomic_json(original, metrics)
    for arm, artifact in manifest["selections"].items():
        with np.load(artifact["path"], allow_pickle=False) as selected:
            if "regions" in selected or "resampling_metadata" in selected:
                from .regions import load_selection
                windows = load_selection(artifact["path"]).selected
                coverage = windows.coverage()
            else:
                windows, ids = selected["windows"], selected["ids"]
                coverage = selection_coverage([tuple(row) for row in ids.tolist()], c + h)
        from .resampling import ResampledWindows
        if isinstance(windows, ResampledWindows) and config.get("budget_windows", windows.pool_n) != windows.pool_n:
            raise ValueError("Training N must equal the shared full-pool manifest N")
        for seed in config["seeds"]:
            cell = out_dir / "cells" / f"{dataset}__{arm}__seed{seed}.json"
            if cell.exists():
                continue
            print(f"[{student} / {dataset} / {arm} / seed {seed}] training", flush=True)
            metrics = run_arm_training(engine, source_windows=windows,
                budget_windows=config.get("budget_windows"),
                source_epochs=config["epochs"], batch_size=config["batch_size"],
                learning_rate=config["learning_rate"], seed=seed, base_state=base,
                checkpoint_dir=out_dir / "checkpoints" / dataset / arm / f"seed{seed}", **exam_kwargs)
            metrics.update(student=student, dataset=dataset, arm=arm, runtime=runtime,
                           selection_artifact=artifact["path"],
                           selection_cost_scope="shared_once_in_task_manifest",
                           selection_coverage=coverage)
            atomic_json(cell, metrics)
            print(f"[{student} / {dataset} / {arm} / seed {seed}] MASE={metrics['mase']}", flush=True)
    from .checkpoint_report import write_checkpoint_report
    write_checkpoint_report(out_dir)
    return out_dir
