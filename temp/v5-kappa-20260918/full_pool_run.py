"""Prepare and orchestrate the document-frozen full-pool matrix."""
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .development_inputs import save_fragments
from .exact_pool import build_exact_pool
from .full_pool import ARMS, TEMPERATURE_ARMS, GATELESS_ARMS, TEMPERATURE_DEFAULT_FORM
from .resampling import REVISION
from .students import STUDENTS, ENVIRONMENTS, atomic_json


def validate_config(config):
    if config.get("experiment_version") == "v4":
        from .v4_protocol import validate_v4_config
        return validate_v4_config(config)
    if config.get("experiment_version") == "v3":
        from .v3_protocol import validate_v3_config
        return validate_v3_config(config)
    from .tasks import MAIN_DATASETS
    if config["resampling_revision"] != REVISION:
        raise ValueError("Unsupported full-pool protocol")
    if config.get("purpose") in ("formal", "rags_random_comparison"):
        required = dict(pool_equivalents=16284, context_length=512, prediction_length=64,
                        students=list(STUDENTS), datasets=list(MAIN_DATASETS),
                        arms=list(ARMS), seeds=[0, 1, 2])
        if config.get('task_protocol') == 'shared-ten-formal-20260915':
            decisions = json.loads(Path(config['shared_task_decisions']).read_text())
            required['datasets'] = decisions['datasets']
            required['seeds'] = [0]
            # The linked temperature supplement run carries only the T=0.25 arms;
            # the main run keeps the frozen six training arms. The linked
            # RAGS-Gateless ablation run carries only rags_gateless. The
            # temperature-default form swaps rags/less for their T=0.25 arms.
            if config.get('arms') == list(TEMPERATURE_DEFAULT_FORM):
                required['arms'] = list(TEMPERATURE_DEFAULT_FORM)
            elif config.get('arms') == list(TEMPERATURE_ARMS):
                required['arms'] = list(TEMPERATURE_ARMS)
            elif config.get('arms') == list(GATELESS_ARMS):
                required["arms"] = list(GATELESS_ARMS)
        if config["purpose"] == "rags_random_comparison":
            required.update(arms=["random", "rags"], online_evaluation=True)
        for key, value in required.items():
            if config.get(key) != value:
                raise ValueError(f"Formal matrix requires {key}={value!r}")
    if not config.get("settings"):
        raise ValueError("Model-specific settings must come from completed development exploration")
    if config.get('task_protocol') == 'shared-ten-formal-20260915':
        for key,value in dict(cover_density=.02,error_density=.01,split_density=.01).items():
            if config['crb'].get(key)!=value:
                raise ValueError(f'Frozen shared-ten CRB budget requires {key}={value}')
        decisions = json.loads(Path(config['shared_task_decisions']).read_text())
        for student in config['students']:
            recipe = decisions['models'][student]
            expected = dict(learning_rate=recipe['learning_rates'][0],epochs=recipe['epochs'][0],
                batch_size=recipe['batch_size'],weight_decay=recipe['weight_decay'],
                optimizer=recipe['optimizer'],schedule_epochs=recipe['schedule_epochs'])
            for key,value in expected.items():
                if config['settings'][student].get(key)!=value:
                    raise ValueError(f'Frozen shared-ten recipe mismatch: {student}.{key}')
    for student in config["students"]:
        settings = config["settings"][student]
        keys = ["epochs", "learning_rate", "temperature", "development_evidence"]
        if config["crb"].get("epsilon_policy") not in ("trajectory_p10", "none"):
            keys.append("epsilon_calibration")
        for key in keys:
            if not settings.get(key):
                raise ValueError(f"Missing frozen development setting: {student}.{key}")
        if isinstance(settings["temperature"], dict):
            for arm in set(config["arms"]) - {"random", "tsds"}:
                if not settings["temperature"].get(arm):
                    raise ValueError(f"Missing frozen development temperature: {student}.{arm}")


def prepare_inputs(config, run_dir):
    from .tasks import build_task_specs, build_task_data
    folder = Path(config.get("inputs") or Path(run_dir) / "inputs")
    folder.mkdir(parents=True, exist_ok=True)
    pool_path = folder / "pool.npz"
    manifest_path = folder / "pool_manifest.json"
    if not pool_path.exists() or not manifest_path.exists():
        pool, manifest = build_exact_pool(load_dir=Path(config["source_load"]),
            equivalents=config["pool_equivalents"], seed=config.get("pool_seed", 0),
            window_length=config["context_length"] + config["prediction_length"])
        save_fragments(pool_path, pool)
        atomic_json(manifest_path, manifest)
    manifest = json.loads(manifest_path.read_text())
    if manifest["total_equivalents"] != config["pool_equivalents"]:
        raise ValueError("Prepared pool N does not match the requested exact equivalent count")
    if config.get('task_protocol') == 'shared-ten-formal-20260915':
        from .shared_formal_inputs import prepare_shared_formal_inputs
        prepare_shared_formal_inputs(json.loads(Path(config['shared_task_decisions']).read_text()),
                                     folder, evaluation_mode='rolling')
        return str(folder.resolve())
    specs = build_task_specs(keywords=tuple(config["datasets"]), reference_root=Path(config["reference_root"]))
    for spec in specs:
        done = folder / f"{spec.dataset}__manifest.json"
        if done.exists():
            continue
        task = build_task_data(spec, context_length=config["context_length"],
            prediction_length=config["prediction_length"], pool_fragments=[])
        save_fragments(folder / f"{spec.dataset}__reference.npz", task.reference_fragments)
        np.savez_compressed(folder / f"{spec.dataset}__exam.npz", windows=task.test_windows,
                           entities=np.asarray(task.test_entities), scales=task.test_scales)
        atomic_json(done, dict(dataset=spec.dataset, series=len(task.reference_fragments),
            exam_windows=len(task.test_windows), full_reference_history=True,
            reference_lengths=[f.length for f in task.reference_fragments],
            source_training_only=True))
    return str(folder.resolve())


def worker_command(student, config, run_dir):
    environment = config.get("student_environments", {}).get(student, ENVIRONMENTS[student])
    prefix = ([environment] if "/" in environment else [sys.executable] if student == "chronos-base"
              and environment == "ts3" else ["conda", "run", "--no-capture-output", "-n", environment, "python"])
    return prefix + ["-m", "rags.experiments.full_pool_worker", "--run", str(run_dir), "--student", student]


def summarize(run_dir, config):
    run_dir = Path(run_dir)
    records, missing = [], []
    original_arm = "orig" if config.get("experiment_version") in ("v3", "v4") else "original"
    random_arm = "rand" if config.get("experiment_version") in ("v3", "v4") else "random"
    training_arms = [arm for arm in config["arms"] if arm != "orig"]
    for student in config["students"]:
        for dataset in config["datasets"]:
            folder = run_dir / "students" / student / "cells"
            original = folder / f"{dataset}__{original_arm}__seed0.json"
            if not original.exists():
                missing.append(str(original))
            else:
                records.append(json.loads(original.read_text()))
            for arm in training_arms:
                for seed in config["seeds"]:
                    path = folder / f"{dataset}__{arm}__seed{seed}.json"
                    if path.exists():
                        records.append(json.loads(path.read_text()))
                    else:
                        missing.append(str(path))
    comparisons = []
    for student in config["students"]:
        for dataset in config["datasets"]:
            group = [r for r in records if r["student"] == student and r["dataset"] == dataset]
            originals = [r["mase"] for r in group if r["arm"] == original_arm]
            random = [r["mase"] for r in group if r["arm"] == random_arm]
            for arm in training_arms:
                values = [r["mase"] for r in group if r["arm"] == arm]
                if values:
                    mean = float(np.mean(values))
                    comparisons.append(dict(student=student, dataset=dataset, arm=arm,
                        seeds=len(values), mean_mase=mean,
                        std_mase=float(np.std(values, ddof=1)) if len(values)>1 else None,
                        improvement_vs_original=(1 - mean / originals[0]) if originals and originals[0] else None,
                        improvement_vs_random=(1 - mean / float(np.mean(random))) if random and np.mean(random) else None))
    paired = []
    for student in config['students']:
        for dataset in config['datasets']:
            group = [r for r in records if r['student']==student and r['dataset']==dataset]
            baseline = {r['seed']:r['mase'] for r in group if r['arm']==random_arm}
            for arm in training_arms:
                if arm==random_arm:
                    continue
                differences = [{'seed':r['seed'],'delta_mase':float(r['mase']-baseline[r['seed']])}
                    for r in group if r['arm']==arm and r['seed'] in baseline]
                if differences:
                    values = [r['delta_mase'] for r in differences]
                    paired.append(dict(student=student,dataset=dataset,arm=arm,baseline=random_arm,
                        pairs=differences,mean_delta_mase=float(np.mean(values)),
                        std_delta_mase=float(np.std(values,ddof=1)) if len(values)>1 else None))
    summary = dict(paired_differences=paired, complete=not missing, completed_cells=len(records), missing=missing, cells=records,
                   comparisons=comparisons)
    atomic_json(run_dir / "results.json", summary)
    return summary


def run_suite(config, run_dir, *, dry_run=False, execution=None):
    run_dir = Path(run_dir).resolve()
    commands = [worker_command(student, config, run_dir) for student in config["students"]]
    if dry_run:
        for command in commands:
            print(" ".join(command))
        return 0
    validate_config(config)
    from .full_pool_worker import _assert_training_enabled
    _assert_training_enabled(run_dir, experiment_version=config.get("experiment_version"))
    original_arm = "orig" if config.get("experiment_version") in ("v3", "v4") else "original"
    if config.get("purpose") == "formal" and (config.get("experiment_version") not in ("v2", "v3", "v4")
            or config["crb"].get("split_policy") != "closest_zero_v2"):
        raise ValueError("Formal runs require the supported closest-zero CRB protocol; preserve v1 runs")
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol = run_dir / "protocol.json"
    frozen = dict(config, inputs=str(Path(config.get("inputs") or run_dir / "inputs").resolve()))
    if not protocol.exists():
        frozen.setdefault("checkpoint_retention", "completed_cell_cleanup" if config.get("purpose") == "formal" else "keep")
    else:
        previous = json.loads(protocol.read_text())
        if "checkpoint_retention" in previous:
            frozen.setdefault("checkpoint_retention", previous["checkpoint_retention"])
    if protocol.exists() and json.loads(protocol.read_text()) != frozen:
        raise ValueError("Frozen run config changed; create a new run directory")
    atomic_json(protocol, frozen)
    execution_path = run_dir / "execution.json"
    if execution is not None:
        if set(execution) - {"student_parallelism", "student_selection_execution", "original_first"}:
            raise ValueError("Execution overrides cannot change scientific/training settings")
        atomic_json(execution_path, execution)
    elif execution_path.exists():
        execution = json.loads(execution_path.read_text())
    execution = execution or {}
    provenance = run_dir / "provenance.json"
    if not provenance.exists():
        atomic_json(provenance, dict(git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(), protocol=REVISION))
    prepare_inputs(frozen, run_dir)
    if execution.get("original_first", config.get("original_first")):
        original_jobs = []
        for student, command in zip(config["students"], commands):
            cells = run_dir / "students" / student / "cells"
            if not all((cells / f"{dataset}__{original_arm}__seed0.json").exists()
                       for dataset in config["datasets"]):
                original_jobs.append((student, command))
        original_failures = []
        def execute_original(student, command):
            with (run_dir / f"{student}.log").open("a") as log:
                result = subprocess.run(command + ["--original-only"],
                                        stdout=log, stderr=subprocess.STDOUT)
            return student, result.returncode
        with ThreadPoolExecutor(max_workers=execution.get(
                "student_parallelism", config.get("student_parallelism", len(config["students"])))) as executor:
            futures = [executor.submit(execute_original, student, command)
                       for student, command in original_jobs]
            for future in as_completed(futures):
                student, code = future.result()
                summarize(run_dir, frozen)
                if code:
                    original_failures.append(student)
        if original_failures:
            raise RuntimeError("Original evaluation failed: " + ", ".join(original_failures))
    def execute_student(student, command):
        with (run_dir / f"{student}.log").open("a") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        return student, result.returncode

    failures = []
    with ThreadPoolExecutor(max_workers=execution.get("student_parallelism", config.get("student_parallelism", len(config["students"])))) as executor:
        futures = [executor.submit(execute_student, student, command)
                   for student, command in zip(config["students"], commands)]
        for future in as_completed(futures):
            student, code = future.result()
            summary = summarize(run_dir, frozen)
            if code:
                failures.append(f"{student} ({code})")
    if failures:
        raise RuntimeError(f"Failed students: {', '.join(failures)}; other students were allowed to finish")
    if not summary["complete"]:
        raise RuntimeError("Worker processes exited but the requested matrix is incomplete; inspect results.json")
    return 0
