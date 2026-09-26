"""Model-specific hyperparameters followed by one shared task selection.

Only development reference prefixes are consumed by the runtime. Each trial
starts at the original model, including midpoints earlier than completed runs.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys

from .probe_recipes import MODELS, recipe

SEARCH_POLICY = "extend_or_bisect_v2"


def search_budget(config):
    # The old field is accepted for configuration migration, with the new meaning.
    return config.get("max_search_steps", config.get("max_bisections"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_config(config):
    required = {"inputs", "parameter_tasks", "candidate_tasks", "models", "seeds",
                "coarse_epochs", "task_probe_model", "top_k"}
    allowed = required | {"max_search_steps", "max_bisections", "context_length", "prediction_length", "validation_windows", "eval_batch", "device"}
    if not required <= set(config) or set(config) - allowed:
        raise ValueError("Invalid probe fields; required=%s" % sorted(required))
    if set(config["models"]) != set(MODELS):
        raise ValueError("All three models must receive independent hyperparameter probes")
    for key in ("context_length", "prediction_length", "validation_windows", "eval_batch"):
        if key in config and (type(config[key]) is not int or config[key] <= 0):
            raise ValueError("%s must be a positive integer" % key)
    for field in ("parameter_tasks", "candidate_tasks"):
        tasks = config[field]
        if not isinstance(tasks, list) or not tasks or len(set(tasks)) != len(tasks):
            raise ValueError("%s must contain unique task IDs" % field)
        if any(not isinstance(t, str) or not t or Path(t).name != t or t in {".", ".."} for t in tasks):
            raise ValueError("Task IDs must be safe file stems (use aliases for dataset/frequency)")
    if not set(config["parameter_tasks"]) <= set(config["candidate_tasks"]):
        raise ValueError("Parameter tasks must belong to the common candidate set")
    if len(config["candidate_tasks"]) > 20:
        raise ValueError("At most 20 candidate tasks in the shared search pool")
    seeds = config["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("seeds must be distinct non-negative integers")
    epochs = config["coarse_epochs"]
    if len(epochs) < 2 or epochs != sorted(set(epochs)) or any(not math.isfinite(e) or e <= 0 for e in epochs):
        raise ValueError("coarse_epochs must be increasing positive observations, e.g. [1, 2]")
    if sum(key in config for key in ("max_search_steps", "max_bisections")) != 1:
        raise ValueError("Specify exactly one search budget: max_search_steps (legacy: max_bisections)")
    if type(search_budget(config)) is not int or not 0 <= search_budget(config) <= 4:
        raise ValueError("At most four additional searches per model, including extensions and bisections")
    if type(config["top_k"]) is not int or not 1 <= config["top_k"] <= len(config["candidate_tasks"]):
        raise ValueError("top_k must fit the common candidate set")
    if config["task_probe_model"] not in (*MODELS, "best_relative_gain"):
        raise ValueError("Choose one task probe model or best_relative_gain")
    for model, settings in config["models"].items():
        if set(settings) - {"python", "model_path", "learning_rates"}:
            raise ValueError("Per-model settings allow only python, model_path, learning_rates; tasks are shared")
        rates = settings["learning_rates"]
        if not rates or len(set(rates)) != len(rates) or any(not math.isfinite(lr) or lr <= 0 for lr in rates):
            raise ValueError("learning_rates must be unique positive values")
        if rates[0] != recipe(model)["learning_rate"]:
            raise ValueError("The first learning rate must be the official reference for " + model)
    return config


def ensure_protocol(config, out):
    validate_config(config)
    out = Path(out)
    path = out / "probe_protocol.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Probe configuration changed; use a new output directory")
    policy_path = out / "epoch_search_policy.json"
    policy = dict(revision=SEARCH_POLICY, budget=search_budget(config),
                  schedule_epochs=max(config["coarse_epochs"]) + search_budget(config))
    if path.exists() and (not policy_path.exists() or json.loads(policy_path.read_text()) != policy):
        raise ValueError("Epoch search policy changed; use a new output directory")
    if not path.exists():
        write_json(policy_path, policy)
        write_json(path, config)
    references = {model: recipe(model) for model in MODELS}
    reference_path = out / "official_references.json"
    if reference_path.exists() and json.loads(reference_path.read_text()) != references:
        raise ValueError("Official reference settings changed; use a new output directory")
    if not reference_path.exists():
        write_json(reference_path, references)


def relative_gain(base, trained):
    if base is None or trained is None or not math.isfinite(base) or not math.isfinite(trained) or base <= 0 or trained < 0:
        raise ValueError("Relative MASE gain requires finite positive baseline and non-negative trained score")
    return (base - trained) / base


def summarize(report, tasks, seeds):
    """Mean of paired per-task relative gains, then equally across seeds."""
    baseline = report["original"]["by_dataset"]
    if set(baseline) != set(tasks) or report.get("formal_exam_used") is not False:
        raise ValueError("Probe report has wrong tasks or uses formal exam")
    by_epoch = {}
    if sorted(t["seed"] for t in report["trials"]) != sorted(seeds):
        raise ValueError("Every configured seed must complete exactly once")
    for trial in report["trials"]:
        seen = set()
        for point in trial["curve"]:
            epoch = float(point["epoch"])
            scores = point["metrics"]["by_dataset"]
            if epoch in seen or set(scores) != set(tasks):
                raise ValueError("Duplicate epoch or missing task score")
            seen.add(epoch)
            gains = {task: relative_gain(baseline[task]["mase"], scores[task]["mase"]) for task in tasks}
            by_epoch.setdefault(epoch, []).append(dict(seed=trial["seed"], gains=gains,
                relative_gain=sum(gains.values()) / len(gains), step=point["step"],
                actual_epoch=point.get("actual_epoch", epoch)))
    if not by_epoch or any(len(rows) != len(seeds) for rows in by_epoch.values()):
        raise ValueError("Incomplete epoch observations across seeds")
    return {epoch: dict(epoch=epoch, relative_gain=sum(r["relative_gain"] for r in rows) / len(rows),
                        by_seed=rows) for epoch, rows in by_epoch.items()}


def trial_config(config, model, tasks, lr, epochs):
    reference = recipe(model)
    return dict(student=model, model_path=config["models"][model].get("model_path"),
        inputs=config["inputs"], datasets=tasks, training_data="safe_history_random",
        learning_rates=[lr], epochs=epochs, seeds=config["seeds"],
        batch_size=reference["batch_size"], weight_decay=reference["weight_decay"],
        optimizer=reference["optimizer"], schedule_epochs=max(config["coarse_epochs"]) + search_budget(config),
        context_length=config.get("context_length", 512), prediction_length=config.get("prediction_length", 64),
        validation_windows=config.get("validation_windows", 8), eval_batch=config.get("eval_batch", 32),
        device=config.get("device", "cuda"))


def execute_trial(config, out):
    from .full_pool_exploration import run_exploration
    return run_exploration(config, out)


def parameter_probe(config, model, out, *, run_trial=execute_trial):
    ensure_protocol(config, out)
    if model not in MODELS:
        raise ValueError("Unknown probe model")
    destination = Path(out) / "parameters" / model
    all_points, trace = [], []
    for index, lr in enumerate(config["models"][model]["learning_rates"]):
        trial = trial_config(config, model, config["parameter_tasks"], lr, config["coarse_epochs"])
        path = destination / ("coarse-%02d" % index)
        report = run_trial(trial, path)
        points = summarize(report, trial["datasets"], config["seeds"])
        if set(points) != set(config["coarse_epochs"]):
            raise ValueError("Coarse trial did not evaluate all requested epochs")
        all_points.extend(dict(value, learning_rate=lr, evidence=str(path)) for value in points.values())
    # Deterministic ties: prefer smaller duration, then earlier LR trial.
    best = max(all_points, key=lambda p: (p["relative_gain"], -p["epoch"]))
    lr = best["learning_rate"]
    observations = {p["epoch"]: p for p in all_points if p["learning_rate"] == lr}
    observations[0.0] = dict(epoch=0.0, relative_gain=0.0)
    previous, frontier = map(float, config["coarse_epochs"][-2:])
    extending = observations[frontier]["relative_gain"] > observations[previous]["relative_gain"]
    pivot = max(observations, key=lambda e: (observations[e]["relative_gain"], -e))
    ordered = sorted(observations)
    index = ordered.index(pivot)
    lo, hi = (pivot, ordered[index + 1]) if index + 1 < len(ordered) else (ordered[index - 1], pivot)
    for iteration in range(search_budget(config)):
        action = "extend" if extending else "bisect"
        candidate = frontier + 1 if extending else (lo + hi) / 2
        # If exposure rounding already maps this point to an endpoint step,
        # further subdivision cannot produce a new trained checkpoint.
        probe_side = observations[frontier] if extending else observations[lo] if lo > 0 else observations[hi]
        n_per_batch = probe_side["by_seed"][0]["step"] / probe_side["by_seed"][0]["actual_epoch"]
        if math.ceil(candidate * n_per_batch) in {p["by_seed"][0]["step"] for e, p in observations.items() if e > 0}:
            trace.append(dict(iteration=iteration + 1, action=action, bracket=[lo, hi],
                              stop="optimizer_step_resolution", budget_consumed=False))
            break
        path = destination / ("search-%02d-%s" % (iteration + 1, action))
        trial = trial_config(config, model, config["parameter_tasks"], lr, [candidate])
        points = summarize(run_trial(trial, path), trial["datasets"], config["seeds"])
        if set(points) != {candidate}:
            raise ValueError("Epoch search did not evaluate requested point")
        point = dict(points[candidate], learning_rate=lr, evidence=str(path))
        all_points.append(point)
        observations[candidate] = point
        trace.append(dict(iteration=iteration + 1, action=action, epoch=candidate,
                          bracket=[frontier, candidate] if extending else [lo, hi],
                          relative_gain=point["relative_gain"], budget_consumed=True))
        if extending:
            lo, hi = frontier, candidate
            extending = point["relative_gain"] > observations[frontier]["relative_gain"]
            frontier = candidate
            continue
        if observations[lo]["relative_gain"] >= observations[hi]["relative_gain"]:
            if point["relative_gain"] > observations[lo]["relative_gain"]:
                lo = candidate
            else:
                hi = candidate
        elif point["relative_gain"] > observations[hi]["relative_gain"]:
            hi = candidate
        else:
            lo = candidate
    best = max(all_points, key=lambda p: (p["relative_gain"], -p["epoch"]))
    selected = trial_config(config, model, config["parameter_tasks"], best["learning_rate"], [best["epoch"]])
    report = dict(model=model, official_reference=recipe(model), points=all_points,
                  search_policy=SEARCH_POLICY, search_trace=trace, search_budget=search_budget(config),
                  search_steps_used=sum(p["budget_consumed"] for p in trace),
                  selected=selected, best_relative_gain=best["relative_gain"],
                  improves_baseline=best["relative_gain"] > 0, formal_exam_used=False,
                  selection_scope="development_only",
                  epoch_search_upper_bound=max(config["coarse_epochs"]) + search_budget(config),
                  largest_evaluated_epoch=max(p["epoch"] for p in all_points))
    write_json(destination / "result.json", report)
    return report


def parameter_results(config, out):
    ensure_protocol(config, out)
    result = {}
    for model in MODELS:
        report = json.loads((Path(out) / "parameters" / model / "result.json").read_text())
        if report["model"] != model or report["selected"]["datasets"] != config["parameter_tasks"] or report["formal_exam_used"]:
            raise ValueError("Parameter report identity/tasks mismatch")
        result[model] = report
    return result


def choose_task_model(config, results):
    model = config["task_probe_model"]
    if model == "best_relative_gain":
        model = max(MODELS, key=lambda key: results[key]["best_relative_gain"])
    if results[model]["best_relative_gain"] <= 0:
        raise ValueError("Task probe requires a parameter setting with positive development gain; revise LR/epoch search")
    return model


def task_probe(config, out, *, run_trial=execute_trial):
    results = parameter_results(config, out)
    model = choose_task_model(config, results)
    frozen = deepcopy(results[model]["selected"])
    ranking = []
    for task in config["candidate_tasks"]:
        trial = dict(frozen, datasets=[task])
        path = Path(out) / "task_probe" / task
        points = summarize(run_trial(trial, path), [task], config["seeds"])
        epoch = frozen["epochs"][0]
        if set(points) != {epoch}:
            raise ValueError("Task probe must hold epochs fixed")
        ranking.append(dict(task=task, **points[epoch], evidence=str(path)))
    ranking.sort(key=lambda row: (-row["relative_gain"], row["task"]))
    selected = [row["task"] for row in ranking if row["relative_gain"] > 0][:config["top_k"]]
    report = dict(probe_model=model, fixed_training=frozen, ranking=ranking,
                  shared_tasks=selected, shared_by=list(MODELS), requested_top_k=config["top_k"],
                  formal_exam_used=False, metric="mean paired relative MASE gain",
                  relative_gain_formula="(untrained - trained) / untrained")
    write_json(Path(out) / "task_probe" / "result.json", report)
    if not selected:
        raise ValueError("No task improved; ranking was saved but no shared task set was frozen")
    # One global task list; deliberately no model-specific task field.
    settings = {}
    for key in MODELS:
        item = deepcopy(results[key]["selected"])
        for field in ("datasets", "inputs", "student", "model_path", "validation_windows", "eval_batch", "device"):
            item.pop(field, None)
        settings[key] = item
    frozen_artifact = dict(schema_version=1, datasets=selected, models=settings,
                           task_selection_evidence=str(Path(out) / "task_probe" / "result.json"),
                           stage="development_selection", formal_exam_used=False)
    write_json(Path(out) / "shared_selection.json", frozen_artifact)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=["plan", "parameters", "tasks", "all"], default="plan")
    parser.add_argument("--student", choices=MODELS)
    args = parser.parse_args(argv)
    config = validate_config(json.loads(args.config.read_text()))
    if args.stage == "plan":
        print(json.dumps(dict(models={m: recipe(m) for m in MODELS}, config=config,
              workflow="three independent parameter probes -> one fixed-parameter task probe -> one shared task set"), indent=2))
    elif args.stage == "parameters":
        if not args.student:
            parser.error("--student is required for parameters")
        parameter_probe(config, args.student, args.out)
    elif args.stage == "tasks":
        task_probe(config, args.out)
    else:
        ensure_protocol(config, args.out)
        for model in MODELS:
            subprocess.run([config["models"][model].get("python", sys.executable), "-m",
                "rags.experiments.probing", "--config", str(args.config.resolve()),
                "--out", str(args.out.resolve()), "--stage", "parameters", "--student", model], check=True)
        model = choose_task_model(config, parameter_results(config, args.out))
        subprocess.run([config["models"][model].get("python", sys.executable), "-m",
            "rags.experiments.probing", "--config", str(args.config.resolve()),
            "--out", str(args.out.resolve()), "--stage", "tasks"], check=True)


if __name__ == "__main__":
    main()
