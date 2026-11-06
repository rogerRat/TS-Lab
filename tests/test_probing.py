from copy import deepcopy
import json
import math

import pytest

from rage.experiments.probing import (MODELS, choose_task_model, parameter_probe,
    parameter_results, relative_gain, summarize, task_probe, validate_config)
from rage.experiments.probe_recipes import recipe


def config():
    return dict(inputs="unused", parameter_tasks=["a"], candidate_tasks=["a", "b", "c"],
        models={model: {"learning_rates": [recipe(model)["learning_rate"], recipe(model)["learning_rate"] / 10]}
                for model in MODELS}, seeds=[0, 1], coarse_epochs=[1, 2],
        max_search_steps=4, task_probe_model="timemoe-200m", top_k=2)


def runner(calls):
    def execute(trial, out):
        calls.append(deepcopy(trial))
        model = trial["student"]
        reference_lr = recipe(model)["learning_rate"]
        optimal = reference_lr if model == "chronos-base" else reference_lr / 10
        rate_factor = 1.0 if trial["learning_rates"][0] == optimal else .2
        family_factor = {"chronos-base": .8, "timesfm-2.5": 1., "timemoe-200m": .9}[model]
        baseline = {task: {"mase": {"a": 10., "b": 1., "c": 100.}[task]} for task in trial["datasets"]}
        trials = []
        for seed in trial["seeds"]:
            curve = []
            for epoch in trial["epochs"]:
                gain = (.35 - .12 * (epoch - 1.25) ** 2) * rate_factor * family_factor
                scores = {task: {"mase": base["mase"] * (1 - gain * {"a": 1, "b": 1.4, "c": -.5}[task])}
                          for task, base in baseline.items()}
                step = math.ceil(epoch * 100)
                curve.append(dict(epoch=epoch, step=step, actual_epoch=step / 100,
                                  metrics=dict(by_dataset=scores)))
            trials.append(dict(seed=seed, curve=curve))
        return dict(original=dict(by_dataset=baseline), trials=trials, formal_exam_used=False)
    return execute


def test_independent_models_official_first_and_four_bisections(tmp_path):
    settings, calls = config(), []
    for model in MODELS:
        start = len(calls)
        report = parameter_probe(settings, model, tmp_path, run_trial=runner(calls))
        assert calls[start]["learning_rates"] == [recipe(model)["learning_rate"]]
        assert [p["epoch"] for p in report["search_trace"][:2]] == [1.5, 1.25]
        assert len(report["search_trace"]) == report["search_steps_used"] == 4
        assert report["selected"]["epochs"] == [1.25]
        assert all(c["schedule_epochs"] == 6 for c in calls[start:])
    results = parameter_results(settings, tmp_path)
    assert results["chronos-base"]["selected"]["learning_rates"] == [1e-3]
    assert results["timesfm-2.5"]["selected"]["learning_rates"] == [1e-5]
    assert results["timemoe-200m"]["selected"]["learning_rates"] == [1e-5]


def test_task_probe_changes_only_task_and_emits_one_shared_set(tmp_path):
    settings, calls = config(), []
    for model in MODELS:
        parameter_probe(settings, model, tmp_path, run_trial=runner(calls))
    results = parameter_results(settings, tmp_path)
    assert choose_task_model(dict(settings, task_probe_model="best_relative_gain"), results) == "timesfm-2.5"
    calls.clear()
    report = task_probe(settings, tmp_path, run_trial=runner(calls))
    assert report["probe_model"] == "timemoe-200m"
    assert report["shared_tasks"] == ["b", "a"]
    assert all(c["student"] == "timemoe-200m" for c in calls)
    frozen = deepcopy(results["timemoe-200m"]["selected"])
    frozen.pop("datasets")
    for trial in calls:
        trial.pop("datasets")
        assert trial == frozen
    artifact = json.loads((tmp_path / "shared_selection.json").read_text())
    assert artifact["datasets"] == ["b", "a"] and set(artifact["models"]) == set(MODELS)
    assert all("datasets" not in item for item in artifact["models"].values())


def test_invalid_config_and_resume_identity(tmp_path):
    settings = config()
    settings["models"]["chronos-base"]["datasets"] = ["b"]
    with pytest.raises(ValueError, match="tasks are shared"):
        validate_config(settings)
    settings = config()
    settings["max_search_steps"] = 5
    with pytest.raises(ValueError, match="four"):
        validate_config(settings)
    settings = config()
    parameter_probe(settings, "chronos-base", tmp_path, run_trial=runner([]))
    settings["seeds"] = [7]
    with pytest.raises(ValueError, match="configuration changed"):
        parameter_probe(settings, "chronos-base", tmp_path, run_trial=runner([]))


@pytest.mark.parametrize("baseline,trained", [(0, 1), (None, 1), (1, float("nan")), (1, -1)])
def test_invalid_relative_gain_is_not_ranked(baseline, trained):
    with pytest.raises(ValueError):
        relative_gain(baseline, trained)


def test_total_candidate_task_limit_is_twenty():
    settings = config()
    settings["candidate_tasks"] = ["a"] + ["task_%d" % i for i in range(19)]
    validate_config(settings)
    settings["candidate_tasks"].append("task_20")
    with pytest.raises(ValueError, match="At most 20"):
        validate_config(settings)


def test_relative_ranking_is_not_absolute_error_drop():
    assert relative_gain(10, 9) < relative_gain(1, .8)


def test_missing_seed_and_exam_use_rejected():
    settings = config()
    from rage.experiments.probing import trial_config
    trial = trial_config(settings, "chronos-base", ["a"], 1e-3, [1, 2])
    report = runner([])(trial, None)
    report["trials"].pop()
    with pytest.raises(ValueError, match="seed"):
        summarize(report, ["a"], [0, 1])
    report["formal_exam_used"] = True
    with pytest.raises(ValueError, match="formal exam"):
        summarize(report, ["a"], [0])


def test_bad_parameters_do_not_silently_select_tasks():
    results = {m: {"best_relative_gain": -.1} for m in MODELS}
    with pytest.raises(ValueError, match="positive"):
        choose_task_model(config(), results)


def test_stop_bisection_at_optimizer_step_resolution(tmp_path):
    def tiny_pool(trial, out):
        report = runner([])(trial, out)
        for seed in report["trials"]:
            for point in seed["curve"]:
                point["step"] = math.ceil(point["epoch"])
                point["actual_epoch"] = point["step"]
        return report
    report = parameter_probe(config(), "chronos-base", tmp_path, run_trial=tiny_pool)
    assert report["search_steps_used"] == 0
    assert report["search_trace"] == [dict(iteration=1, action="bisect", bracket=[1., 2.],
                                         stop="optimizer_step_resolution", budget_consumed=False)]


@pytest.mark.parametrize("gain,expected,actions", [
    (lambda e: .05 * e, [3, 4, 5, 6], ["extend"] * 4),
    (lambda e: .4 - .1 * (e - 2) ** 2, [3, 2.5, 2.25, 2.125], ["extend", "bisect", "bisect", "bisect"]),
    (lambda e: .4 - .1 * (e - 3) ** 2, [3, 4, 3.5, 3.25], ["extend", "extend", "bisect", "bisect"]),
    (lambda e: -.05 * e, [.5, .25, .125, .0625], ["bisect"] * 4),
])
def test_shared_budget_for_extensions_and_bisections(tmp_path, gain, expected, actions):
    calls = []
    def curve(trial, out):
        calls.append(deepcopy(trial))
        report = runner([])(trial, out)
        for seed in report["trials"]:
            for point in seed["curve"]:
                point["metrics"]["by_dataset"]["a"]["mase"] = 10 * (1 - gain(point["epoch"]))
        return report
    settings = config()
    result = parameter_probe(settings, "chronos-base", tmp_path, run_trial=curve)
    assert [p["epoch"] for p in result["search_trace"]] == expected
    assert [p["action"] for p in result["search_trace"]] == actions
    assert result["search_steps_used"] == result["search_budget"] == 4
    assert len(calls) == len(settings["models"]["chronos-base"]["learning_rates"]) + 4
    assert all(t["schedule_epochs"] == 6 for t in calls)
    assert result["selected"]["schedule_epochs"] == 6


def test_search_policy_rejects_old_artifacts_and_preserves_budget_on_rerun(tmp_path):
    settings = config()
    report = parameter_probe(settings, "chronos-base", tmp_path, run_trial=runner([]))
    repeated = parameter_probe(settings, "chronos-base", tmp_path, run_trial=runner([]))
    assert report == repeated and repeated["search_steps_used"] == 4
    (tmp_path / "epoch_search_policy.json").unlink()
    with pytest.raises(ValueError, match="policy changed"):
        parameter_probe(settings, "chronos-base", tmp_path, run_trial=runner([]))


def test_all_cli_uses_three_environments_then_only_the_chosen_one(tmp_path, monkeypatch):
    from rage.experiments import probing
    settings = config()
    settings["task_probe_model"] = "best_relative_gain"
    for model in MODELS:
        settings["models"][model]["python"] = "/env/%s/python" % model
    path = tmp_path / "config.json"
    path.write_text(json.dumps(settings))
    out = tmp_path / "out"
    commands = []
    def process(command, check):
        commands.append(command)
        stage = command[command.index("--stage") + 1]
        if stage == "parameters":
            model = command[command.index("--student") + 1]
            parameter_probe(settings, model, out, run_trial=runner([]))
        else:
            task_probe(settings, out, run_trial=runner([]))
    monkeypatch.setattr(probing.subprocess, "run", process)
    probing.main(["--config", str(path), "--out", str(out), "--stage", "all"])
    assert [command[0] for command in commands] == ["/env/%s/python" % m for m in (*MODELS, "timesfm-2.5")]
    assert (out / "shared_selection.json").is_file()
