from types import SimpleNamespace

from adapters.chronos.eval.fev import ChronosFEVEvaluationAdapter, configure_evaluation_windows
from adapters.chronos.eval.time import ChronosTIMEEvaluationAdapter
from evaluate.time import evaluation_test_length


def fev_config(**inference_overrides):
    inference = {
        "batch_size": 2,
        "num_samples": 4,
        "temperature": 1.0,
        "top_k": 2,
        "top_p": 1.0,
        "seed": 3,
        **inference_overrides,
    }
    return {"model": {"device": "cpu", "torch_dtype": "float32"}, "inference": inference}


def time_config(**inference_overrides):
    inference = {
        "batch_size": 2,
        "context_length": 8,
        "num_samples": 4,
        "temperature": 1.0,
        "top_k": 2,
        "top_p": 1.0,
        "quantile_levels": [0.1, 0.5, 0.9],
        "seed": 3,
        **inference_overrides,
    }
    return {"model": {"device": "cpu", "torch_dtype": "float32"}, "inference": inference}


def test_fev_defaults_to_latest_single_ground_truth_window(tmp_path):
    parsed = ChronosFEVEvaluationAdapter.parse_config(fev_config(), tmp_path)
    assert parsed["inference"]["evaluation_mode"] == "single_window"
    task = SimpleNamespace(num_windows=3, cutoffs=[10, 20, 30], initial_cutoff=10)
    configure_evaluation_windows(task, parsed["inference"]["evaluation_mode"])
    assert task.num_windows == 1
    assert task.initial_cutoff == 30


def test_fev_rolling_preserves_all_cutoffs(tmp_path):
    parsed = ChronosFEVEvaluationAdapter.parse_config(
        fev_config(evaluation_mode="rolling"), tmp_path
    )
    task = SimpleNamespace(num_windows=3, cutoffs=[10, 20, 30], initial_cutoff=10)
    configure_evaluation_windows(task, parsed["inference"]["evaluation_mode"])
    assert task.num_windows == 3
    assert task.initial_cutoff == 10


def test_time_defaults_to_one_horizon_and_supports_rolling(tmp_path):
    default = ChronosTIMEEvaluationAdapter.parse_config(time_config(), tmp_path)
    rolling = ChronosTIMEEvaluationAdapter.parse_config(
        time_config(evaluation_mode="rolling"), tmp_path
    )
    assert default["inference"]["evaluation_mode"] == "single_window"
    assert rolling["inference"]["evaluation_mode"] == "rolling"
    assert evaluation_test_length(96, 24, "single_window") == 24
    assert evaluation_test_length(96, 24, "rolling") == 96
