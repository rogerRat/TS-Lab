"""Tests for the project-owned FEV task declaration."""

import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from adapters.chronos.eval.fev import ChronosFEVEvaluationAdapter
from evaluate.fev import RQ3_METRICS, _parse_fev_bench_config, _warn_reference_task_mismatch, load_task_declarations


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _metric_name(value):
    """Normalize one test metric declaration.

    @input: value is a string or mapping metric declaration.
    @output: Metric name string.
    @scenario: Compare every task with the fixed project metric set.
    @param value: Raw YAML metric value.
    """
    return value["name"] if isinstance(value, dict) else value


class EvaluateFEVTest(unittest.TestCase):
    """Verify the direct twenty-task FEV research protocol."""

    def test_project_declaration_contains_twenty_direct_tasks(self):
        """The project YAML is complete and does not redirect to a checkout."""
        path = PROJECT_ROOT / "configs/eval/fev/default.yaml"
        tasks = load_task_declarations(path)
        self.assertEqual(len(tasks), 20)
        self.assertNotIn("base_task_file", path.read_text(encoding="utf-8"))

    def test_every_task_declares_all_six_metrics(self):
        """Each task explicitly contains official and added research metrics."""
        tasks = load_task_declarations(
            PROJECT_ROOT / "configs/eval/fev/default.yaml"
        )
        expected = set(RQ3_METRICS)
        for task in tasks:
            names = {_metric_name(task["eval_metric"])}
            names.update(_metric_name(value) for value in task["extra_metrics"])
            self.assertEqual(names, expected)

    def test_explicit_single_task_selection_uses_its_real_index(self):
        path, expected, indices = _parse_fev_bench_config(
            {"task_file": "default.yaml", "expected_tasks": 1, "task_indices": [13]},
            PROJECT_ROOT / "configs/eval/fev",
        )
        self.assertEqual(path, PROJECT_ROOT / "configs/eval/fev/default.yaml")
        self.assertEqual(expected, 1)
        self.assertEqual(indices, (13,))

    def test_explicit_task_selection_rejects_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "equal"):
            _parse_fev_bench_config(
                {"task_file": "default.yaml", "expected_tasks": 2, "task_indices": [13]},
                PROJECT_ROOT / "configs/eval/fev",
            )

    def test_reference_task_mismatch_warning_is_prominent(self):
        import json
        import shutil
        import tempfile

        checkpoint = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(checkpoint))
        (checkpoint / "ts3_checkpoint_manifest.json").write_text(
            json.dumps({"reference_task_record": {"benchmark": "fev", "task_indices": [13]}}),
            encoding="ascii",
        )
        with self.assertLogs("evaluate", level="WARNING") as logged:
            _warn_reference_task_mismatch(checkpoint, [7])
        message = "\n".join(logged.output)
        self.assertIn("====== REFERENCE TASK MISMATCH ======", message)
        self.assertIn("checkpoint reference FEV tasks: [13]", message)
        self.assertIn("evaluation selected FEV tasks: [7]", message)

    def test_reference_task_check_is_skipped_for_random_checkpoint(self):
        import json
        import shutil
        import tempfile

        checkpoint = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(checkpoint))
        (checkpoint / "ts3_checkpoint_manifest.json").write_text(
            json.dumps({"reference_task_record": None}),
            encoding="ascii",
        )
        (checkpoint / "run_manifest.json").write_text(
            json.dumps({"policy": "random"}),
            encoding="ascii",
        )
        with self.assertLogs("evaluate", level="INFO") as logged:
            _warn_reference_task_mismatch(checkpoint, [2])
        message = "\n".join(logged.output)
        self.assertIn("Reference task check skipped", message)
        self.assertNotIn("WARNING", message)

    def test_num_samples_is_forwarded_to_chronos_prediction(self):
        """The declared evaluation sample count controls actual model inference."""
        calls = []

        class _PastData:
            """Small datasets-like input used by the adapter test."""

            def select_columns(self, names):
                """Require the adapter to remove non-model input columns."""
                self.selected_columns = names
                return self

            def with_format(self, _format):
                """Return this fake after a format request."""
                return self

            def cast_column(self, _name, _feature):
                """Return this fake after a schema request."""
                return self

            def iter(self, batch_size):
                """Yield one target batch through the datasets-style API."""
                yield {"target": torch.arange(16, dtype=torch.float32).reshape(2, 8)[:batch_size]}

        class _Pipeline:
            """Record predict_quantiles keyword arguments."""

            def predict_quantiles(self, batch, **kwargs):
                """Return correctly shaped deterministic forecast tensors."""
                calls.append(kwargs)
                count = len(batch)
                return torch.zeros(count, 2, 3), torch.zeros(count, 2)

        class _Task:
            """Minimum task fields consumed by ChronosFEVEvaluationAdapter._fit_predict."""

            quantile_levels = [0.1, 0.5, 0.9]
            horizon = 2
            eval_metric = "SQL"
            target_columns = ["target"]

            def iter_windows(self):
                """Yield one fake evaluation window."""
                yield object()

        model = ChronosFEVEvaluationAdapter("checkpoint", device="cpu", batch_size=2, num_samples=17)
        with patch("chronos.BaseChronosPipeline.from_pretrained", return_value=_Pipeline()), patch(
            "fev.convert_input_data", return_value=(_PastData(), None)
        ), patch(
            "fev.utils.combine_univariate_predictions_to_multivariate", return_value="prediction"
        ):
            result = model._fit_predict(_Task())
        self.assertEqual(result, ["prediction"])
        self.assertEqual(calls[0]["num_samples"], 17)


if __name__ == "__main__":
    unittest.main()
