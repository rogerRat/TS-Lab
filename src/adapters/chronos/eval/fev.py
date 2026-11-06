# Derived from autogluon/fev models/chronos/model.py at
# commit 96535744f69e066598004632fe73f0bc534100e0.
# SPDX-License-Identifier: Apache-2.0

"""Chronos-to-FEV model adapter for project-trained checkpoints.

This file is derived from ``models/chronos/model.py`` in autogluon/fev at
commit 96535744f69e066598004632fe73f0bc534100e0. The project-owned copy makes
the effective sampling arguments explicit and removes runtime source loading.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import datasets
import numpy as np

import fev


@dataclass(frozen=True)
class ChronosFEVEvaluationConfig:
    """Frozen normalized Chronos FEV inference settings."""

    values: Mapping[str, Any]


def configure_evaluation_windows(task: Any, evaluation_mode: str) -> Any:
    """Select all rolling cutoffs or only the latest ground-truth cutoff."""
    if evaluation_mode not in {"single_window", "rolling"}:
        raise ValueError("evaluation_mode must be single_window or rolling")
    if evaluation_mode == "single_window" and getattr(task, "num_windows", 1) > 1:
        final_cutoff = task.cutoffs[-1]
        task.initial_cutoff = final_cutoff
        task.num_windows = 1
    return task


class ChronosFEVEvaluationAdapter(fev.ForecastingModel):
    """Adapt one local Chronos checkpoint to the pinned FEV API.

    @input: Local model path, device, dtype, batch size, and sample count.
    @output: FEV-compatible point and quantile predictions.
    @scenario: Evaluate the current Chronos tiny research checkpoints without
               importing an adapter from the third-party source checkout.
    """

    model_name = "chronos"

    @classmethod
    def parse_config(cls, value: Any, base: Path) -> Mapping[str, Any]:
        """Validate and normalize model and explicit sampling settings."""
        del base
        if not isinstance(value, Mapping) or set(value) != {"model", "inference"}:
            raise ValueError("adapter_config requires only model and inference")
        model = value["model"]
        inference = value["inference"]
        if not isinstance(model, Mapping) or not isinstance(inference, Mapping):
            raise TypeError("model and inference must be mappings")
        if set(model) != {"device", "torch_dtype"}:
            raise ValueError("model requires device and torch_dtype only")
        if not isinstance(model["device"], str) or not model["device"] or not model["device"].isascii():
            raise ValueError("model.device must be non-empty ASCII text")
        if model["torch_dtype"] not in ("float32", "float16", "bfloat16"):
            raise ValueError("unsupported torch_dtype")
        required = {"batch_size", "num_samples", "temperature", "top_k", "top_p", "seed"}
        if set(inference) not in (required, required | {"evaluation_mode"}):
            raise ValueError("inference requires batch_size, num_samples, temperature, top_k, top_p, seed, and optional evaluation_mode")
        def positive_int(name: str, minimum: int = 1) -> int:
            v = inference[name]
            if isinstance(v, bool) or int(v) != v or int(v) < minimum:
                raise ValueError("%s must be an integer >= %d" % (name, minimum))
            return int(v)
        batch_size = positive_int("batch_size")
        num_samples = positive_int("num_samples", 2)
        temperature = float(inference["temperature"])
        top_p = float(inference["top_p"])
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and > 0")
        top_k = inference["top_k"]
        if isinstance(top_k, bool) or int(top_k) != top_k or int(top_k) == 1 or int(top_k) < 0:
            raise ValueError("top_k must be 0 or an integer >= 2")
        if not np.isfinite(top_p) or not 0 < top_p <= 1:
            raise ValueError("top_p must be finite and in (0, 1]")
        seed = inference["seed"]
        if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
            raise ValueError("seed must be a non-negative integer")
        evaluation_mode = inference.get("evaluation_mode", "single_window")
        if evaluation_mode not in {"single_window", "rolling"}:
            raise ValueError("evaluation_mode must be single_window or rolling")
        return copy.deepcopy({"model": dict(model), "inference": {
            "batch_size": batch_size, "num_samples": num_samples, "temperature": temperature,
            "top_k": int(top_k), "top_p": top_p, "seed": int(seed),
            "evaluation_mode": evaluation_mode}})

    @classmethod
    def from_config(cls, config: Mapping[str, Any], checkpoint: Path, seed: int) -> "ChronosFEVEvaluationAdapter":
        """Create one FEV adapter for a checkpoint and derived task seed."""
        model = config["model"]
        inference = config["inference"]
        return cls(str(checkpoint), device=model["device"], torch_dtype=model["torch_dtype"],
                   batch_size=inference["batch_size"], num_samples=inference["num_samples"],
                   temperature=inference["temperature"], top_k=inference["top_k"], top_p=inference["top_p"],
                   seed=seed, evaluation_mode=inference["evaluation_mode"])

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        batch_size: int = 256,
        num_samples: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        seed: int = 0,
        evaluation_mode: str = "single_window",
    ) -> None:
        """Store explicit Chronos inference parameters.

        @input: model_path is a local checkpoint; device is a PyTorch device
                map; torch_dtype is accepted by from_pretrained; batch_size
                and num_samples are positive integers.
        @output: Configured FEV forecasting model.
        @scenario: Ensure metadata values are the actual inference values.
        @param model_path: Local Chronos checkpoint directory.
        @param device: cuda, cpu, or auto device selection.
        @param torch_dtype: Requested model loading dtype.
        @param batch_size: Maximum series per inference batch.
        @param num_samples: Sample paths used for quantile estimation.
        """
        super().__init__()
        if batch_size <= 0 or num_samples < 2:
            raise ValueError("batch_size and num_samples must be positive")
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.batch_size = int(batch_size)
        self.num_samples = int(num_samples)
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.seed = int(seed)
        self.evaluation_mode = evaluation_mode
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and > 0")
        if self.top_k < 0 or self.top_k == 1:
            raise ValueError("top_k must be 0 or >= 2")
        if not np.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be finite and in (0, 1]")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.evaluation_mode not in {"single_window", "rolling"}:
            raise ValueError("evaluation_mode must be single_window or rolling")

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        """Generate predictions for every window in one FEV task.

        @input: task is a fully declared FEV evaluation task.
        @output: One prediction DatasetDict per evaluation window.
        @scenario: Convert FEV inputs to univariate Chronos batches and restore
                   the required multivariate prediction layout.
        @param task: Active FEV task.
        """
        import torch
        from chronos import BaseChronosPipeline

        # Default to one forecast from the latest available ground-truth
        # history.  In rolling mode FEV evaluates every declared cutoff.  We
        # never append model output to context in either mode.
        configure_evaluation_windows(task, self.evaluation_mode)

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = fev.utils.maybe_cache_from_s3(self.model_path)
        pipeline = BaseChronosPipeline.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=self.torch_dtype,
        )
        # The benchmark declaration keeps dataset_config for the RAGE
        # reference-catalog lookup, but evaluation must load the already
        # materialized file from data/bench/fev.  FEV's stock loader selects
        # Hub loading whenever dataset_config is present, so temporarily
        # switch this task to its local-file branch for this process.
        restore_dataset_config = None
        dataset_path = getattr(task, "dataset_path", None)
        if dataset_path is None:
            pass  # lightweight adapter-contract tests do not materialize a dataset
        elif Path(dataset_path).is_file():
            restore_dataset_config = task.dataset_config
            task.dataset_config = None
        else:
            raise FileNotFoundError(
                "FEV evaluation requires an existing local dataset: %s" % dataset_path
            )
        quantile_levels = task.quantile_levels.copy()
        if 0.5 not in quantile_levels:
            quantile_levels.append(0.5)

        predictions_per_window = []
        for window in task.iter_windows():
            past_data, _ = fev.convert_input_data(window, adapter="datasets", as_univariate=True)
            past_data = past_data.select_columns(["target"]).cast_column(
                "target", datasets.Sequence(datasets.Value("float32"))
            ).with_format("torch")
            quantiles_all = []
            mean_all = []
            with self._record_inference_time():
                for batch in self._batchify(past_data):
                    quantiles, mean = pipeline.predict_quantiles(
                        batch,
                        prediction_length=task.horizon,
                        limit_prediction_length=False,
                        quantile_levels=quantile_levels,
                        num_samples=self.num_samples,
                        temperature=self.temperature,
                        top_k=self.top_k,
                        top_p=self.top_p,
                    )
                    quantiles_all.append(quantiles.numpy())
                    mean_all.append(mean.numpy())
            quantiles_np = np.concatenate(quantiles_all, axis=0)
            mean_np = np.concatenate(mean_all, axis=0)
            if task.eval_metric in {"MSE", "RMSE", "RMSSE"}:
                point_forecast = mean_np
            else:
                point_forecast = quantiles_np[:, :, quantile_levels.index(0.5)]
            predictions = {"predictions": point_forecast}
            for index, level in enumerate(task.quantile_levels):
                predictions[str(level)] = quantiles_np[:, :, index]
            predictions_per_window.append(
                fev.utils.combine_univariate_predictions_to_multivariate(
                    datasets.Dataset.from_dict(predictions),
                    target_columns=task.target_columns,
                )
            )
        if restore_dataset_config is not None:
            task.dataset_config = restore_dataset_config
        return predictions_per_window

    def _batchify(self, dataset: Any) -> Iterator[list[Any]]:
        """Iterate univariate target batches without materializing the full column.

        @input: dataset is a formatted Hugging Face Dataset with target column.
        @output: Consecutive non-empty target batches no larger than batch_size.
        @scenario: Bound FEV adapter memory by one inference batch.
        @param dataset: Current task window dataset.
        """
        for batch in dataset.iter(batch_size=self.batch_size):
            values = batch.get("target", [])
            if len(values) > 0:
                yield values


__all__ = ["ChronosFEVEvaluationAdapter", "configure_evaluation_windows"]
