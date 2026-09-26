"""Chronos-T5 adapter for the pinned TIME benchmark."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


class ChronosTIMEEvaluationAdapter:
    """Load one original Chronos-T5 checkpoint and forecast TIME windows.

    @input: A local Chronos-T5 checkpoint and explicit inference parameters.
    @output: Quantile arrays shaped ``(instances, quantiles, horizon)``.
    @scenario: Evaluate univariate TIME data and TIME's flattened multivariate
               path without using Chronos-2's native multivariate interface.
    @logic: Lazily loads BaseChronosPipeline once, truncates every univariate
            context to the configured length, and validates every batch result.
    """

    def __init__(self, checkpoint: Path, config: Mapping[str, Any]) -> None:
        """Store a validated checkpoint and normalized inference config.

        @input: checkpoint is a Chronos-T5 directory; config is normalized.
        @output: Ready adapter with no loaded model until first forecast.
        @scenario: Keep one model instance for serial TIME task execution.
        @logic: Defers heavyweight model loading while retaining explicit values.
        @param checkpoint: Complete project checkpoint directory.
        @param config: Validated adapter.config mapping.
        """
        self.checkpoint = Path(checkpoint).resolve()
        self.config = copy.deepcopy(dict(config))
        self._pipeline: Any = None

    @classmethod
    def parse_config(cls, value: Any, base: Path) -> Mapping[str, Any]:
        """Strictly validate TIME model inference settings.

        @input: value is adapter.config; base is unused configuration context.
        @output: Deep-copied normalized settings mapping.
        @scenario: Prevent TIME data/task/source settings from leaking into adapter config.
        @logic: Requires model and inference mappings with all sampling values explicit.
        @param value: Raw adapter configuration.
        @param base: Configuration parent directory, unused by this adapter.
        """
        del base
        if not isinstance(value, Mapping) or set(value) != {"model", "inference"}:
            raise ValueError("TIME adapter.config requires only model and inference")
        model = value["model"]
        inference = value["inference"]
        if not isinstance(model, Mapping) or set(model) != {"device", "torch_dtype"}:
            raise ValueError("TIME model requires only device and torch_dtype")
        if not isinstance(inference, Mapping):
            raise TypeError("TIME inference must be a mapping")
        required = {"batch_size", "context_length", "num_samples", "temperature", "top_k", "top_p", "quantile_levels", "seed"}
        if set(inference) not in (required, required | {"evaluation_mode"}):
            raise ValueError("TIME inference requires batch_size, context_length, num_samples, temperature, top_k, top_p, quantile_levels, seed, and optional evaluation_mode")
        device = model["device"]
        dtype = model["torch_dtype"]
        if not isinstance(device, str) or not device.isascii() or not device:
            raise ValueError("TIME model.device must be non-empty ASCII text")
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("unsupported TIME torch_dtype")

        def integer(name: str, minimum: int, allow_zero: bool = False) -> int:
            raw = inference[name]
            if isinstance(raw, bool) or int(raw) != raw or (int(raw) < minimum and not (allow_zero and int(raw) == 0)):
                raise ValueError("TIME inference.%s has an invalid integer value" % name)
            return int(raw)

        batch_size = integer("batch_size", 1)
        context_length = integer("context_length", 1)
        num_samples = integer("num_samples", 2)
        top_k = integer("top_k", 2, allow_zero=True)
        seed = integer("seed", 0)
        temperature = float(inference["temperature"])
        top_p = float(inference["top_p"])
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("TIME temperature must be finite and > 0")
        if not np.isfinite(top_p) or not 0 < top_p <= 1:
            raise ValueError("TIME top_p must be finite and in (0, 1]")
        levels = inference["quantile_levels"]
        if not isinstance(levels, list) or len(levels) < 2:
            raise ValueError("TIME quantile_levels must be a list with at least two values")
        normalized_levels = [float(level) for level in levels]
        if normalized_levels != sorted(normalized_levels) or len(set(normalized_levels)) != len(normalized_levels):
            raise ValueError("TIME quantile_levels must be strictly increasing")
        if any(not np.isfinite(level) or not 0 < level < 1 for level in normalized_levels) or 0.5 not in normalized_levels:
            raise ValueError("TIME quantile_levels must be finite values in (0, 1) including 0.5")
        evaluation_mode = inference.get("evaluation_mode", "single_window")
        if evaluation_mode not in {"single_window", "rolling"}:
            raise ValueError("TIME evaluation_mode must be single_window or rolling")
        return {
            "model": {"device": device, "torch_dtype": dtype},
            "inference": {
                "batch_size": batch_size,
                "context_length": context_length,
                "num_samples": num_samples,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "quantile_levels": normalized_levels,
                "seed": seed,
                "evaluation_mode": evaluation_mode,
            },
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any], checkpoint: Path) -> "ChronosTIMEEvaluationAdapter":
        """Create a TIME adapter after checking original Chronos-T5 metadata.

        @input: config is normalized settings; checkpoint is a project model directory.
        @output: Configured adapter instance.
        @scenario: Reject incompatible Chronos-2 or non-Chronos model checkpoints.
        @logic: Reads config.json and requires a mapping-valued chronos_config.
        @param config: Normalized TIME adapter configuration.
        @param checkpoint: Checkpoint directory supplied by the unified CLI.
        """
        config_path = Path(checkpoint) / "config.json"
        try:
            checkpoint_config = json.loads(config_path.read_text(encoding="ascii"))
        except (OSError, ValueError) as error:
            raise ValueError("cannot read checkpoint config.json") from error
        chronos_config = checkpoint_config.get("chronos_config") if isinstance(checkpoint_config, Mapping) else None
        if not isinstance(chronos_config, Mapping) or not chronos_config.get("tokenizer_class"):
            raise ValueError("TIME supports only original Chronos-T5 checkpoints with chronos_config")
        return cls(Path(checkpoint), config)

    def _load_pipeline(self) -> Any:
        """Load BaseChronosPipeline exactly once.

        @input: No arguments.
        @output: Loaded Chronos pipeline.
        @scenario: Avoid a model copy per TIME task.
        @logic: Uses explicit device and dtype from the normalized config.
        """
        if self._pipeline is None:
            from chronos import BaseChronosPipeline

            model = self.config["model"]
            self._pipeline = BaseChronosPipeline.from_pretrained(
                str(self.checkpoint), device_map=model["device"], torch_dtype=model["torch_dtype"]
            )
        return self._pipeline

    def predict_quantiles(self, inputs: Iterable[Mapping[str, Any]], prediction_length: int) -> np.ndarray:
        """Forecast TIME's univariate test windows.

        @input: inputs yields GluonTS entries with one-dimensional target arrays;
                prediction_length is a positive TIME horizon.
        @output: Finite float32 array ``(instances, quantiles, prediction_length)``.
        @scenario: Serve native univariate data and TIME's to_univariate expansion.
        @logic: Seeds all sampling RNGs, truncates each context, predicts in batches,
                converts Chronos orientation, and validates every output dimension.
        @param inputs: TIME test-data input entries in TIME's declared order.
        @param prediction_length: Forecast horizon for the active task.
        """
        if isinstance(prediction_length, bool) or int(prediction_length) != prediction_length or int(prediction_length) <= 0:
            raise ValueError("prediction_length must be a positive integer")
        import torch

        inference = self.config["inference"]
        seed = int(inference["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        contexts = []
        for entry in inputs:
            target = np.asarray(entry["target"], dtype=np.float32)
            if target.ndim != 1:
                raise ValueError("TIME adapter expects univariate windows; backend must use to_univariate")
            target = target[-int(inference["context_length"]):]
            if target.size == 0 or not np.isfinite(target).all():
                raise ValueError("TIME input context must be non-empty and finite")
            contexts.append(torch.tensor(target, dtype=torch.float32))
        if not contexts:
            raise ValueError("TIME task produced no prediction inputs")
        pipeline = self._load_pipeline()
        predictions = []
        quantiles = list(inference["quantile_levels"])
        for start in range(0, len(contexts), int(inference["batch_size"])):
            batch = contexts[start:start + int(inference["batch_size"])]
            batch_quantiles, _ = pipeline.predict_quantiles(
                batch,
                prediction_length=int(prediction_length),
                limit_prediction_length=False,
                quantile_levels=quantiles,
                num_samples=int(inference["num_samples"]),
                temperature=float(inference["temperature"]),
                top_k=int(inference["top_k"]),
                top_p=float(inference["top_p"]),
            )
            array = batch_quantiles.detach().cpu().float().numpy() if hasattr(batch_quantiles, "detach") else np.asarray(batch_quantiles)
            expected = (len(batch), int(prediction_length), len(quantiles))
            if array.shape == expected:
                array = np.transpose(array, (0, 2, 1))
            elif array.shape != (len(batch), len(quantiles), int(prediction_length)):
                raise ValueError("unexpected Chronos quantile shape: %s" % (array.shape,))
            if not np.isfinite(array).all():
                raise ValueError("Chronos returned non-finite TIME quantiles")
            predictions.append(array.astype(np.float32, copy=False))
        result = np.concatenate(predictions, axis=0)
        if result.shape != (len(contexts), len(quantiles), int(prediction_length)):
            raise ValueError("TIME quantile concatenation has invalid shape: %s" % (result.shape,))
        return result


__all__ = ["ChronosTIMEEvaluationAdapter"]
