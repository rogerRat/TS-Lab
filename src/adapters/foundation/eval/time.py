"""TIME adapter for deterministic TimerXL and TimesFM point forecasts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .common import build_checkpoint_adapter, clean_context, predict_quantiles


class FoundationTIMEEvaluationAdapter:
    """Adapt native point forecasts to TIME's fixed quantile tensor contract."""

    @classmethod
    def parse_config(cls, value: Any, base: Path) -> Mapping[str, Any]:
        del base
        if not isinstance(value, Mapping) or set(value) != {"model", "inference"}:
            raise ValueError("adapter config requires model and inference")
        model, inference = value["model"], value["inference"]
        if set(model) != {"model_key", "device", "torch_dtype"}:
            raise ValueError("model requires model_key, device, and torch_dtype")
        if model["model_key"] not in {"timerxl-84m", "timesfm-2.5", "chronos-2-small"} or model["torch_dtype"] != "bfloat16":
            raise ValueError("unsupported model or precision")
        required = {"batch_size", "context_length", "quantile_levels", "seed", "evaluation_mode"}
        if set(inference) != required:
            raise ValueError("TIME inference config fields do not match the fixed contract")
        if int(inference["batch_size"]) < 1 or int(inference["context_length"]) < 1:
            raise ValueError("batch_size and context_length must be positive")
        levels = [float(level) for level in inference["quantile_levels"]]
        if not levels or any(not 0 < level < 1 for level in levels) or len(set(levels)) != len(levels):
            raise ValueError("quantile_levels must be unique values in (0, 1)")
        if inference["evaluation_mode"] not in {"single_window", "rolling"}:
            raise ValueError("invalid evaluation_mode")
        return {"model": dict(model), "inference": dict(inference)}

    @classmethod
    def from_config(cls, config: Mapping[str, Any], checkpoint: Path):
        return cls(checkpoint=checkpoint, config=config)

    def __init__(self, checkpoint: Path, config: Mapping[str, Any]):
        self.checkpoint = Path(checkpoint)
        self.config = config
        self._foundation = None

    def predict_quantiles(self, inputs: Iterable[Mapping[str, Any]], prediction_length: int) -> np.ndarray:
        if int(prediction_length) != prediction_length or int(prediction_length) <= 0:
            raise ValueError("prediction_length must be positive")
        if self._foundation is None:
            model = self.config["model"]
            self._foundation = build_checkpoint_adapter(
                model["model_key"], self.checkpoint, model["device"]
            )
        inference = self.config["inference"]
        contexts = []
        for entry in inputs:
            context = clean_context(np.asarray(entry["target"], dtype=np.float32))[-int(inference["context_length"]):]
            if self.config["model"]["model_key"] == "timesfm-2.5":
                required_length = max(32, ((len(context) + 31) // 32) * 32)
                if len(context) < required_length:
                    context = np.pad(context, (required_length - len(context), 0), mode="edge")
            contexts.append(context.astype(np.float32, copy=False))
        if not contexts:
            raise ValueError("TIME task produced no prediction inputs")
        levels = [float(value) for value in inference["quantile_levels"]]
        output = np.empty((len(contexts), len(levels), int(prediction_length)), dtype=np.float32)
        groups: dict[int, list[int]] = {}
        for index, values in enumerate(contexts):
            groups.setdefault(len(values), []).append(index)
        for indices in groups.values():
            for begin in range(0, len(indices), int(inference["batch_size"])):
                rows = indices[begin:begin + int(inference["batch_size"])]
                batch = np.stack([contexts[index] for index in rows])
                output[rows] = predict_quantiles(
                    self._foundation, batch, int(prediction_length), levels)
        if not np.isfinite(output).all():
            raise ValueError("TIME deterministic quantile output contains non-finite values")
        return output


__all__ = ["FoundationTIMEEvaluationAdapter"]
