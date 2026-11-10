"""FEV adapter for the deterministic TimerXL and TimesFM forecasters."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping

import datasets
import fev
import numpy as np

from adapters.chronos.eval.fev import configure_evaluation_windows
from .common import build_checkpoint_adapter, clean_context, tiled_quantiles


class FoundationFEVEvaluationAdapter(fev.ForecastingModel):
    """Serve one local TimerXL or TimesFM checkpoint to FEV."""

    @classmethod
    def parse_config(cls, value: Any, base: Path) -> Mapping[str, Any]:
        del base
        if not isinstance(value, Mapping) or set(value) != {"model", "inference"}:
            raise ValueError("adapter config requires model and inference")
        model, inference = value["model"], value["inference"]
        if set(model) != {"model_key", "device", "torch_dtype"}:
            raise ValueError("model requires model_key, device, and torch_dtype")
        if model["model_key"] not in {"timerxl-84m", "timesfm-2.5"}:
            raise ValueError("unsupported foundation model key")
        if model["torch_dtype"] != "bfloat16":
            raise ValueError("foundation evaluations require BF16")
        if set(inference) != {"batch_size", "evaluation_mode", "seed"}:
            raise ValueError("inference requires batch_size, evaluation_mode, and seed")
        if (int(inference["batch_size"]) < 1 or inference["evaluation_mode"] not in {"single_window", "rolling"}
                or int(inference["seed"]) < 0):
            raise ValueError("invalid foundation evaluation inference settings")
        return {"model": dict(model), "inference": dict(inference)}

    @classmethod
    def from_config(cls, config: Mapping[str, Any], checkpoint: Path, seed: int):
        return cls(checkpoint=checkpoint, config=config, seed=seed)

    def __init__(self, checkpoint: Path, config: Mapping[str, Any], seed: int):
        super().__init__()
        self.checkpoint = Path(checkpoint)
        self.config = config
        self.seed = int(seed)
        self.batch_size = int(config["inference"]["batch_size"])
        self.evaluation_mode = config["inference"]["evaluation_mode"]
        self._foundation = None

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        import random
        configure_evaluation_windows(task, self.evaluation_mode)
        random.seed(self.seed)
        np.random.seed(self.seed)
        if self._foundation is None:
            model = self.config["model"]
            self._foundation = build_checkpoint_adapter(
                model["model_key"], self.checkpoint, model["device"]
            )
        dataset_path = getattr(task, "dataset_path", None)
        restore_dataset_config = None
        if dataset_path is not None:
            if not Path(dataset_path).is_file():
                raise FileNotFoundError("FEV local dataset is missing: %s" % dataset_path)
            restore_dataset_config = task.dataset_config
            task.dataset_config = None
        levels = list(task.quantile_levels)
        if 0.5 not in levels:
            levels.append(0.5)
        result = []
        for window in task.iter_windows():
            past_data, _ = fev.convert_input_data(window, adapter="datasets", as_univariate=True)
            past_data = past_data.select_columns(["target"]).cast_column(
                "target", datasets.Sequence(datasets.Value("float32"))
            )
            contexts = [clean_context(np.asarray(item["target"], dtype=np.float32))[-512:]
                        for item in past_data]
            forecasts = np.empty((len(contexts), int(task.horizon)), dtype=np.float32)
            by_length: dict[int, list[int]] = {}
            for index, context in enumerate(contexts):
                by_length.setdefault(len(context), []).append(index)
            with self._record_inference_time():
                for indices in by_length.values():
                    for begin in range(0, len(indices), self.batch_size):
                        rows = indices[begin:begin + self.batch_size]
                        batch = np.stack([contexts[index] for index in rows])
                        point = self._foundation.forecast(batch, prediction_length=int(task.horizon))
                        forecasts[rows] = np.asarray(point, dtype=np.float32)
            quantiles = tiled_quantiles(forecasts, levels)
            predictions = {"predictions": forecasts}
            for level in task.quantile_levels:
                predictions[str(level)] = quantiles[:, levels.index(level), :]
            result.append(fev.utils.combine_univariate_predictions_to_multivariate(
                datasets.Dataset.from_dict(predictions), target_columns=task.target_columns
            ))
        if restore_dataset_config is not None:
            task.dataset_config = restore_dataset_config
        return result


__all__ = ["FoundationFEVEvaluationAdapter"]
