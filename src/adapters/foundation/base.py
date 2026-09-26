"""Middle-layer model adapter protocol used by experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .spec import ModelCapability, ModelSpec


class FoundationModelAdapter(ABC):
    def __init__(self, spec: ModelSpec, *, device: str = "auto") -> None:
        self.spec = spec
        self.device = device
        self._model: Any | None = None

    @abstractmethod
    def load(self) -> Any:
        """Load the revision-pinned upstream model lazily."""

    @abstractmethod
    def forecast(self, contexts: np.ndarray, *, prediction_length: int) -> np.ndarray:
        """Return a [batch, horizon] point forecast."""

    def training_loss(self, windows: np.ndarray, *, context_length: int, prediction_length: int):
        self.spec.require(ModelCapability.TRAINING_LOSS)
        raise NotImplementedError

    def describe(self) -> dict[str, object]:
        return self.spec.as_mapping()


__all__ = ["FoundationModelAdapter"]
