"""Bottom-layer immutable model identities and capability declarations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class ModelCapability(str, Enum):
    FORECAST = "forecast"
    TRAINING_LOSS = "training_loss"
    GRADIENT = "gradient"
    FINE_TUNE = "fine_tune"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    version: str
    model_id: str
    revision: str
    max_context: int
    capabilities: frozenset[ModelCapability]
    dependency_profile: str
    official_training_recipe: bool
    weights_license: str
    notes: str

    def supports(self, capability: ModelCapability | str) -> bool:
        return ModelCapability(capability) in self.capabilities

    def require(self, capability: ModelCapability | str) -> None:
        requested = ModelCapability(capability)
        if requested not in self.capabilities:
            raise NotImplementedError(
                "%s (%s) does not support %s in RAGE" % (
                    self.key, self.version, requested.value,
                )
            )

    def as_mapping(self) -> dict[str, object]:
        return {
            "key": self.key,
            "family": self.family,
            "version": self.version,
            "model_id": self.model_id,
            "revision": self.revision,
            "max_context": self.max_context,
            "capabilities": sorted(item.value for item in self.capabilities),
            "dependency_profile": self.dependency_profile,
            "official_training_recipe": self.official_training_recipe,
            "weights_license": self.weights_license,
            "notes": self.notes,
        }


_SPECS: Mapping[str, ModelSpec] = MappingProxyType({
    "timemoe-200m": ModelSpec(
        key="timemoe-200m", family="timemoe", version="200M",
        model_id="Maple728/TimeMoE-200M",
        revision="794591bfeb1225fdf742cec0f4c71f20c3f3b87e",
        max_context=4096,
        capabilities=frozenset(ModelCapability),
        dependency_profile="timemoe", official_training_recipe=True,
        weights_license="Apache-2.0",
        notes="Official multi-horizon autoregressive Huber + router loss. 200M active / 453,196,800 total parameters; BF16 full-parameter student.",
    ),
    "timesfm-3.0": ModelSpec(
        key="timesfm-3.0",
        family="timesfm",
        version="3.0",
        model_id="google/timesfm-3.0-pytorch",
        revision="43046b85ec22d584a13f8098c2ed39c889e129c2",
        max_context=15_360,
        capabilities=frozenset({ModelCapability.FORECAST}),
        dependency_profile="timesfm3",
        official_training_recipe=False,
        weights_license="timesfm-non-commercial-license-v1.0",
        notes="Latest model; official PyTorch implementation is explicitly inference-only.",
    ),
    "timesfm-2.5": ModelSpec(
        key="timesfm-2.5",
        family="timesfm",
        version="2.5",
        model_id="google/timesfm-2.5-200m-transformers",
        revision="5a9806b9b291fad9233b5249d88263f1846304d3",
        max_context=16_384,
        capabilities=frozenset({
            ModelCapability.FORECAST,
            ModelCapability.TRAINING_LOSS,
            ModelCapability.GRADIENT,
            ModelCapability.FINE_TUNE,
        }),
        dependency_profile="timesfm2_5",
        official_training_recipe=True,
        weights_license="Apache-2.0",
        notes="Trainable fallback retained because the latest 3.0 release has no training contract.",
    ),
    "sundial-base-128m": ModelSpec(
        key="sundial-base-128m",
        family="sundial",
        version="base-128m",
        model_id="thuml/sundial-base-128m",
        revision="3212e42564493f520593e5414af4367fc4b49226",
        max_context=2_880,
        capabilities=frozenset(ModelCapability),
        dependency_profile="sundial",
        official_training_recipe=False,
        weights_license="Apache-2.0",
        notes=(
            "No released official fine-tuning recipe; RAGE assembles the "
            "checkpoint-native TimeFlow components (flow_loss.py inside the "
            "pinned snapshot) on raw values, matching official inference "
            "(revin off). Probe losses use content-seeded noise for "
            "deterministic gradient semantics; training resamples per exposure."
        ),
    ),
    "timerxl-84m": ModelSpec(
        key="timerxl-84m",
        family="timerxl",
        version="base-84m",
        model_id="thuml/timer-base-84m",
        revision="70077a71acce1b4c00d98332fcaabc694255d8e5",
        max_context=9_600,
        capabilities=frozenset(ModelCapability),
        dependency_profile="timerxl",
        official_training_recipe=True,
        weights_license="Apache-2.0",
        notes=(
            "Timer/Timer-XL lineage univariate checkpoint (the Timer-XL "
            "repository's released HF forecaster; 96-point patches). Training "
            "uses the checkpoint-native shifted-patch MSE on pre-normalized "
            "windows; generate tail-truncates context to a patch multiple "
            "(512 -> 480) and normalizes the kept history by default."
        ),
    ),
})


def get_model_spec(key: str) -> ModelSpec:
    try:
        return _SPECS[key]
    except KeyError as exc:
        raise ValueError("unknown foundation model: %s" % key) from exc


def list_model_specs() -> tuple[ModelSpec, ...]:
    return tuple(_SPECS[key] for key in sorted(_SPECS))


__all__ = ["ModelCapability", "ModelSpec", "get_model_spec", "list_model_specs"]
