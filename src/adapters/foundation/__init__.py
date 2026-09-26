"""Foundation-model adapters with explicit, testable capabilities."""

from .factory import build_adapter
from .spec import ModelCapability, ModelSpec, get_model_spec, list_model_specs

__all__ = [
    "ModelCapability",
    "ModelSpec",
    "build_adapter",
    "get_model_spec",
    "list_model_specs",
]
