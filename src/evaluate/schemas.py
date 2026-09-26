"""Strict common schema for all project evaluation runs."""

from __future__ import annotations

import copy
import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml


_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_TOP_LEVEL_FIELDS = {"schema_version", "benchmark", "model", "adapter", "bench_config"}


def resolve_public_class(spec: Any) -> type:
    """Resolve one public ``module:Class`` symbol.

    @input: spec is an ASCII module and public class attribute separated by ``:``.
    @output: Imported class object.
    @scenario: Verify an explicitly selected model adapter without a registry.
    @logic: Imports exactly the declared module and walks only public attributes.
    @param spec: Adapter class specification from the run configuration.
    """
    if not isinstance(spec, str) or not spec.isascii() or spec.count(":") != 1:
        raise ValueError("adapter.class must use ASCII module:Class syntax")
    module_name, attribute = spec.split(":", 1)
    if not module_name or not attribute or any(part.startswith("_") or not part for part in module_name.split(".")):
        raise ValueError("adapter.class must identify a public class")
    value: Any = importlib.import_module(module_name)
    for part in attribute.split("."):
        if not part or part.startswith("_"):
            raise ValueError("adapter.class attribute must be public")
        value = getattr(value, part)
    if not isinstance(value, type):
        raise TypeError("adapter.class must resolve to a class")
    return value


@dataclass(frozen=True)
class EvaluationRunConfig:
    """Immutable common evaluation contract.

    @input: A YAML mapping with the five public evaluation fields.
    @output: Validated common fields and deep-copied private mappings.
    @scenario: Share CLI and provenance semantics while keeping benchmark details private.
    @logic: Validates only common fields; each selected backend validates its own
            adapter and benchmark mappings.
    """

    schema_version: int
    benchmark: str
    model_id: str
    adapter_class: str
    adapter_config: Mapping[str, Any]
    bench_config: Mapping[str, Any]
    source_path: Path

    @classmethod
    def from_file(cls, path: Path) -> "EvaluationRunConfig":
        """Load one common evaluation YAML.

        @input: path is an existing ASCII YAML file.
        @output: Parsed immutable EvaluationRunConfig.
        @scenario: Keep all CLI callers on the same configuration boundary.
        @logic: Reads YAML once and delegates validation to from_mapping.
        @param path: Evaluation YAML path.
        """
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError("Evaluation config not found: %s" % source)
        return cls.from_mapping(yaml.safe_load(source.read_text(encoding="ascii")), source)

    @classmethod
    def from_mapping(cls, value: Any, source_path: Path) -> "EvaluationRunConfig":
        """Validate a common evaluation mapping.

        @input: value is a YAML mapping; source_path identifies its source file.
        @output: Immutable common run configuration.
        @scenario: Reject accidental benchmark-specific top-level settings.
        @logic: Requires exactly five fields and preserves private mappings without
                interpreting their contents.
        @param value: Decoded YAML mapping.
        @param source_path: Absolute or relative source YAML path.
        """
        if not isinstance(value, Mapping) or set(value) != _TOP_LEVEL_FIELDS:
            raise ValueError("evaluation config must contain only schema_version, benchmark, model, adapter, bench_config")
        version = value["schema_version"]
        if isinstance(version, bool) or version != 1:
            raise ValueError("schema_version must be integer 1")
        benchmark = value["benchmark"]
        if benchmark not in {"fev", "time", "quito"}:
            raise ValueError("benchmark must be fev, time or quito")
        model = value["model"]
        if not isinstance(model, Mapping) or set(model) != {"id"}:
            raise ValueError("model requires only id")
        model_id = model["id"]
        if not isinstance(model_id, str) or not model_id.isascii() or not _MODEL_ID.fullmatch(model_id):
            raise ValueError("model.id must be a non-empty safe ASCII identifier")
        adapter = value["adapter"]
        if not isinstance(adapter, Mapping) or set(adapter) != {"class", "config"}:
            raise ValueError("adapter requires only class and config")
        resolve_public_class(adapter["class"])
        if not isinstance(adapter["config"], Mapping) or not isinstance(value["bench_config"], Mapping):
            raise TypeError("adapter.config and bench_config must be mappings")
        return cls(
            schema_version=1,
            benchmark=benchmark,
            model_id=model_id,
            adapter_class=adapter["class"],
            adapter_config=MappingProxyType(copy.deepcopy(dict(adapter["config"]))),
            bench_config=MappingProxyType(copy.deepcopy(dict(value["bench_config"]))),
            source_path=Path(source_path).resolve(),
        )

    def as_mapping(self) -> dict[str, Any]:
        """Return a detached JSON-compatible common mapping.

        @input: No arguments.
        @output: Deep-copied run mapping with the five public fields.
        @scenario: Write stable provenance without exposing mutable internals.
        @logic: Reconstructs the public YAML shape from frozen attributes.
        """
        return {
            "schema_version": self.schema_version,
            "benchmark": self.benchmark,
            "model": {"id": self.model_id},
            "adapter": {"class": self.adapter_class, "config": copy.deepcopy(dict(self.adapter_config))},
            "bench_config": copy.deepcopy(dict(self.bench_config)),
        }


__all__ = ["EvaluationRunConfig", "resolve_public_class"]
