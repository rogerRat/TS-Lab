"""Shared CLI configuration and runtime helpers for TS-Lab tools."""
from __future__ import annotations

import sys
from pathlib import Path as _Path

_PROJECT_ROOT = str(_Path(__file__).resolve().parents[2])
for _source_path in (_PROJECT_ROOT, str(_Path(_PROJECT_ROOT) / "src")):
    if _source_path not in sys.path:
        sys.path.insert(0, _source_path)


import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

from configs.paths import PROJECT_ROOT, config_file, project_path


def load_config(path: str | Path) -> dict[str, Any]:
    source = config_file(path)
    value = yaml.safe_load(source.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError("tool config must be a YAML mapping")
    if value.get("schema_version", 1) != 1:
        raise ValueError("tool config schema_version must be 1")
    return value


def resolve(value: str | Path) -> Path:
    return project_path(value)


def apply_overrides(config: dict[str, Any], entries: list[str] | None) -> dict[str, Any]:
    """Apply explicit dotted YAML overrides without mutating the loaded mapping."""
    result = dict(config)
    for entry in entries or []:
        key, separator, raw_value = entry.partition("=")
        if not separator or not key or not raw_value:
            raise ValueError("overrides must use KEY=VALUE")
        parts = key.split(".")
        current = result
        for part in parts[:-1]:
            child = current.get(part)
            if child is None:
                child = {}
                current[part] = child
            if not isinstance(child, dict):
                raise ValueError("override parent is not a mapping: " + part)
            current = child
        current[parts[-1]] = yaml.safe_load(raw_value)
    return result


def required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise ValueError("missing required config field: " + key)
    return mapping[key]


def dispatch_model_environment(model: str, argv: list[str]) -> int | None:
    """Run the current CLI in the registered model environment when needed."""
    from rage.experiments.students import ENVIRONMENTS

    environment = ENVIRONMENTS.get(model)
    if not environment or os.environ.get("TSL_TOOL_ENV_DISPATCHED") == "1":
        return None
    if Path(sys.prefix).name == environment:
        return None
    command = ["conda", "run", "--no-capture-output", "-n", environment,
               "python", *argv]
    env = dict(os.environ)
    env["TSL_TOOL_ENV_DISPATCHED"] = "1"
    return subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False).returncode


def output_dir(category: str, value: str | Path) -> Path:
    from configs.paths import output_path

    path = output_path(category, value)
    path.mkdir(parents=True, exist_ok=True)
    return path
