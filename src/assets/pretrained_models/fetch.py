#!/usr/bin/env python3
"""Fetch registered pretrained models with shared field validation and atomic publication."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRETRAINED_MODELS_ROOT = (PROJECT_ROOT / "cache" / "pretrained_models").resolve()


def arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("chronos", "timesfm", "timemoe"))
    parser.add_argument("model", help="registered model key")
    parser.add_argument("--destination", type=Path, help="explicit model directory on the data disk")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=Path, default=None, help="registry YAML")
    args = parser.parse_args(argv)
    args.config = args.config or PROJECT_ROOT / "configs" / "pretrained_models" / (args.family + ".yaml")
    return args


def project_path(value: str) -> Path:
    candidate = Path(value)
    path = candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()
    if path != PRETRAINED_MODELS_ROOT and PRETRAINED_MODELS_ROOT not in path.parents:
        raise ValueError("destination must be below %s: %s" % (PRETRAINED_MODELS_ROOT, path))
    return path


def load_entry(config_path: Path, model_name: str) -> Mapping[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping) or not isinstance(config.get("models"), Mapping):
        raise ValueError("registry must contain a models mapping")
    try:
        entry = config["models"][model_name]
    except KeyError as exc:
        raise ValueError("unknown registered model: %s" % model_name) from exc
    required = {"repo_id", "revision", "destination", "architecture", "parameter_count_range", "status"}
    if not isinstance(entry, Mapping) or not required.issubset(entry):
        raise ValueError("model entry %s must contain exactly %s" % (model_name, sorted(required)))
    if entry["status"] not in ("enabled", "enabled-trainable"):
        raise ValueError("model %s is %s and cannot be fetched" % (model_name, entry["status"]))
    if not all(isinstance(entry[field], str) and entry[field] for field in ("repo_id", "revision", "destination", "architecture")):
        raise ValueError("model %s has invalid text fields" % model_name)
    parameter_range = entry["parameter_count_range"]
    if (
        not isinstance(parameter_range, list)
        or len(parameter_range) != 2
        or not all(isinstance(value, int) and value > 0 for value in parameter_range)
        or parameter_range[0] > parameter_range[1]
    ):
        raise ValueError("model %s has invalid parameter_count_range" % model_name)
    return entry


def model_parameter_count(path: Path, family="chronos") -> tuple[str, int]:
    import torch
    if family == "chronos":
        from transformers import AutoModelForSeq2SeqLM as model_class
    elif family == "timesfm":
        from transformers import TimesFm2_5ModelForPrediction as model_class
    else:
        from adapters.foundation.timemoe import official_classes
        _, model_class = official_classes()
    model = model_class.from_pretrained(str(path), local_files_only=True, torch_dtype=torch.bfloat16)
    architecture = model.__class__.__name__
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    del model
    return architecture, parameter_count


def validate_checkpoint(path: Path, entry: Mapping[str, Any], family="chronos") -> None:
    config_path = path / "config.json"
    if not config_path.is_file():
        raise ValueError("missing config.json: %s" % path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or entry["architecture"] not in architectures:
        raise ValueError("unexpected architecture in %s" % config_path)
    architecture, parameter_count = model_parameter_count(path, family)
    lower_bound, upper_bound = entry["parameter_count_range"]
    if architecture != entry["architecture"] or not lower_bound <= parameter_count <= upper_bound:
        raise ValueError(
            "checkpoint validation failed: architecture=%s parameters=%d expected_range=%d..%d"
            % (architecture, parameter_count, lower_bound, upper_bound)
        )


@contextmanager
def exclusive_lock(destination: Path) -> Iterator[None]:
    lock_path = destination.parent / (destination.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl

        with lock_path.open("w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield
    finally:
        pass


def download_checkpoint(staging_dir: Path, entry: Mapping[str, Any]) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=entry["repo_id"],
        revision=entry["revision"],
        local_dir=str(staging_dir),
        local_dir_use_symlinks=False,
    )


def publish_model_fields(path, destination, entry, family):
    fields = dict(family=family, model_id=entry["repo_id"], revision=entry["revision"],
                  ckpt=str(destination), architecture=entry["architecture"],
                  parameter_count_range=entry["parameter_count_range"], load_dtype="bfloat16")
    marker = path / "model_manifest.json"
    if marker.exists():
        if json.loads(marker.read_text()) != fields:
            raise ValueError("model fields changed; use a new destination")
        return
    with tempfile.NamedTemporaryFile(mode="w", dir=path, delete=False) as stream:
        json.dump(fields, stream, indent=2, sort_keys=True)
        temporary = Path(stream.name)
    temporary.replace(marker)


def main(argv=None) -> int:
    args = arguments(argv)
    entry = load_entry(args.config.resolve(), args.model)
    destination = args.destination.resolve() if args.destination else project_path(entry["destination"])
    if args.dry_run:
        print(json.dumps(dict(family=args.family, model=args.model, ckpt=str(destination), **entry), indent=2))
        return 0
    with exclusive_lock(destination):
        if destination.is_dir():
            validate_checkpoint(destination, entry, args.family)
            publish_model_fields(destination, destination, entry, args.family)
            print("Reusing validated checkpoint: %s" % destination)
            return 0
        if destination.exists():
            raise ValueError("checkpoint destination is not a directory: %s" % destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(tempfile.mkdtemp(prefix=destination.name + ".", dir=destination.parent))
        try:
            print("Downloading %s at revision %s" % (entry["repo_id"], entry["revision"]))
            download_checkpoint(staging_dir, entry)
            validate_checkpoint(staging_dir, entry, args.family)
            publish_model_fields(staging_dir, destination, entry, args.family)
            os.replace(staging_dir, destination)
            print("Published validated checkpoint: %s" % destination)
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
