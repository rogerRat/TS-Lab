"""Strict contracts for immutable benchmark reference extraction artifacts."""

from __future__ import annotations

import base64
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


SPEC_VERSION = "tsl.reference-extraction-spec.v1"
MANIFEST_VERSION = "tsl.reference-extraction-manifest.v2"
PROVENANCE_VERSION = "tsl.reference-runtime-provenance.v3"
BENCHMARKS = ("gifteval", "fev", "time")
PREFIX_POLICY = "training_common_earliest_test"
NATIVE_FORMAT = "huggingface-dataset-on-disk.v1"
SPEC_FILENAME = "spec.yaml"
MANIFEST_FILENAME = "manifest.json"


def _ascii(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or not value.isascii():
        raise ValueError("%s must be non-empty ASCII" % name)
    return value


def _identifier(value: Any, name: str) -> str:
    return _ascii(value, name)


def _relative(value: Any, name: str) -> str:
    text = _ascii(value, name)
    path = Path(text)
    if path.is_absolute():
        raise ValueError("%s must be a relative path" % name)
    return text


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n").encode("ascii")


@dataclass(frozen=True)
class ReferenceExtractionSpecification:
    mapping: Mapping[str, Any]

    def __post_init__(self) -> None:
        value = dict(self.mapping)
        required = {"schema_version", "benchmark", "evaluation_declaration", "upstream", "task_selection", "prefix_policy", "native_format"}
        if set(value) != required or value["schema_version"] != SPEC_VERSION:
            raise ValueError("reference extraction specification fields are invalid")
        if value["benchmark"] not in BENCHMARKS or value["prefix_policy"] != PREFIX_POLICY or value["native_format"] != NATIVE_FORMAT:
            raise ValueError("reference extraction specification has unsupported values")
        declaration = value["evaluation_declaration"]
        if not isinstance(declaration, Mapping) or set(declaration) != {"path"}:
            raise ValueError("evaluation_declaration fields are invalid")
        _relative(declaration["path"], "evaluation_declaration.path")
        upstream = value["upstream"]
        if not isinstance(upstream, Mapping) or set(upstream) != {"provider", "code_commit", "data_revision", "data_root"}:
            raise ValueError("upstream fields are invalid")
        _ascii(upstream["provider"], "upstream.provider"); _ascii(upstream["data_revision"], "upstream.data_revision"); _relative(upstream["data_root"], "upstream.data_root")
        if upstream["code_commit"] is not None:
            if not isinstance(upstream["code_commit"], str) or not re.fullmatch(r"[0-9a-f]{40}", upstream["code_commit"]):
                raise ValueError("upstream.code_commit is invalid")
        tasks = value["task_selection"]
        if not isinstance(tasks, Mapping) or set(tasks) != {"mode", "items"} or tasks["mode"] not in {"all", "explicit"} or not isinstance(tasks["items"], list):
            raise ValueError("task_selection is invalid")
        if tasks["mode"] == "explicit" and (not tasks["items"] or len(tasks["items"]) != len(set(tasks["items"]))):
            raise ValueError("explicit task_selection must be non-empty and unique")
        if any(_ascii(item, "task_selection item") != item for item in tasks["items"]):
            raise ValueError("task_selection item is invalid")
        object.__setattr__(self, "mapping", value)

    @property
    def spec_id(self) -> str:
        return "spec-" + base64.urlsafe_b64encode(canonical_json(self.mapping)).decode("ascii").rstrip("=")

    @classmethod
    def from_file(cls, path: Path) -> "ReferenceExtractionSpecification":
        source = Path(path)
        text = source.read_text(encoding="ascii")
        suffix = source.suffix.lower()
        if suffix == ".json":
            value = json.loads(text)
        elif suffix in {".yaml", ".yml"}:
            value = yaml.safe_load(text)
        else:
            raise ValueError("reference extraction specification must use .json, .yaml, or .yml")
        return cls(value)


def validate_manifest(value: Any, base_dir: Path | None = None) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("reference extraction manifest must be a mapping")
    required = {"schema_version", "spec_id", "benchmark", "prefix_policy", "assets"}
    if set(value) != required or value["schema_version"] != MANIFEST_VERSION or value["benchmark"] not in BENCHMARKS or value["prefix_policy"] != PREFIX_POLICY:
        raise ValueError("reference extraction manifest fields are invalid")
    _identifier(value["spec_id"], "spec_id")
    assets = value["assets"]
    if not isinstance(assets, list) or not assets:
        raise ValueError("reference extraction manifest assets are invalid")
    seen = set()
    for item in assets:
        fields = {"asset_id", "asset_identity", "series_projection_id", "task_ids", "earliest_test_boundary", "training_boundary", "native_output", "source", "trajectory_count", "value_count"}
        if not isinstance(item, Mapping) or set(item) != fields:
            raise ValueError("reference extraction asset fields are invalid")
        asset_id = _identifier(item["asset_id"], "asset_id")
        if asset_id in seen:
            raise ValueError("reference extraction asset_id is duplicated")
        seen.add(asset_id); _ascii(item["asset_identity"], "asset_identity"); _identifier(item["series_projection_id"], "series_projection_id")
        if not isinstance(item["task_ids"], list) or not item["task_ids"] or item["task_ids"] != sorted(set(item["task_ids"])):
            raise ValueError("reference extraction task_ids must be sorted and unique")
        for task_id in item["task_ids"]: _ascii(task_id, "task_id")
        for name in ("earliest_test_boundary", "training_boundary"):
            boundary = item[name]
            expected = {"kind", "value", "provider_evidence"} if name == "earliest_test_boundary" else {"kind", "value"}
            if not isinstance(boundary, Mapping) or set(boundary) != expected or boundary["kind"] not in {"position", "timestamp"}:
                raise ValueError("%s is invalid" % name)
            if boundary["kind"] == "position" and (isinstance(boundary["value"], bool) or not isinstance(boundary["value"], int) or boundary["value"] < 0):
                raise ValueError("%s position is invalid" % name)
            if boundary["kind"] == "timestamp": _ascii(boundary["value"], "%s value" % name)
            if name == "earliest_test_boundary": _ascii(boundary["provider_evidence"], "provider_evidence")
        native = item["native_output"]
        if not isinstance(native, Mapping) or set(native) != {"location", "format"} or native["format"] != NATIVE_FORMAT:
            raise ValueError("native_output is invalid")
        _relative(native["location"], "native_output.location")
        source = item["source"]
        if not isinstance(source, Mapping) or set(source) != {"identity", "revision"}:
            raise ValueError("source is invalid")
        _ascii(source["identity"], "source.identity"); _ascii(source["revision"], "source.revision")
        for name in ("trajectory_count", "value_count"):
            if isinstance(item[name], bool) or not isinstance(item[name], int) or item[name] < 0: raise ValueError("%s is invalid" % name)
        if base_dir is not None and not (Path(base_dir) / native["location"]).is_dir(): raise ValueError("native output is missing")
    return dict(value)


def load_manifest(path: Path) -> Mapping[str, Any]:
    manifest_path = Path(path).resolve()
    return validate_manifest(json.loads(manifest_path.read_text(encoding="ascii")), manifest_path.parent)


def resolve_manifest_path(ex_dir: str | Path | None, manifest: str | Path | None) -> Path:
    """Resolve an explicit manifest or the fixed contract manifest in an Extract output directory."""
    if manifest is not None:
        manifest_path = Path(manifest).resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError("explicit --manifest must name an existing file: %s" % manifest_path)
        return manifest_path
    extraction_dir = Path(ex_dir).resolve()
    if not extraction_dir.is_dir():
        raise NotADirectoryError("--ex-dir must name an existing directory: %s" % extraction_dir)
    manifest_path = extraction_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError("contract manifest is missing: %s" % manifest_path)
    return manifest_path


def write_manifest(path: Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path)
    validate_manifest(value)
    if destination.exists(): raise FileExistsError("refusing to overwrite reference extraction manifest")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", prefix=".manifest-", dir=destination.parent, delete=False) as stream:
        stream.write(canonical_json(dict(value))); stream.flush(); os.fsync(stream.fileno()); temporary = Path(stream.name)
    os.replace(temporary, destination)
    return destination


def reference_runtime_provenance(manifest_path: Path, asset_id: str) -> Mapping[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_path)
    asset_ids = sorted(item["asset_id"] for item in manifest["assets"])
    if asset_id not in asset_ids:
        raise ValueError("asset_id is absent from reference extraction manifest")
    return {"reference": {"schema_version": PROVENANCE_VERSION, "manifest_location": str(manifest_path), "spec_id": manifest["spec_id"], "benchmark": manifest["benchmark"], "asset_id": _identifier(asset_id, "asset_id"), "asset_ids": asset_ids, "role": "reference"}}


def validate_reference_runtime_provenance(build_config: Any, manifest_path: Path | None = None) -> Mapping[str, Any]:
    if not isinstance(build_config, Mapping) or set(build_config) != {"preserve-invalid-values", "protocol_version", "reference", "shard_value_budget"}:
        raise ValueError("reference runtime build_config fields are invalid")
    value = build_config["reference"]
    required = {"schema_version", "manifest_location", "spec_id", "benchmark", "asset_id", "asset_ids", "role"}
    if not isinstance(value, Mapping) or set(value) != required or value["schema_version"] != PROVENANCE_VERSION or value["role"] != "reference":
        raise ValueError("reference runtime provenance is invalid")
    _ascii(value["manifest_location"], "manifest_location"); _identifier(value["spec_id"], "spec_id"); _ascii(value["benchmark"], "benchmark"); _identifier(value["asset_id"], "asset_id")
    if not isinstance(value["asset_ids"], list) or not value["asset_ids"] or value["asset_ids"] != sorted(set(value["asset_ids"])):
        raise ValueError("reference runtime provenance asset_ids is invalid")
    for item in value["asset_ids"]:
        _identifier(item, "reference runtime provenance asset_id")
    if value["asset_id"] not in value["asset_ids"]:
        raise ValueError("reference runtime provenance asset_id is absent from asset_ids")
    if manifest_path is not None:
        resolved = Path(manifest_path).resolve()
        if Path(value["manifest_location"]).resolve() != resolved:
            raise ValueError("reference runtime provenance manifest mismatch")
    return dict(value)


__all__ = ["BENCHMARKS", "MANIFEST_FILENAME", "MANIFEST_VERSION", "NATIVE_FORMAT", "PREFIX_POLICY", "PROVENANCE_VERSION", "SPEC_FILENAME", "ReferenceExtractionSpecification", "canonical_json", "load_manifest", "reference_runtime_provenance", "resolve_manifest_path", "validate_manifest", "validate_reference_runtime_provenance", "write_manifest"]
