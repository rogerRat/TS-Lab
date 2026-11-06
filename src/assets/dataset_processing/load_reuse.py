"""Safe Load-bin reuse for protocol-specific ShardDataset bundles.

A protocol may change its Arrow index/metadata or downstream Feature stage while
reading exactly the same immutable raw dataset. This module makes that reuse
explicit: identity metadata is persisted in every new Load bundle, candidates
are compared before use, and only load/shards is symlinked. The target keeps
its own index.arrow and meta.arrow.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

from tsl.spec.files.shard_dataset import (
    FORMAT_VERSION,
    LOAD_METADATA_FILENAME,
    LOAD_METADATA_SCHEMA_VERSION,
    validate_shard_dataset_bundle,
)
_IDENTITY_METADATA_NAMES = frozenset(
    {"profile_manifest.json", "fetch_manifest.json", "metadata.json"}
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("ascii")


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _profile_identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    stable = dict(payload)
    stable.pop("status", None)
    stable.pop("exit_codes", None)
    return stable


def dataset_identity_from_raw(
    raw_root: Path,
    *,
    dataset_id: str | None = None,
    batch_id: str | None = None,
) -> dict[str, str]:
    """Build a stable identity from raw-dataset metadata and file inventory.

    A profile manifest supplies the human-readable dataset and batch identity.
    For raw roots without that manifest, callers must provide both identifiers.
    Metadata and marker files are hashed; data files contribute relative names
    and sizes, avoiding a multi-gigabyte hashing pass.
    """
    root = Path(raw_root).resolve()
    if not root.is_dir():
        raise ValueError("raw dataset root does not exist: %s" % root)

    manifest = root / "profile_manifest.json"
    profile_payload: dict[str, Any] | None = None
    if manifest.is_file():
        try:
            loaded = json.loads(manifest.read_text(encoding="ascii"))
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError("raw profile manifest is invalid: %s" % manifest) from exc
        if not isinstance(loaded, dict):
            raise ValueError("raw profile manifest must be an object: %s" % manifest)
        profile_payload = _profile_identity_payload(loaded)
        dataset_id = str(loaded.get("profile") or dataset_id or "")
        batch_id = str(loaded.get("batch_id") or batch_id or dataset_id)
    if not dataset_id or not batch_id:
        raise ValueError(
            "raw dataset identity is unavailable; provide dataset_id and batch_id: %s"
            % root
        )

    digest = hashlib.sha256()
    if profile_payload is not None:
        digest.update(b"profile_manifest\0")
        digest.update(_canonical_json(profile_payload))
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        if path == manifest and profile_payload is not None:
            digest.update(_canonical_json(profile_payload))
        elif (
            path.name in _IDENTITY_METADATA_NAMES
            or path.name == ".complete"
            or path.suffix == ".json"
        ):
            digest.update(_file_digest(path).encode("ascii"))
        else:
            digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")

    return {
        "dataset_id": str(dataset_id),
        "batch_id": str(batch_id),
        "raw_fingerprint": digest.hexdigest(),
    }


def make_load_metadata(
    dataset_identity: Mapping[str, Any],
    *,
    scope: str,
    selection: Mapping[str, Any],
    shard_bytes: int,
) -> dict[str, Any]:
    """Create the persisted identity contract for one Load stage."""
    if scope not in {"source", "reference"}:
        raise ValueError("scope must be source or reference")
    if not isinstance(shard_bytes, int) or isinstance(shard_bytes, bool) or shard_bytes < 8:
        raise ValueError("shard_bytes must be an integer of at least eight")
    identity = dict(dataset_identity)
    required = {"dataset_id", "batch_id", "raw_fingerprint"}
    if set(identity) != required or any(
        not isinstance(identity[key], str) or not identity[key] for key in required
    ):
        raise ValueError(
            "dataset_identity must contain non-empty dataset_id, batch_id, raw_fingerprint"
        )
    return {
        "schema_version": LOAD_METADATA_SCHEMA_VERSION,
        "format_version": FORMAT_VERSION,
        "scope": scope,
        "dataset_identity": identity,
        "selection": dict(selection),
        "shard_bytes": int(shard_bytes),
    }


def write_load_metadata(bundle_root: Path, metadata: Mapping[str, Any]) -> Path:
    """Atomically write the Load identity file below bundle_root/load."""
    checked = dict(metadata)
    if checked.get("schema_version") != LOAD_METADATA_SCHEMA_VERSION:
        raise ValueError("unsupported Load metadata schema")
    target = Path(bundle_root) / "load" / LOAD_METADATA_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    partial.write_text(
        json.dumps(checked, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="ascii",
    )
    partial.replace(target)
    return target


def read_load_metadata(bundle_root: Path) -> dict[str, Any] | None:
    """Read the identity contract, returning None for legacy bundles."""
    target = Path(bundle_root) / "load" / LOAD_METADATA_FILENAME
    if not target.is_file():
        return None
    try:
        payload = json.loads(target.read_text(encoding="ascii"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("Load metadata is invalid: %s" % target) from exc
    if not isinstance(payload, dict):
        raise ValueError("Load metadata must be an object: %s" % target)
    required = {
        "schema_version", "format_version", "scope", "dataset_identity",
        "selection", "shard_bytes",
    }
    if set(payload) - required - {"reuse"} or not required.issubset(payload):
        raise ValueError("Load metadata fields are invalid: %s" % target)
    if payload["schema_version"] != LOAD_METADATA_SCHEMA_VERSION:
        raise ValueError("unsupported Load metadata schema: %s" % target)
    if payload["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported Load format in metadata: %s" % target)
    if payload["scope"] not in {"source", "reference"}:
        raise ValueError("invalid Load scope in metadata: %s" % target)
    if not isinstance(payload["dataset_identity"], dict):
        raise ValueError("invalid dataset identity in metadata: %s" % target)
    if not isinstance(payload["selection"], dict):
        raise ValueError("invalid Load selection in metadata: %s" % target)
    return payload


def _mismatch(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> str | None:
    for field in (
        "format_version", "scope", "dataset_identity", "selection", "shard_bytes",
    ):
        if actual.get(field) != expected.get(field):
            return field
    return None


def _log(logger: logging.Logger | None, level: int, message: str, *args: object) -> None:
    (logger or logging.getLogger("dataset_processing.reuse")).log(level, message, *args)


def reuse_load_stage(
    target_bundle: Path,
    candidate_bundle: Path,
    expected_metadata: Mapping[str, Any],
    *,
    logger: logging.Logger | None = None,
) -> bool:
    """Link a compatible candidate bin pool into a new target bundle."""
    target = Path(target_bundle).resolve()
    candidate = Path(candidate_bundle).resolve()
    _log(
        logger, logging.INFO,
        "BIN SHARD REUSE CHECK: target=%s candidate=%s", target, candidate,
    )
    if target.exists():
        _log(
            logger, logging.INFO,
            "BIN SHARD REUSE FALLBACK: target already exists: %s", target,
        )
        return False
    if not candidate.is_dir():
        _log(
            logger, logging.INFO,
            "BIN SHARD REUSE SKIP: candidate is not a bundle: %s", candidate,
        )
        return False
    try:
        actual = read_load_metadata(candidate)
    except ValueError as exc:
        _log(
            logger, logging.WARNING,
            "BIN SHARD REUSE FALLBACK: invalid candidate metadata=%s error=%s",
            candidate, exc,
        )
        return False
    if actual is None:
        _log(
            logger, logging.INFO,
            "BIN SHARD REUSE FALLBACK: candidate has no identity metadata: %s",
            candidate,
        )
        return False
    mismatch = _mismatch(expected_metadata, actual)
    if mismatch is not None:
        _log(
            logger, logging.INFO,
            "BIN SHARD REUSE FALLBACK: metadata mismatch field=%s candidate=%s",
            mismatch, candidate,
        )
        return False

    source_shards = candidate / "load" / "shards"
    if not source_shards.is_dir():
        _log(
            logger, logging.WARNING,
            "BIN SHARD REUSE FALLBACK: candidate shards are missing: %s",
            source_shards,
        )
        return False
    try:
        validate_shard_dataset_bundle(candidate, include_feature=False)
    except (OSError, ValueError) as exc:
        _log(
            logger, logging.WARNING,
            "BIN SHARD REUSE FALLBACK: candidate validation failed=%s error=%s",
            candidate, exc,
        )
        return False

    target_load = target / "load"
    target_load.mkdir(parents=True)
    for name in ("index.arrow", "meta.arrow"):
        shutil.copy2(candidate / "load" / name, target_load / name)
    for item in (candidate / "load").iterdir():
        if item.name in {"index.arrow", "meta.arrow", LOAD_METADATA_FILENAME, "shards"}:
            continue
        if item.is_file():
            shutil.copy2(item, target_load / item.name)

    relative_source = os.path.relpath(source_shards.resolve(), target_load)
    (target_load / "shards").symlink_to(relative_source, target_is_directory=True)
    persisted = copy.deepcopy(dict(expected_metadata))
    persisted["reuse"] = {
        "mode": "symlink",
        "source_bundle": str(candidate),
    }
    write_load_metadata(target, persisted)
    try:
        validate_shard_dataset_bundle(target, include_feature=False)
    except (OSError, ValueError):
        shutil.rmtree(target)
        raise
    _log(
        logger, logging.INFO,
        "BIN SHARD REUSE: linked target=%s/load/shards -> %s; kept target index/meta",
        target, source_shards.resolve(),
    )
    return True


def find_and_reuse_load_stage(
    target_bundle: Path,
    search_root: Path,
    expected_metadata: Mapping[str, Any],
    *,
    logger: logging.Logger | None = None,
    candidates: Iterable[Path] | None = None,
) -> bool:
    """Try compatible sibling bundles, then report an explicit fallback."""
    target = Path(target_bundle).resolve()
    if candidates is None:
        root = Path(search_root).resolve()
        candidate_list = (
            sorted(
                item for item in root.iterdir()
                if item.is_dir() and item.resolve() != target
            )
            if root.is_dir() else []
        )
    else:
        candidate_list = [
            Path(item).resolve() for item in candidates
            if Path(item).resolve() != target
        ]
    for candidate in candidate_list:
        if reuse_load_stage(target, candidate, expected_metadata, logger=logger):
            return True
    _log(
        logger, logging.INFO,
        "BIN SHARD REUSE FALLBACK: no compatible candidate; rebuilding bins from raw input target=%s",
        target,
    )
    return False


__all__ = [
    "LOAD_METADATA_FILENAME",
    "LOAD_METADATA_SCHEMA_VERSION",
    "dataset_identity_from_raw",
    "find_and_reuse_load_stage",
    "make_load_metadata",
    "read_load_metadata",
    "reuse_load_stage",
    "write_load_metadata",
]
