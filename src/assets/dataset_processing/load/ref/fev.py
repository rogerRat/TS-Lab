"""Reference Load: safely extract fetched benchmark data and immediately write bins."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote

import numpy as np
import yaml

from configs.logger import configure_logger
from configs.paths import config_file, output_path
from assets.dataset_processing.load.bundle import BundleLoadWriter
from assets.dataset_processing.load_reuse import (
    dataset_identity_from_raw,
    make_load_metadata,
    reuse_load_stage,
)
from ts3.spec.files.shard_dataset import DEFAULT_SHARD_BYTES
from ts3.spec.ref_extraction import ReferenceExtractionSpecification

log = logging.getLogger("load.ref")


def _finite_reference_values(values: object) -> np.ndarray:
    """Return the finite univariate observations from a Reference target."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("Reference extractor produced a non-univariate target")
    finite = np.isfinite(array)
    if not finite.all():
        log.warning(
            "REFERENCE VALUE SKIPPED: non_finite_values=%d",
            int((~finite).sum()),
        )
    return array[finite]


def _asset_rows(native_path: Path, asset_id: str):
    from datasets import load_from_disk
    for number, row in enumerate(load_from_disk(str(native_path)).with_format("numpy")):
        values = _finite_reference_values(row["target"])
        yield {
            "dataset": "reference",
            "subset": asset_id,
            "series_id": str(row.get("item_id", row.get("id", number))),
            "values": values,
        }


def _reference_candidates(search_root: Path, asset_id: str):
    root = Path(search_root)
    if not root.is_dir():
        return ()
    result = []
    for protocol_root in sorted(root.iterdir()):
        if not protocol_root.is_dir():
            continue
        for candidate in (
            protocol_root / "fev" / asset_id,
            protocol_root / asset_id,
        ):
            if candidate.is_dir():
                result.append(candidate)
    return tuple(result)


def load_reference(spec_path: Path, fetch_dir: Path | None, bundle_dir: Path,
                   asset_id: str = "all", shard_bytes: int = DEFAULT_SHARD_BYTES,
                   include_future: bool = False,
                   reuse_search_root: Path | None = None) -> tuple[Path, ...]:
    """Use the reference extractor as an in-process Load submodule.

    Its temporary native dataset is deleted before this function returns; the
    persisted Load result is solely each per-asset Bundle's bins and Arrow
    files, one independent bundle directory (`bundle_dir/<asset_id>`) per
    selected asset.
    """
    spec_path = config_file(spec_path)
    spec = ReferenceExtractionSpecification.from_file(spec_path)
    declaration_path = spec_path.parent / spec.mapping["evaluation_declaration"]["path"]
    data_root = output_path("benchmark", fetch_dir) if fetch_dir else output_path("benchmark", spec_path.parent / spec.mapping["upstream"]["data_root"])
    bundle_dir = output_path("reference_shard", bundle_dir)
    declaration = yaml.safe_load(declaration_path.read_text(encoding="utf-8"))
    identity_batch = hashlib.sha256(
        spec_path.read_bytes() + declaration_path.read_bytes()
    ).hexdigest()
    raw_identity = dataset_identity_from_raw(
        data_root, dataset_id="fev-reference", batch_id=identity_batch,
    )
    adapter_name = {"fev": "fev", "gifteval": "gift_eval", "time": "time"}[spec.mapping["benchmark"]]
    adapter = __import__("assets.dataset_processing.extract.ref.%s" % adapter_name, fromlist=["extract_assets"])
    temporary = Path(tempfile.mkdtemp(prefix=".reference-load-"))
    try:
        if include_future and spec.mapping["benchmark"] != "fev":
            raise ValueError("--include-future is currently supported only for FEV")
        if include_future:
            log.warning("REFERENCE DEBUG LOAD INCLUDES NON-SAFE HISTORY; output is forbidden for training and evaluation")
        assets = adapter.extract_assets(spec, declaration_path, data_root, declaration, temporary, _digest, include_future=include_future) if spec.mapping["benchmark"] == "fev" else adapter.extract_assets(spec, declaration_path, data_root, declaration, temporary, _digest)
        selected = {item["asset_id"] for item in assets} if asset_id == "all" else {asset_id}
        unknown = selected - {item["asset_id"] for item in assets}
        if unknown:
            raise ValueError("Unknown reference asset_id: %s" % sorted(unknown))
        produced = []
        for asset in sorted(assets, key=lambda item: item["asset_id"]):
            if asset["asset_id"] not in selected:
                continue
            target_bundle = Path(bundle_dir) / asset["asset_id"]
            load_metadata = make_load_metadata(
                raw_identity,
                scope="reference",
                selection={
                    "asset_id": asset["asset_id"],
                    "spec_batch": identity_batch,
                    "include_future": bool(include_future),
                },
                shard_bytes=shard_bytes,
            )
            reused = False
            if reuse_search_root is not None:
                for candidate in _reference_candidates(reuse_search_root, asset["asset_id"]):
                    if reuse_load_stage(
                        target_bundle, candidate, load_metadata, logger=log,
                    ):
                        reused = True
                        break
            if reused:
                log.info("Reference Load reused bin shards: %s", target_bundle)
                produced.append(target_bundle)
                continue
            writer = BundleLoadWriter(
                target_bundle, scope="reference", shard_bytes=shard_bytes,
                load_metadata=load_metadata,
            )
            writer.write_rows(_asset_rows(temporary / asset["native_output"]["location"], asset["asset_id"]))
            if not writer.rows_written:
                raise ValueError("Reference Load produced no trajectories for asset_id=%s" % asset["asset_id"])
            finalized = writer.finalize()
            (finalized / "load" / "extraction_metadata.json").write_text(json.dumps({"contains_non_safe_history": bool(include_future), "warning": "Non-safe FEV answer history is present; debug/test analysis only.", "training_boundary": asset["training_boundary"]}, sort_keys=True) + "\n", encoding="ascii")
            produced.append(finalized)
        if not produced:
            raise ValueError("Reference Load produced no trajectories")
        return tuple(produced)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Safely extract fetched Reference data into one ShardDatasetBundle")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--fetch-dir", default=None, help="Raw Fetch directory; defaults to the frozen spec data root")
    parser.add_argument("--asset-id", default="all")
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)
    parser.add_argument("--reuse-search-root", default=None, help="Sibling protocol root to scan for compatible reference bin shards")
    parser.add_argument("--include-future", action="store_true", help="FEV only: retain non-safe answer history for debug/test analysis; never use this output in training or evaluation")
    args = parser.parse_args(argv)
    if args.shard_bytes <= 0:
        raise ValueError("--shard-bytes must be positive")
    configure_logger("load.ref", "data/load/ref", "ref", "LOAD/REF")
    produced = load_reference(
        Path(args.spec), Path(args.fetch_dir) if args.fetch_dir else None,
        Path(args.bundle_dir), args.asset_id, args.shard_bytes, args.include_future,
        Path(args.reuse_search_root) if args.reuse_search_root else None,
    )
    for bundle_path in produced:
        logging.getLogger("load.ref").info("Reference Load completed: bundle=%s", bundle_path)
    return 0


__all__ = ["load_reference", "main"]

def _digest(*values: str) -> str:
    """Create the stable asset identifier used by internal extractor adapters."""
    if not values:
        raise ValueError("identifier requires at least one value")
    if values[0] in {"fev", "gifteval", "time"} and len(values) >= 2:
        return "%s-%s" % (values[0], quote(values[1], safe="-_."))
    return "projection-%s" % quote(values[0], safe="-_.")
