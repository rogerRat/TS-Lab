"""Load fetched Source data directly into one ShardDatasetBundle."""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

from configs.logger import configure_logger
from configs.paths import output_path
from assets.dataset_processing.configs.config import DEPRECATED_SOURCES
from assets.dataset_processing.fetch.src.sources import FETCH_SOURCE_NAMES
from assets.dataset_processing.load.bundle import BundleLoadWriter
from assets.dataset_processing.load_reuse import (
    dataset_identity_from_raw,
    find_and_reuse_load_stage,
    make_load_metadata,
)
from tsl.spec.files.shard_dataset import DEFAULT_SHARD_BYTES

log = logging.getLogger("load")


def _discover_datasets(input_dir: Path) -> list[str]:
    return sorted(
        path.name for path in input_dir.iterdir()
        if path.is_dir() and path.name in FETCH_SOURCE_NAMES
    )


def _discover_subsets(input_dir: Path, dataset: str) -> list[str]:
    if dataset in {"chronos", "utsd"}:
        return sorted(path.parent.name for path in input_dir.glob("*/.complete"))
    if dataset == "monash":
        return sorted(path.stem for path in (input_dir / ".complete").glob("*.json"))
    if dataset == "ett":
        return sorted(path.stem for path in input_dir.glob("*.csv"))
    values = []
    for path in (input_dir / ".complete").glob("*.json"):
        try:
            values.append(str(json.loads(path.read_text(encoding="ascii"))["subset"]))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(set(values))


def _run_decoder(kind: str, input_dir: Path, subset: str, writer: BundleLoadWriter,
                 max_series: float, seed: int, preserve_invalid_values: bool) -> int:
    if kind == "monash":
        from .monash import load_monash_hf
        return load_monash_hf(input_dir, writer, subset=subset, max_series=max_series,
                              seed=seed, preserve_invalid_values=preserve_invalid_values) or 0
    if kind == "chronos":
        from .chronos import load_chronos
        return load_chronos(input_dir, writer, subset=subset, max_series=max_series,
                            seed=seed, preserve_invalid_values=preserve_invalid_values) or 0
    if kind == "utsd":
        from .utsd import load_utsd
        return load_utsd(input_dir, writer, subset=subset, max_series=max_series,
                         seed=seed, preserve_invalid_values=preserve_invalid_values) or 0
    if kind == "ett":
        from .ett import load_ett
        return load_ett(input_dir, subset, writer,
                        preserve_invalid_values=preserve_invalid_values) or 0
    module = __import__("assets.dataset_processing.load.src.%s" % kind, fromlist=["load"])
    return module.load(input_dir, writer, subset, max_series=max_series, seed=seed,
                       preserve_invalid_values=preserve_invalid_values) or 0


def _max_series(value: str) -> float:
    return float("inf") if value.lower() == "inf" else float(int(value))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Load fetched Source data into one ShardDatasetBundle")
    parser.add_argument("--dataset", default="all", choices=["all", *FETCH_SOURCE_NAMES])
    parser.add_argument("--subset", default="all")
    parser.add_argument("--input-dir", required=True, help="Fetch output root")
    parser.add_argument("--bundle-dir", required=True, help="New ShardDatasetBundle directory")
    parser.add_argument("--max-series", type=_max_series, default=float("inf"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preserve-invalid-values", action="store_true")
    parser.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)
    parser.add_argument(
        "--reuse-search-root", default=None,
        help="Sibling bundle root to scan for compatible Load bin shards",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    configure_logger("load", "data/load", "load", "LOAD")
    input_root = output_path("source_raw", args.input_dir)
    bundle_root = output_path("source_shard", args.bundle_dir)
    if not input_root.is_dir():
        raise FileNotFoundError("Fetch input directory does not exist: %s" % input_root)
    if args.shard_bytes <= 0:
        raise ValueError("--shard-bytes must be positive")
    selection = {
        "dataset": args.dataset,
        "subset": args.subset,
        "max_series": "inf" if math.isinf(args.max_series) else int(args.max_series),
        "seed": int(args.seed),
        "preserve_invalid_values": bool(args.preserve_invalid_values),
    }
    load_metadata = None
    try:
        load_metadata = make_load_metadata(
            dataset_identity_from_raw(input_root),
            scope="source",
            selection=selection,
            shard_bytes=args.shard_bytes,
        )
    except ValueError as exc:
        log.warning(
            "BIN SHARD REUSE FALLBACK: source identity unavailable (%s); "
            "rebuilding bins from raw input",
            exc,
        )
    if load_metadata is not None and args.reuse_search_root:
        if find_and_reuse_load_stage(
            bundle_root, Path(args.reuse_search_root), load_metadata, logger=log,
        ):
            log.info("Source Load completed by bin-shard reuse: %s", bundle_root)
            return 0
    datasets = _discover_datasets(input_root) if args.dataset == "all" else [args.dataset]
    datasets = [item for item in datasets if item not in DEPRECATED_SOURCES]
    if not datasets:
        raise ValueError("No supported fetched Source datasets found")
    writer = BundleLoadWriter(
        bundle_root, scope="source", shard_bytes=args.shard_bytes,
        load_metadata=load_metadata,
    )
    for dataset in datasets:
        dataset_dir = input_root / dataset if (input_root / dataset).is_dir() else input_root
        subsets = _discover_subsets(dataset_dir, dataset) if args.subset == "all" else [args.subset]
        if not subsets:
            log.warning("No fetched subsets found for %s", dataset)
        for subset in subsets:
            count = _run_decoder(dataset, dataset_dir, subset, writer, args.max_series,
                                 args.seed, args.preserve_invalid_values)
            log.info("Loaded %s/%s: %d trajectories", dataset, subset, count)
    if not writer.rows_written:
        raise ValueError("Load produced no trajectories")
    writer.finalize()
    log.info("Source ShardDatasetBundle written: %s trajectories=%d", bundle_root, writer.rows_written)
    return 0
