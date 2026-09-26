"""DEPRECATED manual merger for resumable per-config Parquet shards.

@input: dataset output directory containing a fetch manifest and Parquet shards.
@output: 0 on success, 1 when the shard set is incomplete or merge fails.
@scenario: Manual export for consumers that explicitly require one Parquet file.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from configs.logger import configure_logger

from assets.dataset_processing.series import (
    is_complete_parquet_shard,
    merge_parquet_shards,
)


log = logging.getLogger("fetch")


def _load_fetch_shard_manifest(output_dir: Path) -> dict:
    """Load the expected-config fetch shard manifest.

    @input: output_dir - Dataset output directory.
    @output: dict - JSON fetch shard manifest with final_name and configs fields.
    @scenario: Makes merge input explicit and prevents incomplete final files.
    """
    manifest_path = output_dir / "fetch_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing fetch manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        fetch_shard_manifest = json.load(handle)
    if not isinstance(fetch_shard_manifest.get("configs"), list) or not fetch_shard_manifest["configs"]:
        raise ValueError(f"Invalid fetch manifest: {manifest_path}")
    return fetch_shard_manifest


def merge_downloaded_shards(
    output_dir: Path, force: bool = False, delete_sources: bool = False
) -> int:
    """Merge all complete expected shards for an explicit manual request.

    @input: output_dir - Dataset directory with fetch_manifest.json and shards.
             force - Allow replacing an existing final Parquet file.
             delete_sources - Delete verified source shards after merging.
    @output: int - Number of rows in the final file.
    @scenario: Produces a legacy single-file export; fetch and load do not call this.
    """
    output_dir = Path(output_dir)
    fetch_shard_manifest = _load_fetch_shard_manifest(output_dir)
    final_path = output_dir / fetch_shard_manifest["final_name"]
    if final_path.exists() and not force:
        log.info("Final dataset already exists at %s -- skipping merge.", final_path)
        return 0

    shard_paths = []
    missing = []
    for config_name in fetch_shard_manifest["configs"]:
        shard = output_dir / fetch_shard_manifest["shards"][config_name]
        if is_complete_parquet_shard(shard):
            shard_paths.append(shard)
        else:
            chunks = sorted(shard.parent.glob(f"{shard.stem}_chunk_*.parquet"))
            if chunks and all(is_complete_parquet_shard(chunk) for chunk in chunks):
                shard_paths.extend(chunks)
            else:
                missing.append(config_name)
    if missing:
        log.warning("Merge postponed; incomplete configs: %s", ", ".join(missing))
        return 1

    rows_written = merge_parquet_shards(shard_paths, final_path)
    if rows_written <= 0:
        raise ValueError("Merged dataset contains no rows")

    if delete_sources:
        for shard in shard_paths:
            shard.unlink()
            marker = Path(str(shard) + ".complete")
            if marker.exists():
                marker.unlink()
        log.info(
            "Merged %d shards into %s (%d rows); removed verified source shards.",
            len(shard_paths), final_path, rows_written,
        )
    else:
        log.info(
            "Merged %d shards into %s (%d rows); source shards retained.",
            len(shard_paths), final_path, rows_written,
        )
    return rows_written


def parse_args(argv=None):
    """Parse the manual merge command arguments.

    @input: argv - Optional argument list.
    @output: argparse.Namespace - Validated command options.
    @scenario: Exposes merge finalization for manual recovery operations.
    """
    parser = argparse.ArgumentParser(description="Merge completed dataset fetch shards")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--del-src", action="store_true",
        help="Delete verified source shards after a successful merge.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Run the deprecated manual merge command.

    @input: argv - Optional command-line arguments.
    @output: int - Process exit status.
    @scenario: Allows operators to retry only the final merge step.
    """
    configure_logger("fetch.merge", "data/fetch", "fetch_merge", "FETCH/MERGE")
    log.warning("DEPRECATED: fetch and load use subsets/ shards directly; run this only for a required legacy export.")
    args = parse_args(argv)
    try:
        from configs.paths import output_path
        merge_downloaded_shards(
            output_path("source_raw", args.output_dir),
            force=args.force,
            delete_sources=args.del_src,
        )
    except Exception as exc:
        log.error("Shard merge failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
