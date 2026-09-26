"""Download Benchmark data declared by a frozen Reference extraction spec."""

from __future__ import annotations

import argparse
import logging
from importlib import import_module
from pathlib import Path

from ts3.spec.ref_extraction import ReferenceExtractionSpecification
from configs.paths import config_file, output_path

log = logging.getLogger("fetch.ref")


def fetch(spec_path: Path, output_dir: Path | None = None) -> Path:
    """Populate an explicit or spec-owned local data root through its Benchmark adapter."""
    spec_path = config_file(spec_path)
    spec = ReferenceExtractionSpecification.from_file(spec_path)
    data_root = (
        output_path("benchmark", output_dir)
        if output_dir is not None
        else output_path("benchmark", spec_path.parent / spec.mapping["upstream"]["data_root"])
    )
    declaration_path = (spec_path.parent / spec.mapping["evaluation_declaration"]["path"]).resolve()
    adapter_name = {"fev": "fev", "gifteval": "gift_eval", "time": "time"}[spec.mapping["benchmark"]]
    log.info("Reference fetch started: spec=%s benchmark=%s adapter=%s output_dir=%s", spec_path, spec.mapping["benchmark"], adapter_name, data_root)
    adapter = import_module("assets.dataset_processing.fetch.ref.%s" % adapter_name)
    adapter.fetch(spec, declaration_path, data_root)
    log.info("Reference fetch completed: benchmark=%s output_dir=%s", spec.mapping["benchmark"], data_root)
    return data_root


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Download Benchmark data for Reference extraction")
    parser.add_argument("--spec", required=True, help="Frozen Reference extraction spec")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Benchmark data output directory (default: upstream.data_root in --spec)",
    )
    args = parser.parse_args(argv)
    from configs.logger import configure_logger
    configure_logger("fetch.ref", "data/fetch/ref", "fetch_ref", "FETCH/REF")
    log.info("CLI arguments: spec=%s output_dir=%s", args.spec, args.output_dir or "<spec default>")
    try:
        fetch(Path(args.spec), args.output_dir)
    except Exception:
        log.exception("Reference fetch failed")
        raise
    return 0


__all__ = ["fetch", "main"]
