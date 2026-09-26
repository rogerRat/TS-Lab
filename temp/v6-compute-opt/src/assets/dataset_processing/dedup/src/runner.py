"""ShardDatasetBundle Dedup CLI."""
import argparse
import logging
from pathlib import Path
from configs.logger import configure_logger
from .bundle import deduplicate_bundle
def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--bundle",required=True); p.add_argument("--scope",action="append"); a=p.parse_args(argv)
    configure_logger("assets.dataset_processing.dedup", "data/dedup", "dedup", "DEDUP")
    logging.getLogger("assets.dataset_processing.dedup").info("CLI arguments: bundle=%s scopes=%s", a.bundle, a.scope or "all")
    deduplicate_bundle(Path(a.bundle), set(a.scope) if a.scope else None)
    logging.getLogger("assets.dataset_processing.dedup").info("Dedup completed: output=%s/dedup/dedup.arrow", Path(a.bundle).resolve())
    return 0
