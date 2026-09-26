"""Run or resume a full-pool experiment from an explicit JSON configuration."""
import argparse
import json
from pathlib import Path

from rags.experiments.full_pool_run import run_suite


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True)
    parser.add_argument("--execution-config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return run_suite(
        json.loads(args.config.read_text()), args.out, dry_run=args.dry_run,
        execution=json.loads(args.execution_config.read_text()) if args.execution_config else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
