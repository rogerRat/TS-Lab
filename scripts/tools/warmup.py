"""Run a short training smoke check for one model and dataset."""
from __future__ import annotations

import sys
from pathlib import Path as _Path

_PROJECT_ROOT = str(_Path(__file__).resolve().parents[2])
for _source_path in (_PROJECT_ROOT, str(_Path(_PROJECT_ROOT) / "src")):
    if _source_path not in sys.path:
        sys.path.insert(0, _source_path)


import argparse
from pathlib import Path

from scripts.tools._common import apply_overrides, dispatch_model_environment, load_config, required, resolve
from scripts.tools.train import run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    config = apply_overrides(load_config(args.config), args.set)
    model = str(required(config, "model"))
    dispatched = dispatch_model_environment(
        model, [str(Path(__file__).resolve()), "--config", str(resolve(args.config)),
                *sum((["--set", value] for value in args.set), [])])
    if dispatched is not None:
        return dispatched
    run(config, stage="warmup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
