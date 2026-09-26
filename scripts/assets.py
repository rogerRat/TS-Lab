"""Resource preparation CLI: datasets, pretrained models and cached gradients."""
from __future__ import annotations

import argparse
from importlib import import_module

WORKFLOWS = {
    "fetch-model": "assets.pretrained_models.fetch",
    "gradient": "assets.gradient.runner",
    "fetch-source": "assets.dataset_processing.fetch.src.runner",
    "fetch-reference": "assets.dataset_processing.fetch.ref.runner",
    "load-source": "assets.dataset_processing.load.src.runner",
    "load-reference": "assets.dataset_processing.load.ref",
    "dedup-source": "assets.dataset_processing.dedup.src.runner",
    "dedup-reference": "assets.dataset_processing.dedup.ref.runner",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=WORKFLOWS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    return import_module(WORKFLOWS[args.stage]).main(args.arguments)


if __name__ == "__main__":
    raise SystemExit(main())
