"""Thin CLI router; experiment implementations live in src/rage."""
from __future__ import annotations

import argparse
from importlib import import_module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", choices=("run", "main_table", "curves"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    return import_module("rage.experiments." + args.workflow).main(args.arguments)


if __name__ == "__main__":
    raise SystemExit(main())
