"""Select a fixed fraction of prepared windows and save an index artifact."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from configs.logger import configure_logger, format_duration
from configs.paths import output_path
from rage.static_selection import selection_size, select_round_robin, select_topk


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="NPZ with windows and ids")
    parser.add_argument("--method", choices=("rr", "topk"), required=True)
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--ratio", type=float)
    budget.add_argument("--count", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-key", default="scores")
    parser.add_argument("--group-key", default="groups")
    args = parser.parse_args(argv)
    log = configure_logger("rage.select_static", "selection", "select_static", "SELECT")
    started = time.monotonic()
    source = np.load(args.input, allow_pickle=False)
    if "windows" not in source or "ids" not in source:
        raise ValueError("input NPZ must contain windows and ids")
    windows, ids = source["windows"], source["ids"]
    if len(windows) != len(ids) or len(windows) == 0:
        raise ValueError("windows and ids must be non-empty and aligned")
    count = selection_size(len(windows), ratio=args.ratio, count=args.count)
    log.info("method=%s candidates=%d selected=%d input=%s", args.method, len(windows), count, args.input)
    if args.method == "topk":
        if args.score_key not in source:
            raise ValueError("TopK input lacks score array: " + args.score_key)
        indices = select_topk(source[args.score_key], count)
    else:
        if args.group_key not in source:
            raise ValueError("RoundRobin input lacks group array: " + args.group_key)
        if args.score_key not in source:
            raise ValueError("RoundRobin input lacks score array: " + args.score_key)
        indices = select_round_robin(source[args.group_key], count, source[args.score_key])
    destination = output_path("selection", args.output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite selection artifact: %s" % destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    protocol = {"method": args.method, "ratio": args.ratio, "count": len(indices),
                "candidate_count": len(windows), "input": str(Path(args.input).resolve())}
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(temporary, indices=indices, windows=windows[indices], ids=ids[indices],
                        protocol=np.asarray(json.dumps(protocol, sort_keys=True)))
    temporary.replace(destination)
    destination.with_suffix(destination.suffix + ".complete").write_text("complete\n", encoding="ascii")
    log.info("selected=%d output=%s elapsed=%s", len(indices), destination,
             format_duration(time.monotonic() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
