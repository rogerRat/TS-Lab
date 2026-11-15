"""Shared task and method registry for v1-rage-ads-v2."""
from __future__ import annotations

import argparse

TASKS = ("fev-09", "time-01", "time-02", "time-03", "time-04", "time-06", "time-07", "time-08")
ARMS = ("ads-vic", "uni", "traj", "orig", "reft")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-tasks", action="store_true")
    parser.add_argument("--print-arms", action="store_true")
    parser.add_argument("--print-cells", action="store_true")
    args = parser.parse_args()
    if sum((args.print_tasks, args.print_arms, args.print_cells)) != 1:
        parser.error("select exactly one output option")
    if args.print_tasks:
        print("\n".join(TASKS))
    elif args.print_arms:
        print("\n".join(ARMS))
    else:
        print(len(TASKS) * len(ARMS))


if __name__ == "__main__":
    main()
