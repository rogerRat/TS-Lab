"""CLI only: epoch training/evaluation in a student's isolated environment."""
from __future__ import annotations

import argparse
from pathlib import Path

from rage.experiments.students import STUDENTS, run_student_task


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=STUDENTS, default="timesfm-2.5")
    parser.add_argument("--task-manifest", required=True,
                        help="shared selection/exam and D2 schedule from main_table")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    run_student_task(args.model, Path(args.task_manifest), Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
