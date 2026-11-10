#!/usr/bin/env python
"""Remove evaluated intermediate adapters after they are no longer needed."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
import time

from configs.logger import configure_logger

ROOT = Path(__file__).resolve().parents[2]
BATCHES = ("v1-rage-chronosT5-trev-base-rand", "v1-rage-chronosT5-trev-base-full")
STEPS = (100, 200, 500, 1000, 2000, 3000, 4000)


def evaluated(batch: str, step: int) -> bool:
    root = ROOT / "outputs/eval" / batch / ("step-%04d" % step)
    fev = root / "fev/fev_results.csv"
    time_meta = root / "time/time_metadata.json"
    if not fev.is_file() or not time_meta.is_file():
        return False
    try:
        time_values = json.loads(time_meta.read_text())
        return len(fev.read_text().splitlines()) == 5 and bool(time_values.get("completed_at"))
    except (OSError, ValueError):
        return False


def main() -> None:
    logger = configure_logger("v1_trev_cleanup", "v1-rage-chronosT5-trev-base",
                              "checkpoint_cleanup", "waiting_for_evaluations")
    while True:
        all_complete = True
        for batch in BATCHES:
            train_root = ROOT / "outputs/train" / batch
            checkpoint_root = train_root / "checkpoints"
            if not (train_root / ".complete").exists():
                all_complete = False
            for step in STEPS:
                adapter = checkpoint_root / ("adapter-step-%04d" % step)
                if step < 4000 and adapter.exists() and evaluated(batch, step):
                    shutil.rmtree(adapter)
                    logger.info("stage=checkpoint-cleaned batch=%s step=%d", batch, step)
            if (train_root / ".complete").exists():
                resume = checkpoint_root / "resume.pt"
                if resume.exists():
                    resume.unlink()
                    logger.info("stage=optimizer-state-cleaned batch=%s", batch)
                base = checkpoint_root / "base-step-0000"
                if base.exists():
                    shutil.rmtree(base)
                    logger.info("stage=base-wrapper-cleaned batch=%s", batch)
        if all_complete:
            logger.info("stage=cleanup-complete final 4,000-step adapters retained")
            return
        time.sleep(20)


if __name__ == "__main__":
    main()
