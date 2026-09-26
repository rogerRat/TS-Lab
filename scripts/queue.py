#!/usr/bin/env python3
"""Standard-library task queue CLI; usable without installing the project."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

if __name__ == '__main__':
    from runtime.task_queue import main
    raise SystemExit(main())
