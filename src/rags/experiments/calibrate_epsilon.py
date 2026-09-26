"""Calibrate the detection thresholds epsilon_mu = epsilon_g (domain C3).

Rule (frozen in the experiment plan): take every real window gradient saved
by the mechanism-1 full scan on the development task ETT_1H (reference and
source sides), pool them, and use the 10th percentile of the sketch norms.
Count-Sketch norms are unbiased estimates of the true gradient norms at any
projection width, so a value calibrated at d=1024 transfers to the d=4096
main configuration.  Calibrate once per model, then freeze for the whole
suite; sensitivity scans at 0.5x / 1x / 2x are separate.

Usage:
    python -m rags.experiments.calibrate_epsilon --scan-dir <mechanism1 out>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def scan_norms(scan_dir: Path) -> np.ndarray:
    norms = []
    for side in ("reference", "source"):
        path = Path(scan_dir) / ("scan_%s.npz" % side)
        with np.load(path) as payload:
            for key in payload.files:
                if key.startswith("gradients_"):
                    matrix = payload[key].astype(np.float32)
                    norms.append(np.linalg.norm(matrix, axis=1))
    if not norms:
        raise SystemExit("no gradients found under %s" % scan_dir)
    return np.concatenate(norms)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", required=True,
                        help="mechanism-1 output dir holding scan_*.npz")
    parser.add_argument("--percentile", type=float, choices=[10.0], default=10.0)
    parser.add_argument("--out", default=None, help="optional json destination")
    args = parser.parse_args(argv)

    norms = scan_norms(Path(args.scan_dir))
    report = {
        "scan_dir": str(args.scan_dir),
        "count": int(len(norms)),
        "percentile": args.percentile,
        "epsilon": float(np.percentile(norms, args.percentile)),
        "distribution": {str(p): float(np.percentile(norms, p))
                         for p in (1, 5, 10, 25, 50, 75, 90, 99)},
        "rule": ("p{p:.0f} over both sides' real window-gradient sketch norms; "
                 "epsilon_mu = epsilon_g = this value (C3)".format(p=args.percentile)),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
