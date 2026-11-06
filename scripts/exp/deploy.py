"""Check and optionally launch the document-frozen RAGE experiment suite."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import yaml

from configs.paths import PROJECT_ROOT
from rage.experiments.deployment import check_deployment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rage/run.yaml")
    parser.add_argument("--out-root", default="outputs")
    parser.add_argument("--report", default="outputs/deployment-0911.json")
    parser.add_argument("--launch", action="store_true", help="launch only if all prerequisites pass")
    args = parser.parse_args(argv)
    config = yaml.safe_load((PROJECT_ROOT / args.config).read_text())
    report = check_deployment(config, out_root=PROJECT_ROOT / args.out_root)
    destination = PROJECT_ROOT / args.report
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["ready"]:
        return 2
    if args.launch:
        from rage.experiments.run import main as run
        return run(["--config", args.config, "--out-root", args.out_root])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
