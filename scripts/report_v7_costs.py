"""Attach zero-compute resource accounting to an existing V7 run/report."""
import argparse
import json
from pathlib import Path
from rage.experiments.v7_costs import collect, render, atomic_text, update_markdown, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/rage/main_v7_curve4.json')
    parser.add_argument('--run')
    parser.add_argument('--markdown', help='Update only the delimited cost block in this file')
    args = parser.parse_args()
    config = read(args.config)
    run = Path(args.run or config['run_root'])
    report = collect(config, run)
    target = run/'resource_costs'
    atomic_text(target/'report.json', json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    block = render(report)
    atomic_text(target/'report.md', block)
    if args.markdown:
        update_markdown(args.markdown, block)
    print(json.dumps(dict(rows=len(report['rows']), assets=len(report['shared_assets']), output=str(target))))


if __name__ == '__main__':
    main()
