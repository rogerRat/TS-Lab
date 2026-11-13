"""Prepare a fixed development-series subset for equal-budget temperature trials."""
import argparse
from pathlib import Path
import shutil

import numpy as np

from rage.experiments.development_inputs import load_fragments, save_fragments
from rage.experiments.students import atomic_json


def prepare(inputs, out, datasets, *, series=8, seed=20260914):
    inputs, out = Path(inputs), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("pool.npz", "pool_manifest.json"):
        shutil.copyfile(inputs / name, out / name)
    rng = np.random.default_rng(seed)
    selected = {}
    for dataset in datasets:
        fragments = load_fragments(inputs / f"{dataset}__reference.npz")
        rows = sorted(rng.choice(len(fragments), size=series, replace=False).tolist())
        chosen = [fragments[row] for row in rows]
        save_fragments(out / f"{dataset}__reference.npz", chosen)
        selected[dataset] = [dict(row=row, entity=f.entity_id, fragment_id=f.fragment_id,
                                  length=len(f.data)) for row, f in zip(rows, chosen)]
    atomic_json(out / "development_subset.json", dict(
        purpose="temperature_development_only_not_formal_series_coverage",
        inputs=str(inputs.resolve()), seed=seed, series=series, selected=selected,
        source_pool_unchanged=True, formal_exam_copied=False))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--series", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    print(prepare(args.inputs, args.out, args.datasets, series=args.series, seed=args.seed), flush=True)


if __name__ == "__main__":
    main()
