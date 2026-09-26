"""Prepare Timer-XL safe-history calibration inputs from the archived 0914 development inputs.

Stage-1 (devhistory): copy archived boomlet reference/manifest/pool files unchanged.
Stage-2 (earlierhistory): truncate each reference by context+8*horizon (1024 points) so
full_pool_exploration.development_split reproduces the archived 0913 earlier-history split —
validation blocks strictly earlier than and disjoint from the stage-1 validation blocks.

Verifies the resulting boundaries against the archived boundaries.json files and writes
an audit note. Never modifies the archive; never touches formal exam arrays.
"""
import json
import shutil
from pathlib import Path

import numpy as np

from rags.experiments.development_inputs import load_fragments, save_fragments
from rags.experiments.full_pool_exploration import development_split

ARCHIVE = Path("/root/autodl-tmp/TS3/archive/outputs-cleanup-20260915")
DEV_SRC = ARCHIVE / "full-pool-exp-0914/development-inputs"
OUT = Path("/root/autodl-tmp/TS3/outputs/timerxl-calibration-20260920")
DATASETS = ("boomlet_1676", "boomlet_619")
CONTEXT, HORIZON, VALIDATION_WINDOWS = 512, 64, 8
TRUNCATE = CONTEXT + VALIDATION_WINDOWS * HORIZON  # 1024 points


def main():
    dev = OUT / "inputs-dev"
    early = OUT / "inputs-early"
    for folder in (dev, early):
        if folder.exists() and any(folder.iterdir()):
            raise SystemExit(f"Refusing to overwrite existing inputs: {folder}")
        folder.mkdir(parents=True, exist_ok=True)

    # Stage 1: byte-identical copies of the established devhistory inputs.
    for name in ([f"{d}__reference.npz" for d in DATASETS]
                 + [f"{d}__manifest.json" for d in DATASETS]
                 + ["pool.npz", "pool_manifest.json"]):
        shutil.copy2(DEV_SRC / name, dev / name)

    # Stage 2: strict prefixes, one validation block earlier.
    for dataset in DATASETS:
        fragments = load_fragments(DEV_SRC / f"{dataset}__reference.npz")
        truncated = []
        for fragment in fragments:
            keep = len(fragment.data) - TRUNCATE
            assert keep > CONTEXT + VALIDATION_WINDOWS * HORIZON, fragment.entity_id
            from rags.core import TimeSeriesFragment
            truncated.append(TimeSeriesFragment(fragment.data[:keep].copy(),
                                                entity_id=fragment.entity_id,
                                                fragment_id=fragment.fragment_id))
        save_fragments(early / f"{dataset}__reference.npz", truncated)
        manifest = json.loads((DEV_SRC / f"{dataset}__manifest.json").read_text())
        manifest["reference_lengths"] = [len(f.data) for f in truncated]
        manifest["truncated_for"] = ("earlier independent confirmation split; validation block "
                                     "shifted one block earlier than inputs-dev")
        (early / f"{dataset}__manifest.json").write_text(json.dumps(manifest, indent=1))

    # Verify both splits reproduce the archived 0913/0914 boundaries exactly.
    audit = {"datasets": {}, "archive_source": str(DEV_SRC)}
    expected = {
        "dev": json.loads((ARCHIVE / "calibration-0913-devhistory/validation-inputs/boundaries.json").read_text()),
        "early": json.loads((ARCHIVE / "calibration-0913-earlierhistory/validation-inputs/boundaries.json").read_text()),
    }
    splits = {}
    for tag, folder in (("dev", dev), ("early", early)):
        _, boundaries, history = development_split(folder, DATASETS, context=CONTEXT,
                                                   horizon=HORIZON, count=VALIDATION_WINDOWS)
        splits[tag] = boundaries
        for dataset in DATASETS:
            got = [{k: v for k, v in b.items()} for b in boundaries[dataset]]
            want = expected[tag][dataset]
            assert got == want, f"{tag}/{dataset} boundaries differ from archived 0913/0914 split"
            audit["datasets"].setdefault(dataset, {})[tag] = {
                "series": len(boundaries[dataset]),
                "history_points": int(sum(len(f.data) for f in history
                                          if f.entity_id.startswith(dataset))),
                "reference_stop_exclusive": boundaries[dataset][0]["reference_stop_exclusive"],
                "validation_range": [boundaries[dataset][0]["validation_first_target"],
                                     boundaries[dataset][0]["validation_stop_exclusive"]],
            }
    # Disjointness: early validation must end at or before dev history end (== early ref stop + block).
    for dataset in DATASETS:
        e, d = splits["early"][dataset][0], splits["dev"][dataset][0]
        assert e["validation_stop_exclusive"] <= d["reference_stop_exclusive"], dataset
        audit["datasets"][dataset]["validation_disjoint"] = True
    (OUT / "inputs_audit.json").write_text(json.dumps(audit, indent=1))
    print("TIMERXL_INPUTS_READY", json.dumps(audit["datasets"], indent=1))


if __name__ == "__main__":
    main()
