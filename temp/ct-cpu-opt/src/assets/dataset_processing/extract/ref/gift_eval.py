"""GIFT-Eval reference adapter: retain the shared pre-evaluation prefix."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Mapping

from ts3.spec.ref_extraction import ReferenceExtractionSpecification


def _save_records(records, destination: Path) -> tuple[int, int]:
    from datasets import Dataset

    rows = list(records)
    if not rows:
        raise ValueError("GIFT-Eval extraction produced no trajectories")
    Dataset.from_list(rows).save_to_disk(str(destination))
    return len(rows), sum(len(row["target"]) for row in rows)


def _records(data_root: Path, name: str, term: str) -> tuple[list[dict], int, int]:
    from configs.paths import PROJECT_ROOT
    source_root = PROJECT_ROOT / "third_parties" / "gift-eval" / "src"
    sys.path.insert(0, str(source_root))
    from gift_eval.data import Dataset

    os.environ["GIFT_EVAL"] = str(data_root.resolve())
    native = Dataset(name=name, term=term, to_univariate=False, storage_env_var="GIFT_EVAL")
    cutoff = native.prediction_length * (native.windows + 1)
    records = []
    for row_number, row in enumerate(native.hf_dataset.with_format("numpy")):
        target, item_id = row["target"], str(row.get("item_id", row_number))
        if target.ndim == 1:
            records.append({"item_id": item_id, "target": target[:-cutoff].tolist(), "freq": str(row["freq"])})
        else:
            for dimension, values in enumerate(target):
                records.append({"item_id": "%s_dim%d" % (item_id, dimension), "target": values[:-cutoff].tolist(), "freq": str(row["freq"])})
    return records, cutoff, native.prediction_length * native.windows


def extract_assets(spec: ReferenceExtractionSpecification, declaration_path: Path, data_root: Path, declaration: Mapping, output_dir: Path, digest: Callable[..., str]) -> list[dict]:
    selection = spec.mapping["task_selection"]
    if selection["mode"] != "explicit":
        raise ValueError("GIFT-Eval extraction requires an explicit frozen task list")
    groups: dict[str, list[str]] = {}
    for item in selection["items"]:
        name, term = item.rsplit("|", 1)
        groups.setdefault(name, []).append(term)
    assets = []
    for name, terms in sorted(groups.items()):
        selected = max(terms, key=lambda term: _records(data_root, name, term)[1])
        records, training_reservation, test_reservation = _records(data_root, name, selected)
        projection = digest(name, "univariate-target-v1")
        asset_id = digest("gifteval", name, spec.mapping["upstream"]["data_revision"], projection, spec.mapping["prefix_policy"])
        native = Path("assets") / asset_id
        count, values = _save_records(records, output_dir / native)
        assets.append({"asset_id": asset_id, "asset_identity": name, "series_projection_id": projection, "task_ids": sorted("%s|%s" % (name, term) for term in terms), "earliest_test_boundary": {"kind": "position", "value": test_reservation, "provider_evidence": "gift_eval.Dataset.test_data"}, "training_boundary": {"kind": "position", "value": training_reservation}, "native_output": {"location": str(native), "format": "huggingface-dataset-on-disk.v1"}, "source": {"identity": name, "revision": spec.mapping["upstream"]["data_revision"]}, "trajectory_count": count, "value_count": values})
    return assets
