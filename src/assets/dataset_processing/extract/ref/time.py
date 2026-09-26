"""TIME reference adapter: retain the earliest safe common history."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Mapping

from ts3.spec.ref_extraction import ReferenceExtractionSpecification


def _save_records(records, destination: Path) -> tuple[int, int]:
    from datasets import Dataset

    rows = list(records)
    if not rows:
        raise ValueError("TIME extraction produced no trajectories")
    Dataset.from_list(rows).save_to_disk(str(destination))
    return len(rows), sum(len(row["target"]) for row in rows)


def _records(data_root: Path, dataset_name: str, terms: list[str], declaration: Mapping) -> tuple[list[dict], int, int]:
    from configs.paths import PROJECT_ROOT
    source_root = PROJECT_ROOT / "third_parties" / "TIME" / "src"
    sys.path.insert(0, str(source_root))
    from timebench.evaluation.data import Dataset

    settings = declaration["datasets"][dataset_name]
    reservations = [int(settings[term]["prediction_length"]) * (max(1, int(settings["test_length"]) // int(settings[term]["prediction_length"])) + 1) for term in terms]
    reservation, test_boundary = max(reservations), int(settings["test_length"])
    raw = Dataset(name=dataset_name, term=terms[0], to_univariate=False, prediction_length=int(settings[terms[0]]["prediction_length"]), test_length=test_boundary, val_length=int(settings.get("val_length", 0)), storage_path=data_root)
    records = []
    for row_number, row in enumerate(raw.hf_dataset.with_format("numpy")):
        item_id, target = str(row.get("item_id", row_number)), row["target"]
        if target.ndim == 1:
            records.append({"item_id": item_id, "target": target[:-reservation].tolist(), "freq": str(row["freq"])})
        else:
            for dimension, values in enumerate(target):
                records.append({"item_id": "%s_dim%d" % (item_id, dimension), "target": values[:-reservation].tolist(), "freq": str(row["freq"])})
    return records, reservation, test_boundary


def extract_assets(spec: ReferenceExtractionSpecification, declaration_path: Path, data_root: Path, declaration: Mapping, output_dir: Path, digest: Callable[..., str]) -> list[dict]:
    selected = spec.mapping["task_selection"]["items"]
    if spec.mapping["task_selection"]["mode"] != "explicit":
        selected = ["%s|%s" % (name, term) for name, settings in declaration["datasets"].items() for term, value in settings.items() if isinstance(value, Mapping) and "prediction_length" in value]
    groups: dict[str, list[str]] = {}
    for item in selected:
        name, term = item.rsplit("|", 1)
        groups.setdefault(name, []).append(term)
    assets = []
    for name, terms in sorted(groups.items()):
        records, training_boundary, test_boundary = _records(data_root, name, sorted(terms), declaration)
        projection = digest(name, "univariate-target-v1")
        asset_id = digest("time", name, spec.mapping["upstream"]["data_revision"], projection, spec.mapping["prefix_policy"])
        native = Path("assets") / asset_id
        count, values = _save_records(records, output_dir / native)
        assets.append({"asset_id": asset_id, "asset_identity": name, "series_projection_id": projection, "task_ids": sorted("%s|%s" % (name, term) for term in terms), "earliest_test_boundary": {"kind": "position", "value": test_boundary, "provider_evidence": "timebench.Dataset.test_data"}, "training_boundary": {"kind": "position", "value": training_boundary}, "native_output": {"location": str(native), "format": "huggingface-dataset-on-disk.v1"}, "source": {"identity": name, "revision": spec.mapping["upstream"]["data_revision"]}, "trajectory_count": count, "value_count": values})
    return assets
