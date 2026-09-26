from pathlib import Path

import pytest

from ts3.spec.ref_extraction import MANIFEST_VERSION, validate_manifest


def test_reference_manifest_v2_rejects_hash_fields(tmp_path: Path) -> None:
    value = {"schema_version": MANIFEST_VERSION, "spec_id": "spec-example", "benchmark": "fev", "prefix_policy": "training_common_earliest_test", "assets": [{"asset_id": "asset", "asset_identity": "example", "series_projection_id": "projection", "task_ids": ["task"], "earliest_test_boundary": {"kind": "position", "value": 0, "provider_evidence": "fixture"}, "training_boundary": {"kind": "position", "value": 0}, "native_output": {"location": "assets/asset", "format": "huggingface-dataset-on-disk.v1"}, "source": {"identity": "example", "revision": "r1"}, "trajectory_count": 0, "value_count": 0}]}
    (tmp_path / "assets" / "asset").mkdir(parents=True)
    assert validate_manifest(value, tmp_path)["schema_version"] == MANIFEST_VERSION
    value["spec_sha256"] = "legacy"
    with pytest.raises(ValueError):
        validate_manifest(value, tmp_path)
