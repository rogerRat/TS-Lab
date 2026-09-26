"""Bundle stages share Load bins and express decisions through Arrow only."""
from pathlib import Path

from assets.dataset_processing.dedup import deduplicate_bundle
from assets.dataset_processing.shard_dataset import ShardDatasetWriter
from ts3.spec.files.shard_dataset import validate_shard_dataset_bundle


def test_global_dedup_marks_duplicate_without_rewriting_load_bins(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    writer = ShardDatasetWriter(bundle, shard_bytes=1024)
    writer.append(trajectory_id="source/z/subset/z", scope="source", dataset="z_dataset", subset="subset", series_id="z", values=[1., 2., 3.])
    writer.append(trajectory_id="source/a/subset/a", scope="source", dataset="a_dataset", subset="subset", series_id="a", values=[1., 2., 3.])
    writer.finish()
    before = (bundle / "load" / "shards" / "values_000000.bin").read_bytes()
    deduplicate_bundle(bundle)
    loaded = validate_shard_dataset_bundle(bundle)
    assert (bundle / "load" / "shards" / "values_000000.bin").read_bytes() == before
    dedup = {row["trajectory_id"]: row for row in loaded.dedup}
    assert dedup["source/z/subset/z"]["dropped"]
    assert dedup["source/z/subset/z"]["drop_reason"] == "duplication"
    assert not dedup["source/a/subset/a"]["dropped"]
    # RAGS consumes the load shards directly; no feature or catalog stage.
    assert not (bundle / "feature").exists()
