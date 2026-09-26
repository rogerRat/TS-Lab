from pathlib import Path

import numpy as np
import pytest

from assets.dataset_processing.load_reuse import (
    dataset_identity_from_raw,
    make_load_metadata,
    read_load_metadata,
    reuse_load_stage,
    write_load_metadata,
)
from assets.dataset_processing.shard_dataset import ShardDatasetReader, ShardDatasetWriter
from ts3.spec.files.shard_dataset import validate_shard_dataset_bundle


def _identity(batch="profile-10b"):
    return {
        "dataset_id": "profile-10b",
        "batch_id": batch,
        "raw_fingerprint": "raw-" + batch,
    }


def _bundle(root: Path, metadata=None) -> Path:
    writer = ShardDatasetWriter(root, shard_bytes=64)
    writer.append(
        trajectory_id="source/d/s/t", scope="source", dataset="d",
        subset="s", series_id="t", values=np.arange(20, dtype=np.float64),
    )
    writer.finish()
    if metadata is not None:
        write_load_metadata(root, metadata)
    return root


def test_reuse_keeps_index_local_and_links_only_bin_pool(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    candidate = _bundle(
        tmp_path / "candidate",
        make_load_metadata(
            _identity(), scope="source",
            selection={"dataset": "all", "subset": "all", "seed": 0},
            shard_bytes=64,
        ),
    )
    target = tmp_path / "target"
    expected = read_load_metadata(candidate)
    caplog.set_level("INFO", logger="test.reuse")
    assert reuse_load_stage(target, candidate, expected, logger=__import__("logging").getLogger("test.reuse"))
    assert (target / "load" / "shards").is_symlink()
    assert not (target / "load" / "index.arrow").is_symlink()
    assert not (target / "load" / "meta.arrow").is_symlink()
    assert validate_shard_dataset_bundle(target, include_feature=False).root == target.resolve()
    reader = ShardDatasetReader(target)
    np.testing.assert_array_equal(reader.read("source/d/s/t"), np.arange(20, dtype=np.float64))
    reader.close()
    assert "BIN SHARD REUSE: linked" in caplog.text
    assert read_load_metadata(target)["reuse"]["mode"] == "symlink"


def test_reuse_rejects_different_dataset_batch_and_leaves_target_untouched(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    candidate = _bundle(
        tmp_path / "candidate",
        make_load_metadata(
            _identity("profile-10b-seed42"), scope="source",
            selection={"dataset": "all", "subset": "all", "seed": 42},
            shard_bytes=64,
        ),
    )
    expected = make_load_metadata(
        _identity("profile-11b"), scope="source",
        selection={"dataset": "all", "subset": "all", "seed": 42},
        shard_bytes=64,
    )
    caplog.set_level("INFO", logger="test.reuse")
    assert not reuse_load_stage(
        tmp_path / "target", candidate, expected,
        logger=__import__("logging").getLogger("test.reuse"),
    )
    assert not (tmp_path / "target").exists()
    assert "metadata mismatch" in caplog.text


def test_raw_identity_uses_profile_and_batch_metadata(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        root.mkdir()
        (root / "profile_manifest.json").write_text(
            '{"profile":"profile-10b","seed":42,"status":"completed"}\n',
            encoding="ascii",
        )
        (root / "fetch_manifest.json").write_text('{"dataset":"same"}\n', encoding="ascii")
        (root / "data.bin").write_bytes(b"abc")
    assert dataset_identity_from_raw(first) == dataset_identity_from_raw(second)
    (second / "profile_manifest.json").write_text(
        '{"profile":"profile-11b","seed":42,"status":"completed"}\n',
        encoding="ascii",
    )
    assert dataset_identity_from_raw(first) != dataset_identity_from_raw(second)
