"""Fetch only the immutable evaluation snapshot (run on the experiment host)."""
import argparse
import json
import shutil
from pathlib import Path

from evaluate.quito_data import REPO_ID, REVISION, data_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Snapshot directory on data disk")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    identity = {"repo_id": REPO_ID, "revision": REVISION, "license": "CC BY 4.0"}
    manifest = root / "data_manifest.json"
    if root.exists() and any(root.iterdir()):
        if manifest.is_file() and json.loads(manifest.read_text()) == identity and all(
                data_file(root, subset).is_file() for subset in ("hour", "min")):
            return 0
        raise FileExistsError("Existing Quito data is incomplete or has different provenance; use a new root")
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().dataset_info(REPO_ID, revision=REVISION, files_metadata=True)
    total = sum(item.size or 0 for item in info.siblings)
    parent = root
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < total * 1.2:
        raise OSError("Insufficient space on Quito target disk")
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", revision=REVISION,
                      local_dir=str(root), allow_patterns=["README.md", "v20260315/*.parquet"])
    import pyarrow.parquet as pq
    for subset in ("hour", "min"):
        parquet = pq.ParquetFile(data_file(root, subset))
        if not parquet.metadata.num_rows or not {"item_id", "date_time", *(
                "ind_%d" % i for i in range(1, 6))} <= set(parquet.schema.names):
            raise ValueError("Invalid Quito parquet schema")
    manifest.write_text(json.dumps(identity, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
