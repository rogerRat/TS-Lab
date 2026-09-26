from pathlib import Path
import filecmp
import os
import shutil
import sys

from assets.dataset_processing.load_reuse import (
    dataset_identity_from_raw,
    make_load_metadata,
    write_load_metadata,
)
from ts3.spec.files.shard_dataset import DEFAULT_SHARD_BYTES

root = Path("/root/autodl-tmp/TS3/data/src/shard_dataset")
l256 = root / "profile-10b-segl256s256"
l512 = root / "profile-10b-segl512s512"
l1024 = root / "profile-10b-segl1024s1024"
raw = Path("/root/autodl-tmp/TS3/data/src/raw/profiles/profile-10b")
identity = dataset_identity_from_raw(raw)
metadata = make_load_metadata(
    identity,
    scope="source",
    selection={
        "dataset": "all",
        "subset": "all",
        "max_series": "inf",
        "seed": 42,
        "preserve_invalid_values": False,
    },
    shard_bytes=DEFAULT_SHARD_BYTES,
)

def log(message):
    print(message, flush=True)

log("BIN SHARD REUSE NORMALIZE: comparing l256 and l1024 shard sets")
a = l256 / "load" / "shards"
b = l1024 / "load" / "shards"
a_files = sorted(p for p in a.iterdir() if p.is_file())
b_files = sorted(p for p in b.iterdir() if p.is_file())
if [p.name for p in a_files] != [p.name for p in b_files]:
    raise RuntimeError("shard file names differ")
for number, (left, right) in enumerate(zip(a_files, b_files), 1):
    if left.stat().st_size != right.stat().st_size or not filecmp.cmp(left, right, shallow=False):
        raise RuntimeError("shard content differs: %s" % left.name)
    if number == 1 or number % 32 == 0 or number == len(a_files):
        log("BIN SHARD REUSE NORMALIZE: compared %d/%d files" % (number, len(a_files)))

write_load_metadata(l256, metadata)
write_load_metadata(l1024, metadata)

# Convert the pre-existing whole-load symlink into a protocol-local load
# directory. Only its immutable shard directory remains a symlink.
if l512 / "load" in l512.iterdir():
    load_link = l512 / "load"
    if not load_link.is_symlink():
        raise RuntimeError("expected l512/load to be a symlink")
    temp_load = l512 / ".load-localizing"
    if temp_load.exists():
        shutil.rmtree(temp_load)
    temp_load.mkdir()
    shutil.copy2(l256 / "load" / "index.arrow", temp_load / "index.arrow")
    shutil.copy2(l256 / "load" / "meta.arrow", temp_load / "meta.arrow")
    relative = os.path.relpath((l256 / "load" / "shards").resolve(), temp_load)
    (temp_load / "shards").symlink_to(relative, target_is_directory=True)
    write_load_metadata(l512, metadata) if False else None
    load_link.unlink()
    temp_load.rename(load_link)
write_load_metadata(l512, metadata)
log("BIN SHARD REUSE NORMALIZE: l512 owns index/meta and links only load/shards")

# The duplicate l1024 bin pool is now no longer needed. Keep a temporary
# recoverable rename until all three bundles validate, then remove it.
backup = root / ".backup-profile-10b-segl1024s1024-shards"
if backup.exists():
    raise RuntimeError("backup path already exists: %s" % backup)
old_shards = l1024 / "load" / "shards"
old_shards.rename(backup)
relative = os.path.relpath((l256 / "load" / "shards").resolve(), l1024 / "load")
(l1024 / "load" / "shards").symlink_to(relative, target_is_directory=True)

from ts3.spec.files.shard_dataset import validate_shard_dataset_bundle
for bundle in (l256, l512, l1024):
    validate_shard_dataset_bundle(bundle, include_feature=False)
    log("BIN SHARD REUSE NORMALIZE: validated %s" % bundle.name)

shutil.rmtree(backup)
log("BIN SHARD REUSE NORMALIZE: removed verified duplicate l1024 bin pool")
