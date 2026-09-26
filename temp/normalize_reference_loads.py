from pathlib import Path
import filecmp
import hashlib
import os
import shutil

from assets.dataset_processing.load_reuse import (
    dataset_identity_from_raw,
    make_load_metadata,
    write_load_metadata,
)
from ts3.spec.files.shard_dataset import DEFAULT_SHARD_BYTES, validate_shard_dataset_bundle

project = Path("/root/TS3")
data_root = project / "data/bench/fev"
spec_path = project / "configs/ref_spec/fev/spec.yaml"
declaration_path = spec_path.parent / "declaration.yaml"
if not declaration_path.is_file():
    import yaml
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    declaration_path = spec_path.parent / spec["evaluation_declaration"]["path"]
identity_batch = hashlib.sha256(
    spec_path.read_bytes() + declaration_path.read_bytes()
).hexdigest()
identity = dataset_identity_from_raw(
    data_root, dataset_id="fev-reference", batch_id=identity_batch,
)
base = Path("/root/autodl-tmp/TS3/data/ref/shard_dataset")
protocols = [base / "profile-10b-segl256s256/fev", base / "profile-10b-segl512s512/fev", base / "profile-10b-segl1024s1024/fev"]
canonical = protocols[-1]
assets = sorted(p.name for p in canonical.iterdir() if p.is_dir())
if not assets:
    raise RuntimeError("no reference assets in canonical protocol")

def out(message):
    print(message, flush=True)

# Confirm that the existing protocol copies describe the same Load rows and
# values before making the non-canonical shard directories links.
for asset in assets:
    source = canonical / asset
    for protocol in protocols[:-1]:
        target = protocol / asset
        if not target.is_dir():
            raise RuntimeError("missing reference asset: %s" % target)
        for name in ("index.arrow", "meta.arrow"):
            if not filecmp.cmp(source / "load" / name, target / "load" / name, shallow=False):
                raise RuntimeError("reference Load differs: %s/%s" % (asset, name))
out("REFERENCE BIN SHARD REUSE NORMALIZE: confirmed %d identical asset Load tables" % len(assets))

metadata_for = {}
for protocol in protocols:
    for asset in assets:
        bundle = protocol / asset
        metadata = make_load_metadata(
            identity,
            scope="reference",
            selection={
                "asset_id": asset,
                "spec_batch": identity_batch,
                "include_future": False,
            },
            shard_bytes=DEFAULT_SHARD_BYTES,
        )
        write_load_metadata(bundle, metadata)
        metadata_for[bundle] = metadata

backups = []
try:
    for protocol in protocols[:-1]:
        for asset in assets:
            target_shards = protocol / asset / "load" / "shards"
            backup = protocol / asset / "load" / ".shards-before-reuse"
            if target_shards.is_symlink():
                continue
            if backup.exists():
                raise RuntimeError("backup already exists: %s" % backup)
            target_shards.rename(backup)
            backups.append(backup)
            relative = os.path.relpath((canonical / asset / "load" / "shards").resolve(), target_shards.parent)
            target_shards.symlink_to(relative, target_is_directory=True)
    for protocol in protocols:
        for asset in assets:
            validate_shard_dataset_bundle(protocol / asset, include_feature=False)
    out("REFERENCE BIN SHARD REUSE NORMALIZE: validated %d protocol asset bundles" % (len(protocols) * len(assets)))
finally:
    for backup in backups:
        if backup.exists():
            shutil.rmtree(backup)
out("REFERENCE BIN SHARD REUSE NORMALIZE: l256/l512 link only load/shards to l1024")
