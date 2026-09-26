from pathlib import Path
p=Path("ts3/spec/files/shard_dataset.py")
s=p.read_text()
old='FORMAT_VERSION = "ts3.shard-dataset.v1"\nDEFAULT_SHARD_BYTES = 256 * 1024 * 1024\n'
new='FORMAT_VERSION = "ts3.shard-dataset.v1"\n# Batch-level Load identity is kept beside index/meta so protocol-specific\n# bundles can safely share immutable numeric shards without changing row schema.\nLOAD_METADATA_FILENAME = "metadata.json"\nLOAD_METADATA_SCHEMA_VERSION = "ts3.load-identity.v1"\nDEFAULT_SHARD_BYTES = 256 * 1024 * 1024\n'
if old not in s: raise SystemExit("spec anchor missing")
p.write_text(s.replace(old,new,1))

p=Path("assets/dataset_processing/load_reuse.py")
s=p.read_text()
old='from ts3.spec.files.shard_dataset import FORMAT_VERSION, validate_shard_dataset_bundle\n\nLOAD_METADATA_FILENAME = "metadata.json"\nLOAD_METADATA_SCHEMA_VERSION = "ts3.load-identity.v1"\n'
new='from ts3.spec.files.shard_dataset import (\n    FORMAT_VERSION,\n    LOAD_METADATA_FILENAME,\n    LOAD_METADATA_SCHEMA_VERSION,\n    validate_shard_dataset_bundle,\n)\n'
if old not in s: raise SystemExit("reuse import anchor missing")
p.write_text(s.replace(old,new,1))
print("protocol identity constants added")
