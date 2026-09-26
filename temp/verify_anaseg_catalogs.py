from pathlib import Path
import json, glob, yaml
from ts3.spec.catalogs.reader import CatalogReader
from ts3.spec.files.shard_dataset import validate_shard_dataset_bundle

reader = CatalogReader()
print("SOURCE")
for p in sorted(glob.glob("configs/catalog/src/profile-10b-segl*.yaml")):
    c = reader.read(p, verify=True)
    h = yaml.safe_load(Path(p).read_text())
    print(Path(p).name, "rows=", len(c.records_data), "bundle=", c.records_data[0].bundle_directory, "table=", h["trajectory_table"])
print("REFERENCE")
for root in sorted(Path("configs/catalog/ref").glob("profile-10b-segl*/fev")):
    ys = sorted(root.rglob("*.yaml"))
    long_only = all("/fev-" in str(p) and "fev-" + str(p).split("/fev-")[-1].split("/")[0] in str(p) for p in ys)
    ok = 0
    bundles = set()
    for p in ys:
        c = reader.read(p, verify=True)
        ok += 1
        bundles.add(c.records_data[0].bundle_directory if c.records_data else "")
    print(root, "yaml=", len(ys), "verified=", ok, "bundles=", len(bundles), "long_only=", long_only)
print("FEATURES")
for suffix, expected in (("256s256",256),("512s512",512),("1024s1024",1024)):
    b = Path("/root/autodl-tmp/TS3/data/src/shard_dataset/profile-10b-segl"+suffix)
    m = json.loads((b/"feature/metadata.json").read_text())
    print(suffix, {k:m.get(k) for k in ("analysis_segment_length","analysis_segment_stride","rows","valid_rows","dropped_rows","num_segments")}, "load=", (b/"load").is_symlink(), "dedup=", (b/"dedup").is_symlink())
