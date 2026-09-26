from pathlib import Path
from dataclasses import replace
import yaml
from ts3.spec.catalogs.reader import CatalogReader
from ts3.spec.catalogs.models import CatalogTrajectoryRecord, SourceCatalog
from ts3.spec.files.shard_dataset import validate_shard_dataset_bundle

root = Path("configs/catalog/src")
current = CatalogReader().read(root / "profile-10b-segl1024s1024.yaml", verify=False)
rows = current.records_data
print("input rows", len(rows), "input bundle", rows[0].bundle_directory)
for suffix in ("256s256", "512s512", "1024s1024"):
    bundle = Path("/root/autodl-tmp/TS3/data/src/shard_dataset/profile-10b-segl" + suffix)
    validated = validate_shard_dataset_bundle(bundle, include_feature=False)
    ids = {x["trajectory_id"] for x in validated.load_index}
    if len(ids) != len(rows) or any(r.catalog_trajectory_id not in ids for r in rows):
        raise RuntimeError("trajectory IDs do not match " + suffix)
    records = tuple(CatalogTrajectoryRecord(
        r.catalog_trajectory_id, r.scope, r.dataset, r.subset, r.series_id,
        r.length, str(validated.root), r.shard_id, r.value_offset,
        r.dropped, r.drop_stage, r.drop_reason
    ) for r in rows)
    out = root / ("profile-10b-segl" + suffix + ".yaml")
    table_name = "trajectory-profile-10b-segl" + suffix + ".arrow"
    CatalogReader().write(SourceCatalog(table_name, records, str(out)), out, validated.root)
    checked = CatalogReader().read(out, verify=True)
    print(out, "rows", len(checked.records_data), "bundle", checked.records_data[0].bundle_directory, "table", table_name)
