"""Arrow-only deduplication for one ShardDatasetBundle.

This stage must never copy or mutate Load bins; it only records the new drop
decision so every consumer shares one immutable value store.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

from assets.dataset_processing.shard_dataset import ShardDatasetReader
from tsl.spec.files.shard_dataset import DEDUP_SCHEMA, validate_shard_dataset_bundle, write_arrow

log = logging.getLogger("assets.dataset_processing.dedup")


def _digest(values: np.ndarray) -> str | None:
    if not np.isfinite(values).all():
        return None
    return hashlib.sha256(np.ascontiguousarray(np.rint(values * 1e8), dtype="<i8").tobytes()).hexdigest()


def deduplicate_bundle(bundle_root: Path, scopes: set[str] | None = None) -> Path:
    """Write `dedup.arrow` without changing any Load shard or locator."""
    log.info("Dedup starting: bundle=%s scopes=%s", bundle_root, sorted(scopes) if scopes else "all")
    bundle = validate_shard_dataset_bundle(bundle_root)
    log.info("Bundle validated: load_rows=%d", len(bundle.load_index))
    target = bundle.root / "dedup" / "dedup.arrow"
    if target.exists(): raise FileExistsError(target)
    scope_by_id = {row["trajectory_id"]: row["scope"] for row in bundle.meta}
    reader, winners, digests = ShardDatasetReader(bundle.root), {}, {}
    try:
        for number, row in enumerate(bundle.load_index, 1):
            tid = row["trajectory_id"]
            if row["dropped"] or (scopes is not None and scope_by_id[tid] not in scopes): continue
            digest = _digest(reader.read(tid))
            if digest is None:
                continue
            digests[tid] = digest
            winners[digest] = min(winners.get(digest, tid), tid)
            if number == 1 or number % 1000 == 0:
                log.info("Deduplication scanned %d/%d records; %d candidates retained", number, len(bundle.load_index), len(digests))
    finally: reader.close()
    rows = []
    for row in bundle.load_index:
        tid, digest = row["trajectory_id"], digests.get(row["trajectory_id"])
        duplicate = digest is not None and winners[digest] != tid
        rows.append({"trajectory_id": tid, "dropped": duplicate, "drop_stage": "dedup" if duplicate else None,
                     "drop_reason": "duplication" if duplicate else None,
                     "canonical_trajectory_id": winners[digest] if duplicate else None})
    write_arrow(target, rows, DEDUP_SCHEMA); return target
