# 2026-09-23 critical-only container migration

User changed full clone to essential non-retrievable assets only, then confirmed:
historical results and reproducibility evidence retained; old continuation assets
not migrated. Source exp remains unchanged and must not be released until acceptance.
Target: root@connect.westb.seetacloud.com:43058, 850 GiB data disk.

Whitelist: current dirty code/Git provenance; frozen V7 pool and native reference/test
inputs plus donor input identities; source Arrow identity metadata; three models'
published real-gradient shards/index/manifest; historical result/protocol/cost records.
No old checkpoint/resume, old selection arrays/CRB representatives, full raw source
or reconstructed bins, public model weights, package environments/download caches,
or TimeMoE gradient pool. This is deliberately not a complete clone of the old runs.

V7 frozen feasible.npz holds complete native trajectories; runtime assets/train use
that snapshot. Source index metadata retained for preflight, but missing full shards
must be explicitly treated as snapshot-only deployment (standalone legacy preflight
currently returns nonzero). Do not bypass this without a verified snapshot-mode
acceptance; do not regenerate pools or alter manifests. Legacy input hardlinks and
V7 references transferred in the same rsync batch to preserve samefile checks.

All three gradient indexes were readable, every published shard was present; sampled
NPZ structure has 4096-dimensional gradients and finite losses. This is not yet full
destination integrity acceptance. No asset hashes calculated.

Lossless tests used bounded byte samples and exact decompression comparison. Zstd-1
gradient samples retained 53-83% of bytes, JSON sample ~10%, already compressed pool
~100%. Zstd-3 gave little added compression. Transport uses rsync zstd-1 with suffix
skip disabled, archive/hardlinks/ACLs/xattrs/numeric IDs, partial resume, no --delete.

Current local session exp-critical-migration-20260923. Source lists/report/logs:
/tmp/ts3-critical-migration-20260923. Remote copy of whitelist manifest:
/root/autodl-tmp/migration-exp-20260923/critical-manifest.json.
Kimi owns scheduled inspections, no polling loops; experimental launch stays frozen.
