# V7 computation/storage accounting

This is an observational resource result attached to the existing V7 run, not a
new training/gradient experiment. No model import, BP, score recomputation or
NPZ vector decompression is performed. ZIP directories and NPY array headers
provide storage sizes; tiny cost metadata and existing JSON receipts provide
timing/budget evidence. No content hashes are created.

Run after each result report refresh (single report writer; do not run concurrently
with the main Markdown generator):

```bash
cd /root/TS3
PYTHONPATH=src:. /root/miniconda3/envs/ts3/bin/python -m scripts.report_v7_costs \
  --config configs/rags/main_v7_curve4.json \
  --run /root/autodl-tmp/TS3/outputs/main-v7-20260923 \
  --markdown docs/results/v7.md
```

Outputs are `RUN/resource_costs/report.json` (cell-level auditable source paths,
array logical/compressed bytes, receipt costs, BP requirements, raw executed
training timing, configured-epoch effects) and `report.md`. Atomic replacement
updates only `V7-COSTS-BEGIN/END` in the target report; existing results are retained.
Keep these as experimental results. Workers need no restart or configuration change.

Scope follows current config tasks/seeds/models/arms. Excluded task data is untouched.
Each model is shown separately; methods ordered by paired-vs-rand improvement at the
configured endpoint (Chronos 3, others 4), not best checkpoint. Available-cell counts
are shown because the experiment is incomplete. Missing measurements are null/未测,
never zero. Historical 4-epoch executed costs remain 4-epoch costs.

Selection elapsed includes saving or warm load. Shared scoring is charged to the
first missing gradient arm, normally grad-dot; compare the three-arm group for actual
shared work, not these deltas as independent cold costs. Per-arm required BP is a
logical method budget, not incremental compute. Asset receipts are reported separately;
RAGS variants and rags_traj preparation aliases must not be added together. Cache
hits/imported rows can repeat across tasks; these are observations, not unique identities.

Do not claim FLOPs/device-hours or isolated-method speedups from wall-clock timings
collected across migrated GPUs and concurrent jobs. Historical peak RSS/VRAM, separate
sort/sampling timing and standalone cold cost are unavailable. Shared gradient-cache
physical storage is not attributable per arm and is not included in selection bytes.
Physical shared-cache inventory stats only receipt-referenced namespaces, deduplicating
hardlinks and skipping internal symlinks. No shard contents are read. This snapshot may
include excluded-task/shared resources; it is not marginal per-arm or seven-task storage.
Old donor namespaces are not traversed. Failed/unreceipted work is not reconstructed.
A controlled isolated profiling experiment
would be needed for those claims; this accounting never reruns work to invent them.
