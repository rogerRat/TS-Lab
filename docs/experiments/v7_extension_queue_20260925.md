# V7 exploratory task queue — 2026-09-25

User approved adding tasks after the original seven-task/two-seed matrix, and
explicitly prioritized small reference computation. Existing 840 cells remain
unchanged. No seed2. No new RAGS arm is implicitly introduced by this extension.
Twenty existing arms, three models, seeds0/1 = 120 cells/task, 360 queued cells.
Chronos 3 epochs, Timer/TimesFM 4, full half-epoch curves, original six-epoch
schedule and optimizer/window/rank/CRB settings remain unchanged.

| Queue | Native TIME task | Finite reference trajectories | Legal reference windows | RAGS reference BP |
|---|---|---:|---:|---:|
| 1 (already started) | NE_China_Wind/H | 4 | 29,876 | 444 |
| 2 | SG_PM25/H | 82 | 44,096 | 707 |
| 3 | Coastal_T_S/H | 72 | 305,409 | 4,572 |

Counts use C+H=576, original per-trajectory rounded .005/.005/.005 budgets.
Uniform/LESS union BP can exceed RAGS alone; actual receipts are authoritative.
SG PM2.5 has 6,653 nonfinite pre-exam points; the unchanged native export splits
finite runs, excludes runs shorter than 576, never fills missing values. There
are 170 valid official exam windows across five entities. MetroPT-3/5T was
originally queued with 11,082 legal reference windows / 192 RAGS BP, but failed
the formal input gate with zero valid exam windows. Smart_Manufacturing/H failed
the same gate; Current_Velocity/H is absent from the pinned official protocol.
These geometry-only failures were identified without evaluating model results.
All admitted reference fragments are nonconstant. The census reads pre-exam
history only; no test performance selects tasks. Wind means atmospheric wind
components/geopotential/temperature, not wind power. PM2.5 variability
and coastal environmental variability are hypotheses for adaptive selection,
not claims that RAGS will win. Report these as exploratory, separately from the
fixed original matrix, retaining every outcome and full curves.

## Identity and reuse

New run roots: `outputs/main-v7-20260923/extensions/ne-china-wind-20260925`,
`time_sg_pm25_h-20260925`, `time_coastal_ts_h-20260925`.
Configs are `configs/rags/v7_extension_*.json`. The extension validator accepts
only these task declarations and the exact derived original V7 configuration;
changing scientific settings still fails. Each task has separate immutable
protocol/input/reference/selection/result identities, same seed pool manifests.

Feasible packed source + seed0/1 manifests are the original inodes. Each task's
`gradient_cache/` has a new full manifest containing its reference identity.
Source index keys and every referenced source shard key are verified against the
parent published index. Only immutable source NPZ payloads and field-verified
source CRB payloads are hardlinked; indexes/manifests remain independent files.
Source model/projection/precision/loss, pool identity, physical IDs/starts and
CRB configuration remain exact. Old source shards are never appended to; native
GradientCache only appends unpublished shards created in its current session.
No old gradients/results/configs are overwritten. Source requests missing from
the published extension cache raise BEFORE backward, even during assets.

Wind reused source rows Chronos/TimesFM/Timer = 622411/622296/622270 and 3450 CRB
trajectory payloads/model. Source payload additional physical bytes = 0. Index,
manifest and new reference/selection data are genuinely incremental storage.
`extension_source_reuse.json` and `input_preflight.json` provide the evidence.
Use the extension's `TS3_GRADIENT_CACHE_ROOT=RUN/gradient_cache`, not the parent's
shared cache root. No data hashes are computed. TSDS source feature payloads
are likewise hardlinked with their original validation manifests.

## Autonomous execution

`python -m scripts.run_v7_extension_queue --parent PARENT --adopt-first-assets`
adopts the already-started Wind asset PIDs via pidfd. Do not launch a duplicate;
the parent `extension_queue.lock` enforces a single supervisor.
Next task: prepare inputs/source links -> all six sparse asset receipts -> each
model zero-BP select -> original evaluation -> nonblocking cell workers.
Model pipelines need not wait for other models' selection. Only seeds0/1 are
admitted. Within a model, prior seed results precede the next seed.
The selection worker itself admits a model once that model's two asset receipts
pass identity/manifest gates, even if another model is still preparing assets.
This changes only scheduling, not gradients, scores or scientific settings.
For Coastal, the original supervisor is temporarily stopped while three
independent selectors finish; a pidfd event watcher then resumes it. Its
idempotent select pass skips completed receipts, so no BP or score recomputation
is permitted. The Chronos asset worker is never stopped.
GPU0 has one admitted process slot; GPU1 two (previous measured model overlap).
Available slots claim ready work by task/arm order; busy cells are skipped.
Initialization and selection dependencies still apply; no duplicate computation
or useless work is launched solely to make the utilization meter read 100%.

Scheduling waits on pidfd/selectors process events, not polling. Completed
native cell identities and all curve endpoints are audited before the next task.
Every task emits `resource_costs/report.{json,md}` and `task_queue_complete.json`.
Parent `extension_queue_status.json`, `extensions/queue.log`, per-run `logs/auto_*`
are the runtime evidence. Errors fail explicitly; no blind retry or cache removal.
Kimi owns 30-minute timed inspection, early failure reporting, and adding the
separate exploratory results/cost links to v7.md. Context >30%: compact.
The approved queue is finite. Latest user priority after its three tasks is
the proposed RAGS midpoint+slope shape method on prior two-seed cells, not an
automatic fourth native TIME task. Its distance normalization and four-arm
scientific definition require user confirmation before freezing or launching.

## Checks

32 tests passed before launch (extension source inode/key/manifest reuse, source
BP fail-closed guard, original V7/reduced protocol and queue claims). A further
12 targeted extension/queue/curve/reduced tests passed after original-only queue
initialization support. Native reference census and source CRB warm logs are
production-scale smoke evidence. A first setup-only process was stopped to fix
repeated NPZ index loading (materialize once); no GPU work or reusable payload
was lost. Field-directory names were corrected within the new namespace before
asset launch so native cache discovery resolves the prepopulated cache first.
