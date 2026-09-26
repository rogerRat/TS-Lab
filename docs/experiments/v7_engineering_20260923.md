# V7 engineering and runtime handoff — 2026-09-23

## Seven-task matrix and Chronos three-epoch amendment — 2026-09-24

User approved exclusion of `time_finland_15t` and `time_eweld_15t` based on
observed V6 seed0 RAGS task results, and a cost-driven Chronos stop at 3 epochs.
This is a post-observation task subset, not an unbiased nine-task benchmark;
reports must disclose the selection basis and retain excluded results as evidence.
The active `main_v7_curve4.json` now contains seven datasets, Chronos 3 epochs
(1527 steps, six half-epoch evaluations), Timer/TimesFM 4 (2036, eight).
Six-epoch LR schedule (3054 steps), optimizers and all asset identities stay fixed.
Matrix: 1260 cells including orig; 1197 trained cells; seed0 140/model incl orig.
`asset_protocol` intentionally retains the original nine-task identity for reuse.

No CLI change: same config, run root, `--phase train --student MODEL --seed 0`.
After safe reentry, removed tasks are skipped automatically; old workers do NOT
hot-load the new epoch cap. If Kimi waits for a full cell boundary, its last 4ep
cell is retained and reused. A valid earlier half-epoch Resume can instead
continue to 1527, with strict equality of every contract field except the
approved steps/epochs reduction. Completed/later-than-3ep resumes perform no new
optimizer steps; the persisted full .5..3.0 curve must exist, otherwise fail
explicitly rather than fabricate metrics or evaluate the wrong weight state.

Raw `cells/` results remain execution evidence, including full four-epoch curves
and costs; excluded-task raw cells/gradient/selection assets are not deleted.
For each completed retained task the worker writes `reported_cells/` views with
the active identity. Chronos summaries/MASE/improvements use epoch3, not epoch4
or the best checkpoint. Raw trained steps/costs remain actual, and raw result
paths plus executed/reporting epochs distinguish reuse. Final primary tables
MUST use these views AND the active config datasets; never glob raw cells as the
new primary matrix. Old Timer/TimesFM results are reused without retraining.
The live `training_protocol.json` is atomically migrated only from the exact
known curve4 protocol, archived in `training_protocol_history/`; unrelated
identity changes still fail. Old per-task completion markers are reentered to
materialize views without training completed cells, then updated to the amendment.

Validation: 38 focused tests including exact prefix Resume weights, protocol
rejection, raw-result preservation and reporting metrics; real read-only audit
accepted 95 retained cells, preserved 34 excluded cells and passed asset/select
gates with zero BP at the audit snapshot. Kimi owns boundary reentry and dual-GPU
placement; retain model-independent progress, prioritize seed0, no duplicate worker.

## Timer single-patch forecast compatibility — 2026-09-24

Transformers 4.57.1 routes legacy Timer generate into a token sampler and supplies
a DynamicCache without seen_tokens; old generate also references other removed
helpers. Do not patch cached HF files or downgrade shared environments. For the
frozen C480/H96 case only, the project adapter now performs the pinned greedy
loop's single native forward: five patch attention positions, revin=True,
max_output_length=96, same BF16 input/head and existing nonfinite fallback.
No KV cache is needed for a single output patch. Other horizons keep their prior
path and are not claimed fixed. Training loss, gradient assets and model config
remain unchanged. Kimi resumes Timer train seed0 with existing output identity;
TimesFM and Chronos remain independent and must not wait for Timer.

## Independent model pipelines — user amendment 2026-09-24

Each model advances independently: audit its own 27 selection receipts, then
train its own seeds 0/1/2 in order, task then arm. Other models may still be
selecting or training an earlier seed. This supersedes the cross-model selection
and training-seed barriers below, not the existing scientific artifact identities.
The completed 81-asset gate remains intact. Worker train checks only its model's
selection receipts and previous-seed cells; suite all runs select→train as one
pipeline per model with bounded `--parallel-models` concurrency. Explicit global
audit-select remains available; `--student` makes it a model-local audit.
The obsolete isolated Timer validation check in the suite is removed; curve4
protocol validation remains required.

For the live run, Kimi must not launch suite all/train alongside existing workers
for the same model. Audit and launch ready Timer/TimesFM individually with
`scripts.run_v7 --phase train --student MODEL --seed 0`, original config/run/root;
subsequent seeds wait only for that model. Chronos selection continues untouched.
Coordinate supervisor ownership to avoid duplicate training launches. Allow
model overlap only under GPU memory/throughput admission; scientific batch size,
optimizer, epoch budget, half-epoch evaluation and checkpoint lifecycle unchanged.

## TSDS source-feature reuse — 2026-09-24

V7 now resolves TSDS source feature caches across task directories within the
same model/seed output folder. The first existing source feature file is reused
in place; the existing full feature manifest validation remains mandatory.
Reference features remain task-specific and are not redirected or cached by
this change. No gradient assets, probabilities or native algorithm are changed.
No cache files are moved/deleted/copied, and no cross-seed/model alias is added.
Warm source cost must report `pool_cached=true, executed_windows=0`.
Kimi may adopt at a durable task boundary if useful work remains; do not kill
an in-flight task or restart a finished Timer. Existing workers do not hot-load
this patch. Rebind PID-based supervision before any controlled worker reentry.

## User-approved native zero probabilities — 2026-09-24

User decision: preserve native zero probabilities for TSDS and LP-Mirror;
do not add epsilon, mix uniform mass, rerank, or otherwise smooth probabilities.
`v7_io.validate_support` applies this rule during generation, compact save and
load. All other trained arms still require strictly positive support. Every arm
must retain whole-pool window ownership, finite nonnegative mass summing to one,
pool-external zero exposure and selection-stage zero additional BP. Receipt
coverage includes `support_policy` and the actual zero-mass window count.
This changes only the erroneous unconditional validation, not native algorithms
or stored distributions; existing asset/selection identities remain valid.

Root cause: TSDS assigns OT mass only to retrieved candidates, leaving genuine
zeros elsewhere. The 102-point warning refers to the retrieved-union KDE index,
not missing gradient anchors. LP-Mirror's current V7 path is window-level value
mapping with gating disabled; no change to that implementation is required.
Reenter select with the same config, run root and cache root, reuse all 81 asset
receipts and existing compact selections. Do not rerun assets or delete caches.
Kimi owns formal execution; validate 81 select receipts and zero new BP before
training. Acceptance includes saved/reloaded TSDS zeros remaining exactly zero.

## User-approved Timer donor reuse amendment — 2026-09-23

After disclosure of the numerical differences, the user explicitly approved
reuse ("直接用，我审批通过"). This supersedes the pending-equivalence disposition
in the Timer donor diagnostic; it does not establish numerical equivalence.
`v7_cache.py` permits only Timer revision
`70077a71acce1b4c00d98332fcaabc694255d8e5`, BF16, the old `torch_dtype`
to new `dtype` alias, and serializer version 4.40.1 to 4.57.1. All remaining
model fields and non-data protocol fields must match; existing physical
trajectory/value/reference/window checks remain mandatory.

New receipts' `audit.reuse_audit` records
`compatibility_approval=user-approved-timer-donor-20260923-v1`, original
differences and `numerical_equivalence_claimed=false`. Known 64-window sample:
max relative gradient L2 0.07972383, max relative loss 0.03851220, min cosine
0.99701908; no controlled attribution to GPU/batch/library is claimed.
No old/new manifests or completed receipts are rewritten. Existing destination
rows win; missing rows may be copied bit-exactly from the authorized donor.
This is incremental mixed-provenance reuse, explicitly approved, not a request
to recompute old work. The receipt records the donor path and the exception.

Deployment: Kimi alone coordinates Timer process reentry at a completed asset
cell boundary, with the same config/run root/cache root. Do not restart Chronos
or TimesFM, unlink locks, or allow duplicate Timer writers. If the event-driven
supervisor tracks the old Timer PID, update its dependency before worker exit
can resume the parent suite; keep the parent scheduler from launching a second
Timer writer. Reentry skips durable completed receipts. Acceptance: first new
Timer receipt must show the approval audit and imported_rows (when requested
windows exist in the donor); imported_rows alone is not numerical validation.

## User amendment: four epochs, all checkpoints evaluated, no best selection

This section supersedes the old 3/3/4 and isolated Timer validation statements
below. Use `configs/rags/main_v7_curve4.json` for all new phases. Three models
run at most 4 epochs, evaluate at .5/1/1.5/2/2.5/3/3.5/4, record the complete
curve plus shared pretrained epoch 0. No minimum-MASE selection, validation
gate, test-dependent early stop, or extra validation dataset. Final epoch
metrics are summary fields, not a selected best result. Optimizers and the
6-epoch LR schedule remain unchanged: 2036 max training steps, 3054 schedule
steps, pool_n 16284 for each model.

Checkpoint retention: use the existing recipe-aware online evaluation path.
Each boundary writes a temporary recoverable model/optimizer/RNG state before
evaluation, atomically persists the metrics curve, then replaces the previous
resume state. Preserve the preceding valid state if evaluation fails. After
durable completed-cell results, immediately remove final resume/tmp weights.
Reentry also cleans a completed cell's leftover resume without retraining.
Only reusable cache/gradient resources and experimental results remain long-term.

Asset reuse: the new config embeds the exact original `main_v7.json` as
`asset_protocol`; asset identities, frozen protocol.json, manifests and selection
identities remain unchanged. New `training_protocol.json` and training cell/
phase identities distinguish the amendment. Only the explicitly approved
training fields may differ; the validator rejects unrelated changes. Never
edit old receipts or recompute gradients to resolve a training-only change.
Original `main_v7.json` remains an asset provenance record; the training entry
rejects it without the new amendment. Worker and suite defaults use curve4.

Deployment on cloned instance: `ssh -p 22145 root@connect.bjb1.seetacloud.com`.
Kimi must finish clone integrity/capacity acceptance before phase launch; the
Timer validation blocker is revoked. The 81 asset and selection barriers,
seed/task/arm order and same-model cache locks are unchanged. Use:
`PYTHONPATH=src:. python -m scripts.run_v7_suite --config configs/rags/main_v7_curve4.json --phase all --parallel-models 1`.
The old code/config and evidence sections below are historical where superseded.

Authoritative user protocol: WSL `001-iclr17/rags-md/_exp-0907.md`, V7 opening section.
The superseded 48,852 pool / triple epoch exposure proposal is NOT executable V7.

## Frozen scientific identity

- `configs/rags/main_v7.json`, version `v7-seeded-quarter-pools-20260923`.
- Feasible pool: 65,136 equivalents, pool seed 0, complete native trajectories.
- Actual pool: exact 16,284 equivalents per seed 0/1/2. Deterministic permutation +
  existing 0/1 exact subset-sum witnesses, then stable physical ordering. No cropping,
  concatenation, trajectory-count quota, or forced cross-seed disjointness.
- Stable feasible-pool fragment IDs and 1-based physical starts survive every seed.
  All three models/nine tasks/20 arms share that seed's actual manifest.
- Source and reference CRB: .005/.005/.005; existing endpoint coverage, Python
  per-phase rounding, capacity caps, minimum-kappa/rightmost ties and exhaustion.
- Native windows: CT 512/64, Timer 480/96; single pretrained checkpoint, original
  native loss, 4096-D projection seed 42, existing BF16 resource encoding.
- Uniform baseline K is checked against actual distinct CRB BP per trajectory;
  endpoints included, hypothetical K=1 uses left endpoint. LESS requirements merge
  into the same sparse cache. No full-window gradient sweep.
- Three new arms use raw gradients, signed dot / zero-safe cosine / raw L2 norm.
  Reference means weight legal windows equally within a trajectory, then trajectories
  equally. GraNd's independent reference requirement is zero.
- Ranking is the unchanged frozen exact-tie interval curve with m=1 for windows.
  LP-Mirror and TSDS retain native algorithms, not substituted gradient scores.
- Pool exposure stays 16,284: CT max 1,527 steps (3 epochs), Timer max 2,036 (4 epochs),
  six-epoch schedule 3,054 steps, effective batch 32. Ceil-to-full-batch rounding is
  unchanged. Each seed trains task by task, completing all arms before the next task;
  all models complete seed0 before seed1, then seed2.

## Implementation and immutable resources

Entry: `python -m rags.experiments.v7_run`; optional event-driven phase supervisor
`python -m scripts.run_v7_suite`. Neither imports/calls `seed_rounds` nor hardlinks
seed0 selections into another seed. Kimi owns actual launch/resource admission.

`inputs/feasible.npz` is packed native data, not overlapping windows. Feasible and
three actual-pool JSON manifests are separate; actual pools select stable IDs into
the feasible NPZ. Each model's `gradient_data_identity.json` records the feasible
manifest, all actual manifests, and complete native reference identities. This
joint identity lets actual gradient rows deduplicate across seeds without letting
an out-of-pool row become a selection candidate. Each seed's CRB reconstruction,
ranking and probabilities remain separate.

Existing V6 references/exams may be hardlinked only after explicit native window
and isolation checks; these are input data, not reused V6 formal result cells.
Model cache manifests still undergo the existing full field comparison. The
`TS3_GRADIENT_CACHE_ROOT` model lock is preserved: never run two writers for the
same model, including concurrent V6 and V7 writers. No old cache directory alias,
manifest rewrite, or model/gradient/data hashes are introduced.

Old gradient donors are read-only published-index snapshots. Before importing a
row, compare all non-data gradient fields (model/revision/precision/loss/window/
projection/storage), pool source identity, native source row, complete trajectory
values, and remapped physical window key. Reference values and task geometry
markers are checked separately. Required BF16 bits/losses are copied atomically
into the new V7 field identity. Nonmatching donors are recorded and rejected.
Cross-seed hits use the new joint cache directly and execute no extra BP.

Assets receipts at `seeds/seedN/MODEL/assets/TASK.json` record per-trajectory budgets,
actual BP, imported rows, existing hits, source/reference coverage and norm ranges,
preparation time and gradient time, complete cache fields and method costs.
`TASK__requirements.npz` lists prepared sparse keys; it is not a full window scan.
LP-Mirror's native proxy training/adaptation states are also prepared in this phase
as a small atomic `TASK__lp-mirror.pt` bundle, with proxy costs separate from
foundation-model window BP. Raw scores and probabilities are computed only in the
later zero-backward selection phase. This keeps
the subsequent selection phase strictly free of backward work, without changing LP.

All 81 seed/model/task asset receipts must exist before selection. Reconstruction
uses a cache that raises on any missing gradient, before calling the model. Selection
is followed by another 81-receipt barrier before training. Receipts and resource
paths/identities are checked on resume; missing data cannot silently trigger BP
inside selection. Original is an explicit no-selection/no-training entry.

Selection files use `v7-compact-selection-v1`: atomic compressed NPZ, explicit full
identity, FP64 probabilities/raw scores, stable descending order for new arms, and
contiguous native-unit spans. Window objects are implicit array indices, not Python
singleton arrays. Packed source values and full window identity arrays are NOT
duplicated per arm. Reconstructed samplers use only the current seed's index and
pool_n=16284. Required positive support, full partition and pool containment are
validated. Artifacts and logs stay under the TMP run, never in Git. Keep all gradient,
CRB, probability and result artifacts. Only newly completed V7 cell checkpoint weights
follow inherited `completed_cell_cleanup`, after curves/results are durable; no old
V6 checkpoint is targeted.

## Numerical optimization and tests

New baseline scoring uses linear dot interpolation and the two-anchor Gram quadratic
for L2 norms: O(KD+W), not O(WD). Near-antipodal cancellation is recomputed from raw
linear vectors in bounded chunks. Constant-vector exact ties and endpoint values are
preserved. Reference averaging integrates integer-window interpolation coefficients
onto anchors, avoiding dense reference reconstruction. Ranking remains O(W log W).

Production source regression: **103 passed in 9.99s**, including 20 new V7 tests,
existing gradient/cache/selection/full-pool/Timer tests and CUDA numerical checks.
Tests include actual 65,136 -> 16,284 synthetic whole-trajectory partitions for all
three seeds/models, unique pool identity, stable IDs, overlap, unreachable subsets,
out-of-pool exclusion, seed0 artifact rejection for seed1, budget caps/endpoints,
signed/zero/nonfinite scores, reference weights, dense-oracle agreement, constant
ties/cancellation, atomic compact roundtrip, donor bit preservation, cross-seed cache
hits, missing-row fail-closed selection and isolated-validation tie-breaking.
LP's split prepare/select path produces bitwise identical native scores/probabilities
to the original end-to-end routine, with `Tensor.backward` forbidden during selection.

Synthetic CPU2-thread smoke, 1,000,000 windows / 15,000 anchors / 4096 dimensions:
all three scores **0.508s**, one frozen ranking **0.126s**, peak process RSS
**1,036,180 KiB** (includes synthetic arrays and a separate dense oracle).
Probability sum 0.9999999999999997, strictly positive support, zero BP.
Bounded 10,000-window oracle norm expansion 0.700s; analytic all-three scoring 0.00282s.
This is a kernel measurement, NOT a formal throughput or end-to-end ETA claim.

## Resource planning and known gates

Initial source preflight found an empty linked source shards directory despite intact
Arrow indexes. Kimi restored shards from intact native profiles and checked index/meta
consistency. Subsequent engineering preflight verified all 189 indexed shards present.
No missing rows were skipped or alternate sampling used. Feasible capacity and actual
pools have now been constructed once at `main-v7-20260923/inputs` for reuse:

| Pool | Equivalents | Trajectories | Legal windows |
| --- | ---: | ---: | ---: |
| Feasible | 65,136 | 4,618 | 36,122,806 |
| seed0 | 16,284 | 1,125 | 9,038,454 |
| seed1 | 16,284 | 1,193 | 9,025,570 |
| seed2 | 16,284 | 1,132 | 9,036,387 |

Seed union: 2,659 trajectories; pairwise trajectory overlaps 285 / 273 / 294.
Per model, source union RAGS K=318,737; reference K=74,767 over nine tasks.
Actual per-seed source K=136,145 / 136,014 / 136,102. The sparse RAGS+uniform
union is at most 787,008 rows per model before anchor overlap/donor hits: at most
6.45 decimal GB of BF16 gradient values per model, **19.34 GB across models**.
These are conservative newly allocated-value ceilings, NOT measured new BP counts.
No V7 model gradient or training work was launched by Codex.

Raw storage bounds before measured compression/hits: all three seeds have at most
28,138,752 legal windows combined. New score/probability/order arrays across three
models and nine tasks are at most 54.70 decimal GB; LP score/probability arrays add
12.16 GB. These are uncompressed ceilings, not measured incremental allocations.
Additional gradient/CRB resources and checkpoint headroom are audited separately.
The actual-pool conservative total (including FP64 CRB representation upper bounds)
is **176.88 decimal GB**, versus **11.00 GB free** at input admission. This is an
uncompressed conservative resource ceiling, not a claim that actual compressed or
cache-hit-adjusted usage equals it. `input_preflight.json` records `complete=true`
for constructed inputs but `disk_ready=false`; workers refuse to start gradients.
No old results/caches
may be cleaned without user approval. Compression/hit measurements or added space
are needed before formal resource admission. Re-running input preparation reuses the
validated feasible snapshot and existing manifests rather than rebuilding them.

Existing V6 `train_eval.py` actually reports the final epoch; it does not implement
isolated-validation stopping. V7 adds a separate Timer stopping wrapper, but it
requires an approved validation resource, not an invented split of full-history
reference or test. `timer_validation_manifest` must use
`v7-timer-isolated-validation-v1`, C=480/H=96, rule
`min_validation_mase_earliest_epoch_tie`, per-task NPZ paths, declared entities and
native provenance. Arrays need finite 576-point windows and positive finite scales;
validation entities must be disjoint from source, all task reference and test entities.
Validation receipts freeze those identities before training. Grid 0,.5,...,4 is
selected only by validation MASE, earliest epoch for exact ties, before test checkpoint
scores are evaluated. Missing/unapproved validation blocks training explicitly.

## Kimi launch sequence (after capacity/input approval)

From `/root/TS3`, with the model environment interpreters in `main_v7.json`:

```bash
export TS3_GRADIENT_CACHE_ROOT=/root/autodl-tmp/TS3/cache/grad-pool
export PYTHONPATH=/root/TS3/src
/root/miniconda3/envs/ts3/bin/python -m scripts.run_v7 --phase preflight
/root/miniconda3/envs/ts3/bin/python -m scripts.run_v7 --phase inputs
/root/miniconda3/envs/ts3/bin/python -m scripts.run_v7_suite --phase assets --parallel-models 1
/root/miniconda3/envs/ts3/bin/python -m scripts.run_v7 --phase audit-assets
/root/miniconda3/envs/ts3/bin/python -m scripts.run_v7_suite --phase select --parallel-models 1
/root/miniconda3/envs/ts3/bin/python -m scripts.run_v7 --phase audit-select
/root/miniconda3/envs/ts3/bin/python -m scripts.run_v7_suite --phase train --parallel-models 1
```

Increase concurrent models only after Kimi's measured GPU/memory admission. Runtime
`execution.json` may override per-model SelectionExecution fields, not scientific
config. Supervisor uses blocking process completion events, no polling loops.
Kimi inspects on its existing 30-minute scheduled wakeups: throughput, VRAM/RAM/disk,
cache hits/imports/new BP, phase receipts, ETA, logs and failures. Resume the failed
phase under the same identity after diagnosis; never delete/recompute completed
resources. Formal results remain pending until actual cells exist.
