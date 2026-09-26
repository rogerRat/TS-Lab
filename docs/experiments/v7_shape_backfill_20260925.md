# V7 midpoint+slope RAGS supplement — 2026-09-25

User-approved after the three exploratory two-seed tasks finish. This is a new
scientific arm family and new result namespace, not an edit to any original V7
selection or cell. Ten tasks = the original seven plus Wind, SG_PM25, Coastal;
three models, seeds 0/1, four new arms: 240 training cells. Seed2 is forbidden.
The execution order is a strict global arm barrier: finish and verify all 60
cells (ten tasks, three models, two seeds) of `rags-shape-expSum` before any
`linearSum` training; then finish all 60 `linearSum` cells before
`zeroClipSum`, then `maxSum`. Three model workers may run in parallel inside
one arm, each seed 0 then seed 1, all parent tasks in queue order. Shared
feature/score/selection preparation may calculate the four aggregates together
once per task, but no later arm trains before the preceding arm is globally
accepted. The first-ranked arm is `rags-shape-expSum`; `linearSum`,
`zeroClipSum`, and `maxSum` are matched controls. Existing orig/rand metrics and full curves are
field-verified and relabeled with provenance; neither is re-evaluated/retrained.

## Mathematics and frozen weighting

For each existing retained CRB block, read its two original projected gradient
anchors from the frozen gradient cache. No new backward pass is allowed. Set
`m=(gL+gR)/2`, `d=gR-gL` and rescale the physical block to `u in [-1/2,1/2]`,
so `g(u)=m+u*d` regardless of how many legal starts that block owns. Its
integrated squared energy is `E=||m||²+||d||²/12`. The unit shape is
`z=[m,d/sqrt(12)]/sqrt(E)`; for exactly zero energy, `z=0`.

Two unit shapes have integrated squared distance `||zi-zj||²`. The base
similarity is `s=zi·zj=1-distance²/2` when both energies are nonzero; a
zero-energy shape scores zero by the dot-product definition. Thus constant
blocks reduce to the previous midpoint cosine, opposite slopes remain
opposite, and gradient magnitude cannot dominate merely by scale. Squaring
the signed similarity itself is forbidden. FP32 stored shapes are accumulated
in FP64 dot tiles, as in the existing RAGS opinion scorer.

For each source block and reference block pair, apply the existing four opinion
rules exactly: `linearSum=Σαs`, `zeroClipSum=Σαmax(s,0)`,
`expSum=Σαexp(s)` with inner temperature 1, `maxSum=max(αs)`.
Reference weight `α=block_legal_windows /
(reference_trajectory_legal_windows * number_of_reference_trajectories)`.
Then rank source **blocks equally** with the existing frozen exponential rank
curve, multiply each block's rank mass once by its owned legal-window count,
and normalize. There is no value softmax, gating or epsilon support smoothing.
Block length never enters the shape distance; it enters reference weighting
and source prior only once each. Training still samples uniformly inside the
selected CRB block. Optimizer, epochs (Chronos3, Timer/TimesFM4), half-epoch
evaluation, seed pools and all other frozen V7 settings are unchanged.

## Assets, artifacts and costs

`src/rags/experiments/v7_shape.py` reads cache rows at the original physical
endpoints and writes an explicit-field NPY shape cache. Source features are
shared across all ten tasks by model/seed under
`main-v7-20260923/shape_backfill_cache/source/seedN/<model>`; endpoints are
compared exactly on every reuse, never hashed. Reference features are shared
across seeds under each parent run's `shape_backfill_cache/reference/<model>/<task>`.
The original gradient cache manifest, source-pool manifest, model identity,
CRB config, native reference input identity, endpoint positions and shape
version are checked. These caches are reusable gradient-derived resources and
may be retained. Raw gradients are never copied to the supplement run.

For each parent run, new results live only at `shape_backfill_20260925/`.
`shape_config.json` and `shape_protocol.json` freeze the amendment;
`inputs/` and `models/` are validated links to immutable parent resources.
Per-cell `shape_scores/<task>.npz` stores the four shared score arrays before
selection publication, permitting crash recovery without rescoring. Compact
block selection NPZs and `shape_select/<task>.json` record zero new BP,
individual selection bytes, scoring time, source/reference feature cache hits,
feature bytes, required logical BP, support coverage and parent provenance.
The `shape_select/`, `shape_complete/`, and global `arm_phases/` receipt
directories are created on first atomic publication; a fresh sidecar run needs
no manual directory preparation.
`shape_complete/<task>.json` records full-curve/result acceptance and total
selection storage; native trained cell JSONs retain training/evaluation time.
Temporary checkpoints obey V7's existing completed-cell cleanup.

Six source shape matrices (three models × two seeds) are about 26 GiB if each
has ~130k blocks × 8192 FP32 dimensions. Reference shapes, score caches,
compact selections and training headroom add several tens of GiB; queue
admission requires at least 75 GiB actual free space, then measured growth is
monitored. These are conservative estimates, not reported physical costs.
No data/model/asset hash is calculated. A later observational report must
separate shared feature/gradient assets from per-arm selection storage and
never add the shared four-arm score time four times.

## Event-driven queue

After the existing extension supervisor exits, the new queue verifies all
three `task_queue_complete.json` markers and all original two-seed cells.
`scripts/run_v7_shape_backfill.py --parent ... --queue-pid <verified PID>`
waits on the original supervisor's pidfd, not a polling loop. If it exits
without all 360 extension cells, supplement startup fails closed. Otherwise
three model workers run in parallel (Chronos GPU0; TimerXL+TimesFM GPU1)
within each arm phase, each in seed→task order. The orchestrator verifies all
60 result identities/curves before launching the next arm. Completed arm cells
are skipped by identity; failure leaves existing results/cache intact and is reported rather
than blindly retried. Kimi supervises throughput, GPU/host memory, disk and
the final 240-cell audit.
An execution-only `run_model` seed/task filter can fill an otherwise idle GPU
with disjoint tasks **within the currently active arm**. It leaves the frozen
selection/science identity unchanged; admission must verify zero-BP selection
receipts and use an event guard to stop the helper before the primary worker
reaches those tasks. It must never start a later arm across the global barrier.
If a phase was fully completed before a supervisor restart, the resumed queue
re-audits all its cells and publishes the missing `arm_phases/<arm>.json`
receipt before advancing; it never retrains that phase merely to obtain a marker.
