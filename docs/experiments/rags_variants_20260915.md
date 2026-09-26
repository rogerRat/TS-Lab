# RAGS variants, 2026-09-15

The main full-pool arms are RAGS, Random, LESS, TSDS, LP-Mirror and LTSV;
original is the separate untrained evaluation. The old `trajectory_level`
arm is rejected by the production selector. The variants below are separate
callable selections, not additions to the main matrix.

`prepare_gradients` accepts `rags_traj` and `rags_uniform`. Both prepare the
corresponding complete RAGS reconstruction first using the same cache namespace.
`select_full_pool` accepts either name with its ordinary production arguments.

## Trajectory support and aggregation

RAGS-Traj uses the union of each trajectory's retained RAGS block-owned legal
starts, on both sides. Filtering happens before aggregation. The arithmetic
mean of the raw linear reconstruction is obtained by accumulating endpoint
interpolation coefficients over those starts, then dividing by retained window
count. Shared endpoints count once; cache lookup frequency adds no weight.
Only one additional vector per trajectory is retained in RAM. No interpolated
window vectors are persisted. Existing BF16 anchor caches remain the sole
gradient cache; the aggregate uses floating-point arithmetic without creating a
second gradient dataset.

Reference trajectories remain independent queries, with their original summed
retained-block window shares. Source trajectory mass uses retained legal count;
within a source trajectory sampling is uniform over that retained union (not
over discarded gaps). Near-zero reference aggregate raises `InfeasibleReference`
with the affected original mass, without dropping the reference. A near-zero
source aggregate has zero direction; its affected window count and identity are
reported. Thus aggregation-induced coverage changes remain explicit. The
frozen CRB mean threshold defines near-zero; no new threshold is calibrated.

## Actual K pairing, frozen before variant execution

RAGS-Uniform retains the same phase budgets, changes only
`error_placement='uniform'`, and runs the unchanged classification/split stage.
Its required anchors are acquired from the identity-matched shared cache, with
missing rows computed normally; adaptive RAGS anchors are not relabeled uniform.
The comparison is eligible for same-compute claims only when **every source and
reference trajectory** has exactly the same actual unique anchor count K.
Split queries count toward K. Aggregate total equality is insufficient.
There is no posthoc density search, hidden top-up or repeat-gradient padding.

Prepared assets expose `pairing` (per-side, per-trajectory phase and total K),
`source_trajectory_counts`, and `uniform` with its own blocks/results/counters.
Selection rejects unequal K by default. Explicit
`allow_unmatched_uniform=True` returns a diagnostic distribution labeled
`eligible_same_compute=False`; it does not authorize scheduling an equal-compute
training comparison. This flag must not be silently enabled by production.

These compact assets stay in memory. Saved standard probability artifacts carry
aggregation/support and pairing metadata; no new cache format is introduced.
Method-independent required gradient cost, current cache delta, and incremental
aggregation/scoring time must be reported separately by execution orchestration.

## Observed gradient cost (A7)

`gradient_timing.json` lives beside the BF16 window shards. It contains small
cumulative timing records keyed by explicit model/checkpoint, projection,
precision protocol, C/H, engine type, device/host, Torch version and execution
configuration. It contains no gradients or hashes. The single cache writer
publishes it atomically after each shard and on close. Each actual
`batch_gradient_sketch` call through its host NumPy return contributes elapsed
time and unique newly computed windows; hits contribute neither. The record
also reports observed batch-size frequencies. Reopening under different
execution settings reuses gradient data but cannot reuse the old latency mean.
Missing measurements remain unavailable; timing never requests extra gradients.

`PreparedGradients.method_costs` records independent source/reference K and each
method's preparation call boundaries, including elapsed gradient calls and the
remaining CRB/cache/aggregation overhead. LESS preparation precedes RAGS, followed
by Uniform if requested; new-cache increments depend on that disclosed order.
RAGS-Traj reuses the complete RAGS preparation record, including compact aggregate
construction, rather than claiming zero cold gradient requirement. Shared cache
initialization/finalization remains outside method preparation and should be
reported once at the union level.

`SelectionMetrics.measure(..., gradient_cost={**assets.method_costs[arm],
'benchmark': assets.cache.gradient_benchmark()})` reports estimated method cost
as `K_required * observed_seconds_per_window + method preparation nongrad +
measured selection body`. It does not add the full preparation elapsed time
again. Actual new gradient count and observed preparation are separate fields;
never sum the reused union records over arms. A saved selection cache load does
not supply a new cold selection measurement and leaves the saved original cost
unchanged. This is an estimate from the observed execution/batch mixture, not a
claim that the method was rerun cold or that cache I/O was measured cold.
