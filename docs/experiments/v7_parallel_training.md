# Independent-task V7 training admission

Scientific config, batches, optimizer, seeds, curve and asset identity are unchanged.
Training workers may opt into `--parallel-train`: the existing model lock becomes
shared **only for this opt-in training invocation**. Assets/select and legacy/default
workers retain an exclusive model lease; an old worker therefore blocks new parallel
workers instead of silently becoming unprotected. Never delete lock files.

All updated training workers take an exclusive `(model, seed, task)` lease for the
entire task including orig, all arms, reporting and completion. After acquisition,
the completion marker is rechecked so a second worker skips completed work. Blocking
flock is event-driven, not polling. A failure releases the lease; native per-cell
result/resume verification remains unchanged. Different tasks use different outputs.

`--train-task ACTIVE_DATASET --parallel-train --seed N` runs one task. The model-local
prior-seed and select/asset gates still apply. No filtering of scientific configuration
or identity is used. Filtered workers write `state__TASK.json`, not the main state file.

Deployment on two GPUs: at an evaluated cell boundary, replace the old Chronos worker
with the same invocation plus `--parallel-train` on GPU0. Then start a GPU1 helper with
the same seed/config/run and `--parallel-train --train-task time_port_activity_d`.
TimesFM stays running on GPU1. Admit only if measured memory fits; compare total
completed training throughput, not GPU occupancy alone. Do not start before migrating
the old exclusive Chronos worker. No worker should be killed mid-cell for this change.

IMPORTANT: disarm/update existing PID-exit seed continuations before replacing the
old PID. The main Chronos worker eventually waits on the helper's task lease, verifies
completion and skips it. Before launching the next seed under a default exclusive
model lease, wait for BOTH processes to exit; otherwise the default nonblocking model
lock may reject launch. Failure must be reported; do not blindly advance/retry.

This removes the old one-process-per-model execution limitation without concurrent
gradient-pool writers. It does not assert host bottlenecks are solved by DataLoader
workers: current sampling CDF and NumPy window gathering are already vectorized.
