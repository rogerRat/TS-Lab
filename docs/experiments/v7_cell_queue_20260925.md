# Nonblocking cell queue — 2026-09-25

User requested GPU admission by freely available work, not whole-task waiting.
`python -m rags.experiments.v7_queue --config configs/rags/main_v7_curve4.json
--run RUN --student MODEL` scans configured task/arm order, seeds 0 then 1 only.
Set CUDA_VISIBLE_DEVICES and the model's existing interpreter externally.
Two workers of the same model may claim different arms of the same task.
Busy cells are skipped immediately. No polling or blocking task acquisition.
Completed native cells are never trained again. Model weights reset to the same
pretrained base before every claimed cell; original asset/training identities,
pool, sampler, optimizer, schedule and checkpoint lifecycle remain unchanged.

The queue currently requires durable original evaluations for all admitted tasks,
as present in the existing seven-task V7 matrix. Rand must finish before another
arm of that task is claimed, because native reports require its result.
It does not independently prepare assets, add methods or select new datasets.

Safety: shared model leases still exclude gradient writers. Shared nonblocking
task leases permit queue concurrency but skip legacy exclusive task workers.
Per-cell nonblocking claims prevent duplicate training/overwriting Resume files.
Files live at RUN/cell_queue_locks/MODEL/seedN/TASK__ARM.lock; never unlink them.
Per-process status is atomically written under logs/queue_PID.json. Native cells,
reported_cells, selections and temporary resumes remain at their original paths.
Native task-level completion markers are not falsely claimed by a partial worker.
If needed, an ordinary worker can later materialize full-task markers by reusing
the completed cells. A queue pass exits when no unclaimed work remains; schedule
subsequent ready work through process completion events, not repeated scans.

Migration: do not mix an unmodified exclusive-model worker with queue workers.
Finish its current cell and stop it before queue admission; do not delete locks.
Existing Chronos main PID 222664 was found waiting on GPU1 helper 267504's task.
The one-shot inotify handoff waits for durable cell output before terminating
only that helper. This is a transitional boundary wait, not the new policy.
Seed2 remains forbidden even if old supervisors attempt it; do not auto-retry.

Validation: nonblocking competing-process claim, shared task concurrency versus
legacy exclusion, unchanged asset/training/settings identities, and existing
seed admission tests. GPU throughput acceptance is recorded separately after
live adoption; no assertion of permanent 100% utilization.
