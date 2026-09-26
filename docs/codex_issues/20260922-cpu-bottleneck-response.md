# CT CPU bottleneck: verified causes and tested changes

2026-09-22. Codex owns implementation/tests; local WSL `tmux kimi` owns formal execution, reloads and 30-minute scheduled inspections. No production process was stopped, no formal experiment restarted, and no existing gradient or selection artifact was recomputed or migrated.

## 1. Chronos source CRB is a partial cold fill, not a field mismatch

Read-only evidence from `main-rank-volume-20260921/rounds/seed0` and `/root/autodl-tmp/TS3/cache/grad-pool`:

- Chronos and TimesFM each have only `source-crb/fields-0`, now with 1082/1082 trajectory NPZs.
- The Chronos cache model fields equal `students/chronos-base/model.json` after the existing ckpt/revision defaults; pool fields equal the actual pool manifest; source CRB gradient identity equals its enclosing gradient cache; CRB config equals the current run config. All four comparisons are true for TimesFM as well.
- Chronos log after restart: 1/1082 at 0 s and 251/1082 at 3 s, then 501 at 5034 s, 751 at 12087 s and 1001 at 19486 s. Thus the first cached group DID hit; missing subsequent groups were filled. 12087 s is 201 minutes, not 84 minutes.
- `time_nem_demand_5t__gradient_assets.json`: Chronos 253880 newly computed windows, preparation 21267.13398 s, gradient calls 21154.66905 s (99.47%), total cached 1362361. TimesFM: 79697 newly computed windows, 3235.46565 s preparation, 3188.25747 s gradient calls.
- The existing log contains KeyboardInterrupt stacks INSIDE Chronos forward/backward and Count-Sketch, not an identity rejection. Source CRB timing encloses these model calls; it is not pure Python CRB timing. Low GPU utilization alone cannot identify the split between CPU launch/projection overhead and GPU execution inside those calls.

New diagnostics print the matched source cache path, differing top-level identity fields if any, and warm/missing counts for each group. No cache fields or reuse semantics changed. The remaining tasks can use the now-complete same-model cache; TimerXL assets cannot substitute for Chronos assets.

## 2. Requested batch is not effective native gradient batch

Both Chronos timing records show `gradient_workers=1`. Chronos BF16 explicitly selects the native single-window path when workers=1; BF16 vmap is intentionally excluded by existing numerical policy. Increasing input request batch from 4 to 8 does not turn that into a batched backward.

Under the batch-8 override, recorded input batches are: 8 -> 28503 calls, 5 -> 2471 calls; the modal size is 8, not 5. The input batch-4 record has 85708 calls of size 4. Sparse tails and adaptive CRB dependencies still create smaller requests. Increasing input batch cannot invent missing independent work.

`_gradient_batch_limit` persists only in Chronos FP32 vmap, which this BF16 production run does not use. NativeProbe resets its local OOM batch size on each invocation; a new regression test forces an OOM, observes 4 -> 2, then verifies the next invocation returns to 4. No speculative BF16 "OOM recovery" algorithm was introduced.

New `gradient_timing.json` record field `execution_batches` counts backend/effective batch/workers/VJP/OOM labels reported by the engine. Existing `batch_sizes` remains the input-request histogram. Older records lack effective-execution observations; do not infer those retroactively.

Kimi's relevant existing runtime control is Chronos `gradient_workers`, not TimerXL VJP. Evaluate 2 then 4 independent native workers subject to actual shared GPU memory and throughput, at a safe task boundary. Codex has not changed those runtime settings or stopped the active workers.

## 3. Durable sparse-write coalescing

`cache.py` now combines small writes into a maximum 64-row current-session NPZ tail per dataset. Each adaptive round atomically publishes its tail BEFORE returning gradients; there is no unsaved in-memory gradient queue. Bulk shards retain the existing 1024-row maximum. Only newly created, unpublished tails can grow; reopening a cache never rewrites its old shards. Publishing an index seals all tails, preserving immutable reader snapshots. Existing shared-root support `TS3_GRADIENT_CACHE_ROOT` is preserved.

Rows retain exact dataset/fragment/start identities and BF16 bits, losses and order. Shard format remains `window-gradients-bf16-v1`; no format migration, existing-file compaction, hash, deletion or new directory identity. Appended rows invalidate/refresh the decoded entry. The newly published BF16 data seeds the bounded decoded LRU directly instead of immediately rereading its ZIP. Single-batch materialization avoids creating a thread pool with no following batch to overlap. Multiple batches retain bounded prefetch.

Synthetic 4096-dimensional 256 single-window write/read rounds: 256 shards -> 4, 2370560 -> 2106464 final bytes, 0.2940 -> 0.2614 s in the final test. Rewriting a small tail increases transient write bytes; the 64-row bound limits this cost. These numbers are a microbenchmark, not a formal-run ETA guarantee.

## 4. Share work between selection arms before duplicating workers

`reference_scores_many` computes each FP64 cosine tile once and feeds only the missing arms' requested aggregations. It retains the original matrix dimensions, CPU transforms/reductions and accumulation order. Source GPU upload and cosine GPU-to-CPU transfers are also shared. The worker plans missing aggregations separately for block and trajectory representations. Scores live only in that task's PreparedGradients instance, with no cross-model/reference/task/checkpoint reuse or extra BP. Existing final selections remain authoritative and are loaded without recomputation.

The scientific metadata, scores, exact ties, probabilities and sampling semantics are unchanged. `SelectionCost.notes` explicitly marks shared requested aggregations and score-cache hits: the first consumer is charged the shared scoring pass, subsequent consumers only their incremental work. Compare total group execution costs rather than treating the first arm as a standalone cold benchmark.

Synthetic CPU comparison against the saved original implementation (256 refs x 2048 sources x 1024 dimensions, two CPU threads): four-arm matching 3.41–3.89x faster across three repetitions, with identical scores AND audits. Full selection also includes ranking, probability construction, NPZ compression and TSDS/LP-Mirror; no comparable full-stage or total ETA gain is claimed.

Independent concurrent selection processes would multiply source representation memory, uploads and repeated matching; this implementation shares the dominant common work without adding GPU contenders. Read-only selection decompression/prefetch and CPU compression remain possible follow-ups, but have not been claimed as measured bottlenecks. Missing CJK fonts or py-spy do not prevent these field/counter/fixture diagnostics; neither was installed.

## Validation / handoff

Final deployed-production-path result: **59 passed in 8.28 s**, including CUDA score-equivalence tests; `git diff --check` passed. Tests use small synthetic fixtures and do not access formal experiment caches. No commit or push.

Regression tests cover sparse file count, immediate durable visibility, crash restart with stale index, reader snapshot sealing, failed atomic replacement, effective-vs-requested batch records, native OOM recovery, exact tiled scores/ties/probabilities and one shared score pass across arms. Existing cache, CRB, gradient cost, worker/control, TimerXL and resampling tests are included. New tests live at `tests/test_ct_cpu_optimization.py`.

Development overlay: `/root/TS3/temp/ct-cpu-opt`; synthetic benchmark: `benchmark_ct_cpu.py` there; pre-change backup: `before/`. Formal resources are untouched. Kimi should load the deployed code only at its safe execution boundary, continue scheduled inspections (no polling), and report effective workers, uncached windows, warm/missing source trajectories and observed windows/s. No request to rerun completed cells or rebuild reusable caches.
