# V6 compute optimization — Codex implementation, local tmux Kimi operations

2026-09-21. User owns the frozen V6 protocol in `_exp-0907.md`; seed barrier and asset-first priorities remain in force. Kimi owns all job launches, restarts, GPU benchmarks and scheduled inspections (no polling). Codex implements and tests code.

## Delivered changes

- `assets/gradient/source_crb_cache.py`, `blocks.py`, `prepare.py`: retain source CRB results once per trajectory, including raw block means, directions, legal positions, trajectory anchor averages, phase counts and split audits. Later tasks load the same source geometry instead of replaying adaptive CRB. References remain task-specific.
- `assets/gradient/cache.py`: shard numbering uses an initialized monotonic counter. It no longer rescans every existing shard on every write (quadratic total numbering overhead becomes linear initialization plus constant-time increments).
- `rags/execution.py`, `gradient/base.py`: `gradient_vjp_batch` exposes independent Jacobian rows per backward, default 0 preserves adapter default 4. It does not change training batch, gradient loss or averaging. Existing OOM batch reduction remains. TimerXL `gradient_batch` is still capped at 32; `gradient_workers` does not enable processes for this adapter.
- `scripts/benchmark_native_vjp.py`: a synthetic TimerXL 480→96 CUDA benchmark for Kimi. Reports throughput, actual effective execution, peak allocated memory, loss error, relative projected-gradient L2 and minimum cosine against independent single-window gradients. No formal data, results or gradient caches are read or rewritten.

## Storage and reuse

New artifacts: `<run>/grad_cache/<model-fields>/source-crb/fields-N/cache_manifest.json` and `trajectory-<fragment-id>.npz`. Field identity includes the existing full gradient manifest (model/revision, pool/protocol, projection, precision), complete CRB config, C/H/stride, ordered fragment ID/entity/length/global offset. No content hashes or pickle. Files are atomically published under the existing model gradient-cache writer lock. Different field identities create a separate readable numbered directory; existing resources are kept.

An NPZ holds means and directions separately, offsets/positions, aggregate positions/mean and JSON counts/audits. No anchors or model weights are duplicated. Memory serialization is bounded to one trajectory, not the whole source pool. Approximate added vector storage is `blocks × projection_dimension × 8 bytes`, plus window positions and one aggregate vector per trajectory. Keep these derived resources with gradient assets; do not clean them up with training checkpoints.

Existing gradient shards remain reusable without migration. A source cache absent from old runs is built using cached gradient rows; only genuinely missing windows require model work. Completed trajectory files bypass CRB and gradient lookup entirely on restart. Progress within the current unfinished trajectory group remains recoverable through existing gradient shards; no claim of per-gradient transactionality beyond existing shard boundaries.

## Validation and measured limits

Final production-path validation: 68 tests passed in 3.54 s; benchmark script compilation and git diff whitespace checks passed. CUDA benchmark is supplied but has not been run by Codex.

CPU tests compare exact array equality for raw means, directions, positions and aggregate means; phase budgets/audits; stride 1 and 3; short unusable fragments; partial restart; field isolation; warm loading with a gradient function that raises if called. Existing gradient-cache, TimerXL V6, CRB, multi-trajectory and training-cost tests are included. VJP settings 1/2/4/8/16 match independent-window gradients on the CPU fixture within rtol 1e-6 / atol 1e-7.

Source CRB CPU fixture (4 × 30,000-point trajectories, 12-dimensional synthetic gradients): replay 0.1505 s, warm load 0.0276 s, 5.44× for that phase only. Formal 4096-dimensional disk loading, GPU throughput and total ETA are unmeasured. A coverage-list optimization was evaluated but NOT shipped because its measured speedup was unstable (0.73–1.07×).

## Kimi deployment and tuning

Source files are installed in `/root/TS3`, on existing `rags`, without commit/push. Existing loaded workers do not automatically reload Python modules. Keep the running job advancing; arrange a safe checkpoint/task boundary to launch workers with the new source, preserving durable gradient shards and all result files. Never restart a completed experiment to obtain this optimization.

For GPU tuning when resources permit, use the actual existing local pretrained path:

```sh
PYTHONPATH=/root/TS3/src /root/miniconda3/envs/ts3/bin/python -m scripts.benchmark_native_vjp --model-path EXISTING_TIMERXL_MODEL_PATH --samples 32 --batch 32 --vjp 4 8 16
```

This benchmark is synthetic, not a formal rerun. Retain the default until real CUDA numerical differences and throughput have been inspected. A faster candidate must satisfy the project's native-window numerical policy; report raw comparisons rather than silently accepting a tolerance. Then Kimi can set `gradient_vjp_batch` in `asset_execution_override.json` at a task boundary. Choose based on windows/s and headroom for concurrent required models, not allocated VRAM alone. Record effective batch, peak memory and throughput; update ETA from remaining uncached work. Do not assume that `gradient_batch=256` or `gradient_workers=2` makes TimerXL faster.
