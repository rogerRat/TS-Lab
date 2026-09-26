# Ref anchor budget scan

`python -m rags.experiments.anchor_budget_scan --out <data-disk-output> --model-path <local-TimeMoE-checkpoint>`
samples the first five tasks in `shared_tasks_10.json`, using one RNG seeded 20260914
and up to three complete reference trajectories per task. All legal 512+64 windows
are scanned independently, using the existing BF16 TimeMoE gradient asset protocol.
`--plan-only` validates sample selection and endpoint window identities without a model.

Outputs are retained on the data disk: `sample_manifest.json` records source paths,
identities, lengths, task order and random seed; `cache_location.json` records the
model identity and the asset path; `scan_status.json` is an atomic progress snapshot.
`grad_cache/` uses the existing append-only uint16 BF16 NPZ shards and explicit-field
manifest, with one writer and atomic index snapshots. Each 256-window chunk is saved
before progress advances. Resume reuses matching shards without recalculating them.
No hashes or training checkpoints are created. These caches are retained for CPU
budget simulations; they must not be removed after the scan.

The experiment criterion is per-task mean per-window reconstruction cosine >=0.90.
Production block-constant directions and linear interpolation diagnostics are reported
separately by the companion simulation. This is a reconstruction study on sampled Ref
trajectories, not a claim about downstream training performance.
