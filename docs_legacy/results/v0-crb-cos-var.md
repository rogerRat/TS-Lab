# v0-crb-cos-var results

Last modified: 2026-11-10

Status: completed on 12 M5 FOODS_1 trajectories. This experiment compares Uniform and native CRB partitions at equal anchor budgets. The anchor count is K=max(2,ceil(N/q)); both endpoints count toward K. Uniform uses K rounded evenly spaced anchors. CRB uses the frozen v5 policy, with K-2 discretionary anchors divided as evenly as possible among coverage, error-risk, and recursive minimum-kappa stages.

All-window ChronosTiny LoRA gradients use the same warmed checkpoint, fresh model-seed-0 LoRA initialization (r=8, alpha=16, dropout=0, q/v targets), context 512, prediction 64, and stride 1 as v0-crb-reconst-fine. Each gradient is immediately projected to 4096 dimensions by ParameterHashProjector random sign CountSketch with seed 42. No dense LoRA gradients or full-trajectory projected vectors were written to disk; the projected trajectory matrix was held in memory per process for both matched policies and released after that track.

For each method and q, adjacent anchor points define intervals. The strict interior excludes the two anchors. The reconstructed representative direction is the direction of the arithmetic mean of the two known endpoint projected gradients. Each block score is the mean 1-cosine between this representative and the true projected gradients strictly inside the interval. Lower scores indicate less directional dispersion. q=1 has no strict interior and is undefined. The main aggregate is the equal-track mean of each track’s equal-block mean. The window-weighted aggregate is also reported because CRB blocks have variable sizes. Paired wins count tracks with lower equal-block mean for the named method.

| q | Uniform equal-track block mean | CRB equal-track block mean | CRB wins | Uniform wins | Uniform window-weighted | CRB window-weighted |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | N/A | N/A | 0 | 0 | N/A | N/A |
| 2 | 0.183529 | 0.180344 | 5 | 7 | 0.183630 | 0.193894 |
| 4 | 0.263465 | 0.265794 | 5 | 7 | 0.263509 | 0.294201 |
| 8 | 0.351978 | 0.345769 | 9 | 3 | 0.352008 | 0.382656 |
| 16 | 0.448220 | 0.433205 | 9 | 3 | 0.448221 | 0.470739 |
| 32 | 0.549211 | 0.532104 | 8 | 4 | 0.549049 | 0.590552 |
| 64 | 0.680103 | 0.643582 | 11 | 1 | 0.680067 | 0.700171 |
| 128 | 0.798180 | 0.756923 | 11 | 1 | 0.798204 | 0.804486 |

By equal-block mean, CRB is lower at q=2,8,16,32,64,128; Uniform is lower at q=4. Track wins differ at q=2 and q=4, where the equal-track averages do not represent a majority of tracks. By interior-window-weighted mean, Uniform is lower at all nontrivial q. The two aggregation rules answer different questions: equal-block weighting gives each interval equal influence, whereas window weighting gives longer intervals more influence. CRB creates variable interval lengths, so both are material. The first four tracks favored CRB on the equal-block average at every q>1, but the 12-track q=4 outcome reverses; the larger sample exposes track and aggregation variability.

The result is descriptive for 12 selected FOODS_1 series and one fixed, freshly initialized LoRA model state. At q=2, every interval has one strict-interior vector, so the mean over two known endpoints can be a poor representative despite the simple interval definition. q=1 has no target points. No cross-family or trained-adapter conclusion follows.

Validation: 12 track completion markers and the aggregate .complete exist; trajectory_summary.csv contains 192 rows (12 tracks x 8 q values x 2 methods); aggregate.json contains 16 method-budget rows; no dense gradient cache is present. Peak monitored GPU memory was about 0.9 GiB, cgroup memory about 15 GiB of 62 GiB, and two track processes ran concurrently. The earlier uniform-only diagnostic is excluded and preserved separately under outputs/train/v0-crb-cos-var-uniform-preview/.

Artifacts: outputs/train/v0-crb-cos-var/aggregate.json, trajectory_summary.csv, and per-track block_metrics.csv and summary.json. Config: configs/rage/v0-crb-cos-var.json. Entry and queue: scripts/v0_crb_cos_var.py and scripts/v0_crb_cos_var_queue.py.
