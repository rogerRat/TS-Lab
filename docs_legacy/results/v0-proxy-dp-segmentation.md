# v0-proxy-dp-segmentation

Last modified: 2026-11-10

## Question and method

Does an exact proxy-variance partition improve true-gradient interval metrics over an equal-count uniform partition, and does weighting the proxy objective by interval length change the tradeoff? The run used the four longest retained source trajectories (ordinals 01, 05, 06, 09; N=1388, 1393, 1393, 1394). Each method used K=ceil(N/64)=22 intervals per trajectory. Uniform intervals therefore have length 63 or 64. DP intervals have minimum length 16 and no maximum length.

Output-head gradients sampled every four starts were linearly reconstructed in BF16. The exact DP optimized either (a) sum_j V_j or (b) (1/N) sum_j |I_j| V_j, where V_j is the trace of the population covariance of reconstructed proxy vectors in interval j. Gram products used coordinate chunks with FP32 accumulation. True full-gradient metrics came from retained BF16 caches, read in four-start chunks with FP32 reductions. No new gradients were computed.

The interval metrics are equal-interval means unless identified as window-weighted. Direction dispersion is mean 1-cosine between each true gradient and the interval mean-gradient direction. Window-weighted direction dispersion weights each interval's direction dispersion by its number of starts. Values below are first averaged within each trajectory, then equally across the four trajectories. They are descriptive for this selected four-track sample.

## Equal-track mean results

| DP objective | Metric | DP | Uniform | Change vs uniform | Tracks with lower DP value |
|---|---|---:|---:|---:|---:|
| Unweighted | Equal-interval mean relative variance | 0.50311 | 0.67621 | -25.6% | 4/4 |
| Unweighted | Equal-interval mean direction dispersion | 0.29955 | 0.42977 | -30.3% | 4/4 |
| Unweighted | Window-weighted direction dispersion | 0.58440 | 0.42973 | +36.0% | 0/4 |
| Unweighted | Equal-interval mean absolute variance | 3.77871 | 9.49824 | -60.2% | 4/4 |
| Window-weighted | Equal-interval mean relative variance | 0.57174 | 0.67621 | -15.4% | 4/4 |
| Window-weighted | Equal-interval mean direction dispersion | 0.35724 | 0.42977 | -16.9% | 4/4 |
| Window-weighted | Window-weighted direction dispersion | 0.49229 | 0.42973 | +14.6% | 0/4 |
| Window-weighted | Equal-interval mean absolute variance | 12.56887 | 9.49824 | +32.3% | 0/4 |

Each DP partition uses exactly the same K=22 interval count as its uniform baseline. Across the four tracks, observed DP interval lengths ranged from 16 to 978 for the unweighted objective and 16 to 419 for the window-weighted objective; uniform lengths were 63-64. The unconstrained maximum permits substantial concentration of boundaries in selected regions.

## Per-track DP versus uniform

| Objective | Source ordinal (N) | Relative variance | Direction dispersion | Window-weighted direction | Absolute variance | DP interval length range |
|---|---:|---:|---:|---:|---:|---:|
| Unweighted | 01 (1388) | 0.5903 / 0.7107 | 0.3542 / 0.4601 | 0.6474 / 0.4602 | 4.1509 / 10.4981 | 16-978 |
| Unweighted | 05 (1393) | 0.4356 / 0.6548 | 0.2662 / 0.4111 | 0.5550 / 0.4110 | 3.3835 / 8.6703 | 16-475 |
| Unweighted | 06 (1393) | 0.5227 / 0.6476 | 0.3114 / 0.4045 | 0.5421 / 0.4043 | 4.1988 / 9.5251 | 16-506 |
| Unweighted | 09 (1394) | 0.4638 / 0.6917 | 0.2664 / 0.4434 | 0.5931 / 0.4434 | 3.3817 / 9.2995 | 16-942 |
| Window-weighted | 01 (1388) | 0.5832 / 0.7107 | 0.3596 / 0.4601 | 0.4982 / 0.4602 | 17.2182 / 10.4981 | 16-400 |
| Window-weighted | 05 (1393) | 0.5451 / 0.6548 | 0.3391 / 0.4111 | 0.4907 / 0.4110 | 11.7515 / 8.6703 | 16-419 |
| Window-weighted | 06 (1393) | 0.5518 / 0.6476 | 0.3388 / 0.4045 | 0.4604 / 0.4043 | 10.4985 / 9.5251 | 16-227 |
| Window-weighted | 09 (1394) | 0.6069 / 0.6917 | 0.3914 / 0.4434 | 0.5198 / 0.4434 | 10.8073 / 9.2995 | 16-232 |

Within each pair, the left value is DP and the right value is uniform.

## Interpretation

The unweighted DP lowered equal-interval relative variance, equal-interval direction dispersion, and equal-interval absolute variance on all four tracks. Its mean window-weighted direction dispersion was higher than uniform on all four. The window-weighted DP lowered equal-interval relative variance and equal-interval direction dispersion on all four tracks, while its window-weighted direction dispersion and absolute variance were higher on all four. Thus the variance objective improves some true-gradient criteria but does not dominate all reported criteria. Weighting the proxy objective by interval length did not make window-weighted direction dispersion lower than uniform in this sample.

The two objectives encode different preferences. The unweighted objective gives every interval equal influence, so long intervals can carry large per-interval variance without a proportional length penalty. The weighted objective scales each interval variance by its length. Neither objective directly minimizes true-gradient direction dispersion. These results do not establish a universal benefit beyond the four selected long tracks.

## Artifacts and validation

For each objective, interval_metrics.csv contains 176 interval rows (22 intervals x 2 methods x 4 trajectories), summary.json contains eight track-method rows, four per-trajectory checkpoints are present, and .complete is present. Validation confirmed exactly K intervals per method and track, exact contiguous coverage of [0,N), lengths at least 16, and finite reported metrics.

- Unweighted: outputs/train/v0-proxy-dp-segmentation-longest4-unweighted/
- Window-weighted: outputs/train/v0-proxy-dp-segmentation-longest4-window/
- Protocol: docs/EXP.md and configs/rage/v0-proxy-dp-segmentation.json
- Replay log: logs/v0-proxy-dp-segmentation/longest4.log

After result validation, all raw BF16 full-gradient source caches and their cache completion markers were deleted at the user's direction on 2026-11-10. Derived Gram matrices, proxy caches, and result tables remain. Replaying true-gradient metrics requires regenerating the full-gradient vectors.
