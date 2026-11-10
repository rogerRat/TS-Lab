# Experiment results

Last modified: 2026-11-09

## v0-active-value-rank

Completed 2026-11-09 on ten cached trajectories and 100 references. Batch 1 expanded the cross-track pool to 714-766 candidates per reference (mean 721.8) and found mean per-reference head/full cosine Spearman 0.7104. Direct head-proxy top-10-percent recall was 0.4023 with oracle regret 0.07525. In batch 2, a per-reference ridge residual model and boundary-uncertainty query policy reduced mean 10-percent regret to 0.06536, 0.05454, and 0.04192 at candidate full-gradient budgets 16, 32, and 64. Highest-proxy query regret was 0.07337, 0.06244, and 0.05148 at those budgets. Eight shared initial labels alone worsened direct proxy. At 5-percent selection and B=64, highest-proxy queries beat the boundary policy, which targets a 10-percent cutoff. The result is cached-score replay, not downstream training or measured online savings. Detailed protocol, track sensitivity, cost accounting, and limits: docs/results/v0-active-value-rank.md. Artifacts and .complete markers: outputs/train/v0-active-value-rank/batch-1/ and batch-2/.

## v0-directional-marginal-gain

Completed 2026-11-08 on 46 fixed 256-start blocks at four full-gradient anchor budgets. A fixed leftmost largest-gap midpoint adds one anchor; mean direction-error gains are 0.1072, 0.0407, 0.0144, and 0.0075 for n=2,4,8,16. Most average gain at n=2 and n=4 occurs away from the new anchor.

| Starting anchors | True dispersion versus starting error rho | True dispersion versus gain rho | Sparse proxy dispersion versus gain rho | Sparse proxy versus nonnew gain rho |
|---:|---:|---:|---:|---:|
| 2 | 0.616 | -0.074 | -0.002 | -0.012 |
| 4 | 0.701 | 0.142 | 0.380 | 0.370 |
| 8 | 0.547 | 0.256 | 0.235 | 0.212 |
| 16 | 0.395 | 0.060 | 0.056 | 0.032 |

Direction error is the mean of 1-cosine between each true and linearly reconstructed gradient; zero is aligned, one is orthogonal, and two is opposite. A positive one-anchor gain means that this average error falls; a negative gain means it rises. Pooled Spearman between starting error and total gain is 0.461/0.280/0.254/0.021 for n=2/4/8/16. At n=16, the block with the highest starting error (trajectory 07 block 0, E=0.5299) has negative total gain -0.00587 under the fixed added anchor. The true direction dispersion predicts current reconstruction difficulty much more consistently than marginal benefit. The sparse eight-point output-head dispersion correlates with true direction dispersion at pooled Spearman 0.703; its n=4 gain association is a moderate budget-specific signal, not evidence of a working allocation policy. A high-dispersion block can remain difficult while the fixed next anchor gives little improvement. Detailed protocol, per-track correlations, nonnew-point gains, and sensitivity checks: docs/results/v0-directional-marginal-gain.md. Raw output and .complete: outputs/train/v0-directional-marginal-gain/.

## v0-gradient-variance-decomposition

Completed 2026-11-08 using the 46 fixed 256-start blocks from v0-var-proxy-estimate, all cached full-gradient Gram matrices, and the same four-anchor linear reconstruction. The table reports pooled Spearman and controlled rank association after adjusting for mean gradient norm and track fixed effects. Reconstruction errors use 252 non-anchor positions per block.

| Predictor and target | Pooled Spearman | Controlled rank association |
|---|---:|---:|
| Vector variance versus mean gradient norm | 0.970 | Mean norm is a control |
| Vector variance versus amplitude variance | 0.569 | 0.663 |
| Vector variance versus direction dispersion | -0.222 | 0.578 |
| Vector variance versus magnitude reconstruction error | 0.638 | 0.286 |
| Direction dispersion versus direction reconstruction error | 0.701 | 0.713 |
| Relative vector variance versus direction reconstruction error | 0.639 | 0.605 |
| Vector variance versus direction reconstruction error | -0.324 | 0.431 |

The median amplitude-variance share of total vector variance is 0.176; only one of 46 blocks has a share over 0.5. This distinguishes the strong cross-block association with gradient scale from the within-block variance decomposition. The original pooled vector-variance versus reconstruction-cosine Spearman of +0.324 becomes -0.431 as a controlled rank association, whereas controlled Pearson is only -0.109. The eight-proxy variance versus cosine changes from +0.398 pooled Spearman to -0.202 controlled rank association, with controlled Pearson near zero (+0.001). These are descriptive results for the fixed ten-track sample.

Excluding the four exact anchors raises mean magnitude error from 1.083 to 1.100 and mean direction error from 0.638 to 0.648. Correlations are unchanged because each block excludes the same four zero-error positions. Full definitions, Pearson and per-track results, leave-one-track-out checks, and limitations: docs/results/v0-gradient-variance-decomposition.md. Raw and controlled scatter plots are `outputs/train/v0-gradient-variance-decomposition/figures/scatter_all.png` and `scatter_nonanchor.png` (also SVG). Block data, correlation tables, and `.complete` are in `outputs/train/v0-gradient-variance-decomposition/`.

## v0-grad-traj-level-norm-gap

Completed 2026-11-08 for the same ten retained trajectories and 12,836 legal windows using saved full/head Gram matrices. The norm of each trajectory's mean gradient vector ranges 0.7780-1.5050 for full gradients (1.9345x) and 0.0607-0.1515 for output-head gradients (2.4960x). Full/head track ordering is strongly associated (Spearman 0.8788). Tracks 00 and 07 are relatively high in both; 03 and 04 are relatively low. Mean per-window norms vary less across tracks: 1.4057x full and 1.4433x head spans. Full/head vector magnitudes occupy different spaces and should not be directly compared. See docs/results/v0-grad-traj-level-norm-gap.md for per-track values, definitions, and limitations. Output: outputs/train/v0-grad-traj-level-norm-gap/.

## v0-var-proxy-estimate

Completed 2026-11-08 by replaying ten retained trajectories with 46 complete 256-start blocks. Eight evenly spaced output-head proxies per block produced a population vector variance; four evenly spaced full-gradient anchors reconstructed all 256 positions by linear interpolation. Pooled proxy-variance Spearman correlations were 0.8538 with mean L2 error, 0.3977 with mean directional cosine, and 0.4970 with mean absolute norm difference. The positive cosine association means this proxy did not predict worse directional reconstruction in this sample. A completed all-256 full-gradient variance supplement gives pooled Spearman 0.9463 with mean L2, 0.3240 with mean cosine, and 0.6381 with mean absolute norm difference; full variance versus eight-proxy variance is 0.9169. The full-variance baseline is an oracle descriptive comparison, not an eight-call estimate. Using the four full-anchor gradients as one constant vector mean, true-variance Spearman becomes 0.9873 with mean L2, 0.2670 with mean cosine, and 0.8078 with mean absolute norm difference; corresponding eight-proxy values are 0.9126, 0.3374, and 0.6444. Mean L2/cosine are 3.5273/0.2519 for the constant mean versus 3.5127/0.3618 for linear reconstruction. The true-variance versus constant-representative squared-error association is partly guaranteed by the variance decomposition identity. Full protocol, per-track results, and limitations: docs/results/v0-var-proxy-estimate.md. Raw values and .complete: outputs/train/v0-var-proxy-estimate/.

## v0-crb-reconst-test (terminated; partial results)

User-terminated on 2026-11-08 because the available results were broadly clear and container resources were constrained. The replay completed all eight q groups on trajectory batches 00-02, partially completed batches 03-04, and did not start batches 05-09. In total, 429 configuration attempts, 326 unique metric files, and 28 matched uniform baselines were recorded. Per-batch q coverage and matched-baseline outcomes are in [the CRB result report](results/v0-crb-reconst-test.md); raw batch outputs and the regenerated plots are archived under `archive/v0-crb-reconst-test/`. The ten BF16 full-gradient caches remain intact in `outputs/train/v0-crb-reconst-test/gradients/`. The partial search must not be interpreted as a complete ten-trajectory result.

| Trajectory batch | Committed CRB attempts | Budget-group coverage |
|---:|---:|---|
| 00 | 128 | q=1,4,16,64,128,256,512,576: 16/16 each |
| 01 | 128 | q=1,4,16,64,128,256,512,576: 16/16 each |
| 02 | 128 | q=1,4,16,64,128,256,512,576: 16/16 each |
| 03 | 29 | q=1: 16/16; q=4: 13/16 |
| 04 | 16 | q=1: 16/16 |
| 05-09 | 0 | Not started |

Per-batch median-error comparisons against matched uniform baselines are in [the detailed report](results/v0-crb-reconst-test.md).

## v0-grad-mech-l2-norm

Completed 2026-11-08 as a ten-trajectory supplement to v0-grad-mech-test using retained BF16 full-gradient caches and their full-coordinate Gram matrices. Across 12,826 lag-1 pairs, relative L2 distance normalized by the mean gradient norm has pair-weighted mean 0.6372, median 0.7620, and p95 1.2411. The ten-trajectory mean of trajectory means is 0.6462. All 575 lags and 30 direct cache pair checks passed validation. The empirical local relative-smoothness hypothesis is contradicted in these ten trajectories: despite 99.83% lag-1 span overlap, the mean of the per-pair gradient-difference norms divided by each pair's mean gradient norm is 63.7%, and 55.0% of pairs have a relative distance at least 0.6. Same-ten lag-1 cosine mean is 0.7215; directional alignment around 0.7 does not establish a small full-vector L2 difference. This result is limited to the ten retained late-manifest trajectories and does not replace the original 1,000-trajectory summary or prove mathematical non-Lipschitzness. Full report: docs/results/v0-grad-mech-l2-norm.md. Result artifacts are archived at archive/v0-grad-mech-test/outputs/train/v0-grad-mech-l2-norm/; outputs/train/v0-grad-mech-l2-norm/ remains a symbolic link. The ten BF16 gradient caches remain unarchived at outputs/train/v0-crb-reconst-test/gradients/.

## v0-grad-mech-test

Completed: 2026-11-07. The probe processed all 1,000 selected M5 trajectories using the fixed warm-up checkpoint and FP32 gradients. Windows span 512 context and 64 prediction points, with stride 1. Statistics below are weighted by window-pair count; SD is the population standard deviation.

| Lag | Pair count | Cosine mean | Cosine SD | Relative norm difference mean | Relative norm difference SD |
|---:|---:|---:|---:|---:|---:|
| 1 | 1143885 | 0.753878 | 0.253848 | 0.128111 | 0.135564 |
| 8 | 1136885 | 0.464294 | 0.249467 | 0.231456 | 0.170033 |
| 32 | 1112885 | 0.239849 | 0.268146 | 0.321463 | 0.213633 |
| 64 | 1080885 | 0.088391 | 0.268003 | 0.366607 | 0.233860 |
| 128 | 1016885 | 0.043590 | 0.237899 | 0.382731 | 0.242262 |
| 288 | 858074 | 0.024572 | 0.214649 | 0.393459 | 0.247553 |
| 575 | 581872 | 0.013485 | 0.195253 | 0.402461 | 0.253443 |

Cosine similarity declines with lag in this dataset and approaches zero at lag 575. This describes the observed aggregate, not a claim that individual pairs have zero cosine or that the pattern generalizes to other datasets or models. Pair counts vary with lag and observations within a trajectory are dependent.

Artifacts: archive/v0-grad-mech-test/outputs/train/v0-grad-mech-test/; the full visualization is under vis/snapshot_20261107_123811_utc_n1000/. Only the probe results and visualization are archived; their original paths are symbolic links. Data, logs, configuration, scripts, checkpoint, input manifest, and CRB-owned gradient caches remain in their original locations.

### Dataset, trajectory index, and result files

- Dataset: autogluon/chronos_datasets, m5 configuration, revision eeecad0b82a8c237e212ce6f8d1abecb513e2cec. First 1,000 eligible finite nonconstant series of length at least 576 in source order; no random sampling.
- Series IDs, zero-based source rows, lengths, and zero-based manifest ordinals: archive/v0-grad-mech-test/results/trajectory_index.csv. The full input_manifest.json is copied into the same directory.
- Per-trajectory per-lag statistics: archive/v0-grad-mech-test/results/trajectory_lag_similarity.csv.gz, containing 575,000 rows. Columns give pair count, cosine mean, population SD, minimum and maximum, and relative norm difference mean and population SD. Empty metrics mean no pairs. Individual pair values were not retained.
- Original per-trajectory summaries and histograms: archive/v0-grad-mech-test/outputs/train/v0-grad-mech-test/probe/trajectories/trajectory-NNNN.npz; NNNN is the zero-based manifest ordinal.
- Pooled 575-lag CSV and JSON, and PNG, SVG and HTML figures: archive/v0-grad-mech-test/outputs/train/v0-grad-mech-test/vis/snapshot_20261107_123811_utc_n1000/. Pooled means weight trajectories by pair count.
- Verified the exported table against pooled pair counts and cosine means at every lag.


## v0-gradient-proxy-test

Completed both parts on 2026-11-08 using the same ten retained M5 tail trajectories.

- Local-Risk: Result category Other. The full-variance target has real within-track spread: after normalizing each track by its median, p90/p10 is 7.44, 5.25, 4.42, and 3.88 at block lengths 32, 64, 128, and 256. At 576 starts the ratio is 2.06 over only 19 blocks and is exploratory. Sparse output-head samples retain variance ranking better than direction ranking; details and the block distribution are in docs/results/v0-gradient-proxy-test.md. These are retrospective subsampling results, not runtime savings or a budgeted interval-selection test.
- Cross-Rank: Result category Other (qualified positive ranking signal for output-head cosine on the fixed candidate pool; per-reference order remains moderate). Across 45,048 cross-track pairs, pooled Pearson/Spearman are 0.786/0.795. Per-reference mean Spearman/Kendall are 0.640/0.481; mean top-5 overlap is 0.487 versus 0.208 expected by chance. Head cosine MAE/bias are 0.071/-0.016. Embedding cosine is near-random or worse; CE loss difference is weak.

These findings are descriptive, use overlapping reference queries and only two reference providers, and do not establish general M5 performance or a budgeted interval-selection gain. Detailed question-specific conclusions and artifacts: docs/results/v0-gradient-proxy-test.md.


## v0-gradient-proxy-cost

Completed 2026-11-08. On 80 windows from the ten fixed tracks, timed three repeats per method at batch size one. Median end-to-end latency: full BP 42.80 ms/window, output-head gradient 14.21 ms (2.93x paired speedup), mean decoder embedding 11.35 ms (3.72x), and forward loss 11.28 ms (3.71x). Timing includes window tokenization, GPU computation, CPU result transfer, and synchronization. Details: docs/results/v0-gradient-proxy-cost.md; raw repeats: outputs/train/v0-gradient-proxy-cost/per_window_timings.csv.

## v0-vage Static

Completed 2026-11-08 on ten retained trajectories using saved full/head Gram matrices derived from BF16 gradients. VAGE-Static accounts for 204 full-gradient and 204 output-head proxy calls; Uniform accounts for 204 full-gradient and zero proxy calls. Equal-trajectory mean L2 is 3.5437 versus 3.4824 and mean cosine is 0.3651 versus 0.3971, respectively. VAGE wins mean L2 on 2/10 tracks and mean cosine on 1/10. The fixed Static design does not improve the primary pointwise reconstruction metrics. Full protocol and limitations: docs/results/v0-vage.md.

Last modified: 2026-11-08

## v0-vage n_min=1 sensitivity

Completed 2026-11-08 with unchanged budgets and proxy scores. Equal-track mean L2 is 3.8747 versus 3.5437 for n_min=2 and 3.4824 for Uniform; mean cosine is 0.3156 versus 0.3651 and 0.3971. Every n_min=1 track lacks the first global start among its final anchors, and four lack the last; uncovered ends use nearest-anchor extrapolation. This comparison does not isolate minimum-anchor count from endpoint coverage. Details: docs/results/v0-vage.md.

Last modified: 2026-11-08

## v0-vage additional n_min=1 methods

Completed 2026-11-08 on the same ten trajectories. Equal-track mean L2: Uniform 3.4824, VAGE-Quota 3.5031, Residual diagnostic 3.5159, Static 3.8747, Fusion 3.8762, Oracle 3.8774. Mean cosine: 0.3971, 0.3675, 0.3801, 0.3156, 0.3163, and 0.3163 respectively. No VAGE variant improves both primary metrics. Residual is a proxy-stratified anchor diagnostic because Bg=Bp forces selection of every observed proxy candidate. Full protocol and limitations: docs/results/v0-vage.md.

Last modified: 2026-11-08

## v0-uni-reconst-scale

Completed 2026-11-08 on the ten retained trajectories using true complete-gradient anchors and whole-trajectory linear interpolation. Equal-trajectory mean L2 for s={1,2,4,8,16,32,64,128} is {0.0000,0.9650,1.6969,2.2654,2.7382,3.1250,3.4824,3.8809}; mean cosine is {1.0000,0.8952,0.7941,0.6956,0.6002,0.5063,0.3971,0.2563}. The s=64 row matches the previous Uniform baseline. Full protocol, anchor totals, and limitations: docs/results/v0-uni-reconst-scale.md.

Last modified: 2026-11-08

## v0-uni-data-evaluation

Completed 2026-11-08 with 100 seeded, nonadjacent full-gradient references sampled as ten per retained trajectory. Excluding those starts leaves 12,736 query windows. For within-trajectory query pairs one start apart and the same reference, equal-track mean absolute score difference is 0.0402 for cosine and 0.6100 for dot product; mean relative difference is 0.4856 and 0.5372 using the mean absolute score magnitude as denominator. Sign flips occur in 9.40% of lag-1 comparisons. Across 1,273,600 valid query-reference scores, pooled signed mean/population variance are 0.0284/0.0283 for cosine and 0.6101/9.1331 for dot product; pooled signed medians are 0.0284 and 0.1635; pooled mean absolute scores are 0.1378 and 1.8782, with absolute-value medians 0.1207 and 1.0334. At lag 64, mean relative differences rise to 1.1524 and 1.2668. The scores show local association but not near-constant adjacent values under this measure. Full report and 1-575 curves: docs/results/v0-uni-data-evaluation.md.

Last modified: 2026-11-08

## Pair-mean reconstruction supplements: v0-vage and v0-uni-reconst-scale

Completed 2026-11-08 using the same ten trajectories, cached complete-gradient anchors, and exact budgets as the preceding linear-reconstruction runs. Each unobserved start in an anchor gap is represented by the mean of its two endpoint gradients; observed anchors return exact gradients. At 204 total anchors, Uniform equal-track mean L2/cosine change from 3.4824/0.3971 under linear interpolation to 3.5445/0.3498 under pair mean. VAGE-Static n_min=2 changes from 3.5437/0.3651 to 3.5880/0.3205. The pair-mean rule worsens both primary equal-track metrics across the tested VAGE allocations on average. In the Uniform scale sweep, s=1 stays exact and s=2 is nearly unchanged; s=4-128 have higher mean L2 and lower mean cosine. Full tables and limitations: docs/results/v0-vage.md and docs/results/v0-uni-reconst-scale.md.

Last modified: 2026-11-08

## v0-active-value-rank diagnostic extension

Last modified: 2026-11-09

Completed batches 3-proxy-coverage and 4-equal-budget-query on the same 100 references and cached candidate scores as the earlier active-value-rank study. Mean true top-k coverage is 0.4023, 0.6480, 0.7950, 0.8744, 0.9265, and 0.9754 at proxy pools k,2k,3k,4k,5k,7k. The mean number of true top-k windows in proxy rank bins k+1-2k, 2k+1-3k, and beyond 5k is 17.87, 10.70, and 5.33; 65/100 references have at least one true top-k window beyond 5k. The proxy misses are not confined to the immediate k boundary.

At B=64, mean Recall@k is 0.5064 for pure boundary, 0.4776 for proxy high, 0.4295 for random (30 seeds), and 0.5415 for the existing complex method. The complex-minus-pure-boundary difference is -0.0431, +0.0078, +0.0282, and +0.0352 at B=8,16,32,64. This is cached-score replay on the same references used in the earlier study; no held-out claim or downstream training benefit follows. Per-reference tables, figures, methods, and limits: docs/results/v0-active-value-rank-diagnostics.md. Output: outputs/train/v0-active-value-rank/batch-3 and batch-4.


## v0-crb-var

Completed 2026-11-09. Across ten trajectories at matched budgets of ceil(window_count/64), CRB had lower mean relative within-block variance (0.6410 vs 0.6802) and mean direction dispersion (0.4051 vs 0.4402) than Uniform on every track. CRB block size showed a positive association with local dispersion on all ten tracks; larger CRB blocks retained greater dispersion, so its shorter intervals did not preferentially fall in the highest-dispersion regions. Absolute variance did not consistently improve. Details and limitations: docs/results/v0-crb-var.md.

## v1-rage-trev TimerXL seed 0

Last modified: 2026-11-10

MASE is the sole metric used for inspection, comparison, and reporting across the Chronos, TimerXL, and TimesFM Trev batches. All Chronos, TimerXL, and TimesFM Trev rand/full phases use LoRA with frozen base weights; `full` refers to the full eligible data pool, not full-parameter training. TimerXL 84M seed 0 random and full phases completed 4,000 updates and all eight FEV/TIME checkpoint evaluations. At step 4000, FEV MASE for hospital admissions, M_DENSE, US consumption, and world CO2 was 0.6672/2.0354/13.4942/5.7754 for random and 0.6675/2.0330/12.5499/5.2563 for full. Source CSVs, task coverage, and limits: docs/results/v1-rage-trev-timerxl-seed0.md.
