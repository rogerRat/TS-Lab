# 实验结果总览

最后修改日期：2026-11-12

## 结果文件索引

本页为 `docs/results/` 下全部报告的总入口。旧版 v1-rage 结果已废弃，仅作历史参考。

| 报告 | 说明 |
|---|---|
| [v0-crb-mech-test](results/v0-crb-mech-test.md) | 原 64 条平滑 mock；CRB/Uniform、CES 与 est-real 虚拟分块 |
| [64-large-var-k-error-composition](results/64-large-var-k-error-composition.md) | 新 64 条高变化 mock；q≤128 的 K-error 与 MSE 分解（完成） |
| [64-large-var-k-error-composition-full-tables](results/64-large-var-k-error-composition-full-tables.md) | 全 q/K/方法汇总曲线、CI、效率与 MSE 分解表 |
| [v0-active-value-rank-diagnostics](results/v0-active-value-rank-diagnostics.md) | 详见独立报告 |
| [v0-active-value-rank](results/v0-active-value-rank.md) | 详见独立报告 |
| [v0-crb-cos-var](results/v0-crb-cos-var.md) | 详见独立报告 |
| [v0-crb-reconst-fine](results/v0-crb-reconst-fine.md) | 详见独立报告 |
| [v0-crb-reconst-test](results/v0-crb-reconst-test.md) | 详见独立报告 |
| [v0-crb-var](results/v0-crb-var.md) | 详见独立报告 |
| [v0-directional-marginal-gain](results/v0-directional-marginal-gain.md) | 详见独立报告 |
| [v0-grad-mech-l2-norm](results/v0-grad-mech-l2-norm.md) | 详见独立报告 |
| [v0-grad-traj-level-norm-gap](results/v0-grad-traj-level-norm-gap.md) | 详见独立报告 |
| [v0-gradient-proxy-cost](results/v0-gradient-proxy-cost.md) | 详见独立报告 |
| [v0-gradient-proxy-test](results/v0-gradient-proxy-test.md) | 详见独立报告 |
| [v0-gradient-variance-decomposition](results/v0-gradient-variance-decomposition.md) | 详见独立报告 |
| [v0-proxy-dp-segmentation](results/v0-proxy-dp-segmentation.md) | 详见独立报告 |
| [v0-uni-data-evaluation](results/v0-uni-data-evaluation.md) | 详见独立报告 |
| [v0-uni-reconst-scale](results/v0-uni-reconst-scale.md) | 详见独立报告 |
| [v0-vage](results/v0-vage.md) | 详见独立报告 |
| [v0-var-proxy-estimate](results/v0-var-proxy-estimate.md) | 详见独立报告 |
| [v1-computational-effort-estimate](results/v1-computational-effort-estimate.md) | 计时完成；正式实验未开始 |
| [v1-rage-trev-seed0-final](results/v1-rage-trev-seed0-final.md) | 详见独立报告 |
| [v1-rage-trev-timerxl-seed0](results/v1-rage-trev-timerxl-seed0.md) | 详见独立报告 |
| [v1-rage-trev-timesfm-seed0](results/v1-rage-trev-timesfm-seed0.md) | 详见独立报告 |

## v2-foundation-expansion 工程验证

六个固定修订权重均在 CPU 上完成 512+64 窗口的预测、训练损失、LoRA 反传和 AdamW 单步、双窗口批量 VJP 与逐窗口梯度核对；以下参数数目来自各模型实际加载的 `smoke.json`。这是接入验证，没有正式训练集或预测质量指标。Moirai 权重只可用于非商业研究。

| 模型 | 实际参数 | LoRA 可训练参数 | 冒烟验证 |
|---|---:|---:|---|
| chronos-2-small | 27,934,624 | 196,608 | 通过 |
| chronos-2 | 119,477,664 | 589,824 | 通过 |
| chronos-bolt-tiny | 8,652,672 | 98,304 | 通过 |
| chronos-bolt-mini | 21,236,096 | 172,032 | 通过 |
| chronos-bolt-small | 47,718,016 | 294,912 | 通过 |
| moirai-2.0-small | 11,387,208 | 73,728 | 通过 |

验证产物位于 `outputs/train/v2-foundation-expansion/`，每个模型均有 `smoke.json` 和 `.complete`。

## 已有结果摘要

最新完成批次 `64-large-var-k-error-composition`：mock 达成相邻点完整向量差比约 0.5、lag-16 cosine 约 0；模长 CV=0.323。全切分使 Direction error 在所有 q 都下降，但 L2 在 q≤64 上升、q=128 仅下降 0.18%。Cancellation K=1 对 L2 没有稳定优于 Random；Endpoint-Difference/Predicted-Gain 也未显示一致优势。12 张 K-error 图、18 张 MSE 分解图及 paired 95% CI、相关性与 Oracle 差距见[批次报告](results/64-large-var-k-error-composition.md)。


跨批次结果显示：64-smooth 在共同 q≤32 下全切分均使 L2 总误差下降；64-large-var 在 q≤64 的 L2 最佳点为 K=0，而 q=128 的 Oracle-L2 曲线出现内部最优 K=81/127。此处 K=q−1 只给每个 Uniform 区间再加一个 midpoint，并非切成逐点块。详细机制解释与逐轨迹最优 K 统计见 [64-large-var 批次报告](results/64-large-var-k-error-composition.md)。

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


## v0-proxy-dp-segmentation

Completed 2026-11-10 using cached BF16 gradients from the four longest tracks (ordinals 01, 05, 06, 09), with exactly 22 intervals per track for DP and uniform. Unweighted proxy-variance DP reduced equal-track true relative variance (0.5031 vs 0.6762), equal-interval direction dispersion (0.2995 vs 0.4298), and absolute variance (3.7787 vs 9.4982), but raised window-weighted direction dispersion (0.5844 vs 0.4297). The length-weighted proxy objective reduced relative variance (0.5717 vs 0.6762) and equal-interval direction dispersion (0.3572 vs 0.4298), but raised window-weighted direction dispersion (0.4923 vs 0.4297) and absolute variance (12.5689 vs 9.4982). These are descriptive equal-track means over four selected long tracks; neither objective dominates all metrics. Full definitions and per-track results: docs/results/v0-proxy-dp-segmentation.md. Both output trees have .complete markers. The raw BF16 full-gradient source caches were deleted after validation at the user's direction on 2026-11-10; derived outputs and proxy caches remain.

## v1-rage-trev TimerXL seed 0

Last modified: 2026-11-10

MASE is the sole metric used for inspection, comparison, and reporting across the Chronos, TimerXL, and TimesFM Trev batches. All Chronos, TimerXL, and TimesFM Trev rand/full phases use LoRA with frozen base weights; `full` refers to the full eligible data pool, not full-parameter training. TimerXL 84M seed 0 random and full phases completed 4,000 updates and all eight FEV/TIME checkpoint evaluations. At step 4000, FEV MASE for hospital admissions, M_DENSE, US consumption, and world CO2 was 0.6672/2.0354/13.4942/5.7754 for random and 0.6675/2.0330/12.5499/5.2563 for full. Source CSVs, task coverage, and limits: docs/results/v1-rage-trev-timerxl-seed0.md.

## v1-rage-trev TimesFM seed 0

Last modified: 2026-11-10

TimesFM 2.5 seed 0 random and full phases completed 4,000 BF16 LoRA updates and all eight FEV/TIME checkpoint evaluations. At step 4000, FEV MASE for hospital admissions, M_DENSE, US consumption, and world CO2 was 0.6547/1.3943/8.8224/3.9751 for random and 0.6646/1.4042/8.8852/3.9794 for full. This is one seed; details and source files: docs/results/v1-rage-trev-timesfm-seed0.md.

## v1-rage-trev seed 0 final scope

Last modified: 2026-11-10

The study was closed at one seed by user request. All six Chronos, TimerXL, and TimesFM seed 0 rand/full LoRA phases completed 4,000 updates and eight FEV/TIME checkpoint evaluations. Only MASE is used in the final comparison. The incomplete Chronos seed 1 run was stopped and its partial outputs removed; no seed 1-3 results are reported. Final FEV and TIME MASE tables and source paths: docs/results/v1-rage-trev-seed0-final.md.

## v0-crb-reconst-fine

Last modified: 2026-11-10

Completed one fixed-seed random M5 trajectory (manifest ordinal 771, FOODS_1_079_CA_3; 1,068 legal windows) using all 98,304 trainable ChronosTiny LoRA gradient coordinates at a fixed fresh adapter initialization. CRB and Uniform used exactly matched K=max(2,ceil(1068/q)) anchor calls at q={1,2,4,8,26,32,64,128}; three CRB discretionary stages received equal allocations up to integer remainder. Mean L2 favors CRB at q=2,4,8,128; Uniform at q=26,32,64; q=1 is exact for both. At q=64, mean L2 is 0.012020 CRB versus 0.011536 Uniform. The one-track result is mixed and does not establish cross-trajectory superiority. Fresh LoRA initialization leaves half the coordinates with zero gradient; a trained-adapter conclusion is not supported. Full metrics, comparison to prior full-parameter CRB work, validation, and limitations: docs/results/v0-crb-reconst-fine.md. Raw results and .complete: outputs/train/v0-crb-reconst-fine/.

### v0-crb-reconst-fine twelve-track extension

Last modified: 2026-11-10

The fixed-seed extension completed 12 M5 FOODS_1 trajectories and 192 method-budget rows. Equal-track mean L2 favors CRB only at q=2 (0.024898 vs 0.026663 Uniform; 10/12 track wins). Uniform is lower at q=4,8,26,32,64,128; at q=26 it wins 12/12 tracks. At q=64, CRB/Uniform mean L2 is 0.104674/0.102412 and CRB wins 2/12 tracks. Mean cosine favors Uniform at every nontrivial q. The initial single-track CRB wins at q=4 and 8 do not generalize to most of these 12 selected tracks. All dense gradient caches were released after per-track validation; metrics, anchors, metadata, logs, and completion markers remain. Detailed table and limitations: docs/results/v0-crb-reconst-fine.md. Aggregate source files: outputs/train/v0-crb-reconst-fine/twelve_track_metrics.csv and twelve_track_summary.json.

### v0-crb-reconst-fine direction-cosine decision

Last modified: 2026-11-10

Per user request, the 12-track comparison is now judged by mean per-window cosine of reconstructed versus true LoRA gradients, using saved metrics and unchanged anchors. Uniform has higher equal-track mean cosine at every nontrivial q. At q=2 the means are 0.908663 Uniform versus 0.907173 CRB, with a 6/6 track-win split. At q=4,8,32,64 Uniform wins 11/12 tracks; at q=26 it wins 12/12; at q=128 it wins 8/12. Full table: docs/results/v0-crb-reconst-fine.md. No gradients were recomputed or retained.

## v0-crb-cos-var

Last modified: 2026-11-10

Completed on 12 selected M5 FOODS_1 trajectories. For each q, Uniform and native equal-budget CRB partitions define intervals between true-gradient anchors. The representative direction is the mean of the two known projected endpoint gradients; each block score is mean 1-cosine to true strict-interior projected gradients in 4096d. By equal-block, equal-track mean, CRB is lower at q=2,8,16,32,64,128 and Uniform at q=4. By interior-window-weighted mean, Uniform is lower at every q>1. Both are reported because CRB intervals vary in length. q=1 has no interior and is undefined. Full protocol and results: docs/results/v0-crb-cos-var.md. No full gradients or projected trajectories were retained.

## v1-rage result status update

Last modified: 2026-11-10. All earlier v1-rage batches are deprecated because their source data selection was unsuitable. Preserve result files and the historical numbers above solely as possible references; none is a formal answer. The replacement source collection is src-8192, with 8,192 fixed trajectories in configs/data/src-8192.yaml.

## src-8192 source coverage

Last modified: 2026-11-10. This is a data preparation result, not a v1-rage training result. Version 2 contains exactly 8,192 trajectories from 35 eligible configurations and 191,176,690 scalar values. Per-configuration counts and every selected row are in configs/data/src-8192.yaml; values are in data/src/shard_dataset/src-8192/. Version 1 was removed as part of the user-requested data cleanup. All earlier v1-rage training results remain deprecated.

Data cleanup completed on 2026-11-10: all project data other than the current src-8192 snapshot were removed.


## src-96 仅 Uniform 梯度准备状态

最后修改日期：2026-11-11。用户决定停止并删除本批次 CRB 部分。TimerXL 和 TimesFM 的 Uniform 96/96 条已完成，缓存已过滤为仅保留 Uniform 梯度，CRB 清单及专属缓存行已删除。Chronos 的 Uniform 部分仍在运行，三模型整批完成标记尚未生成。本节只报告实际完成状态，不报告已删除的 CRB 梯度结果。

## src-96、q=4 输出头梯度资源估算

最后修改日期：2026-11-11。仅作估算，未执行。每模型 666,575 个锚点；按 4096 维 BF16 投影，三模型总容量约 15.3–16.2 GiB。用时没有直接实测，低置信度计划范围和前提见 [独立报告](results/src-96-q4-output-head-estimate.md)。


## TimerXL 四轨梯度代表性与曲率

[v0-timerXL-grad-4-lm-head](results/v0-timerXL-grad-4-lm-head.md)：轨迹内非临近窗口对的输出头/LoRA方向余弦 Spearman 为0.297、不可估、0.249、0.303；最接近10%窗口对的LoRA同档召回为49.8%、不可估、18.3%、33.3%。有正向代表性但关联不强，轨迹差异明显。

[v0-timer-XL-grad-4-curvature](results/v0-timer-XL-grad-4-curvature.md)：四轨二阶差中心化RMS/原RMS均约1，二阶差模长CV为1.036、0.822、0.406、0.701；相邻二阶差方向余弦均值为-0.130、-0.520、-0.613、-0.529。不支持轨迹级恒定曲率或逐窗口平滑变化。

曲率补充：相邻变化 d3 RMS/d2 RMS 为1.59–1.80，二阶差相对各轨梯度 RMS 为0.54–1.53；绝对二阶差 RMS 跨轨最大约9.5倍，但归一化后排序不同。16段曲率能量跨度：长轨迹0/2/3为1.38/1.29/1.33倍，153窗口的轨迹1为6.46倍，呈明显早高晚低。详见曲率报告。

## v0-timer-XL-grad-4-change-velocity

[四轨逐滞后变化报告与分轨迹图](results/v0-timer-XL-grad-4-change-velocity.md)：对全484,907个LoRA窗口在各轨内计算k=1..512的所有有效窗口对。方向余弦首次低于0.5的k为6、21、2、6；相对L2差首次超过平均模长的k为5、18、2、5。轨迹2方向变化最快，但相对模长差在k=512仅0.296。完整2048行均值、标准差与对数见输出CSV；四张独立三面板图见报告。

## v0-timerXL-grad-4-lm-head：变化曲线比较

[输出头与LoRA同窗口段对照报告](results/v0-timerXL-grad-4-lm-head-change-velocity.md)：长轨各两个768窗口段、短轨全153窗口，合计4761个头梯度。按相同窗口对和k=1..512比较三项指标。输出头与LoRA的三指标不能整体数值近似；尤其输出头在四轨上都低估相对模长变化。长轨同段LoRA与全轨LoRA曲线的三指标MAE仅0.017–0.039，小于多数头/LoRA差异。四张叠加图和逐k CSV见报告。

方向余弦退化排序补充：在同段k=1..16，按平均余弦由低到高及首次跌破0.5的k，两种口径均排轨迹2、0、3、1；按c(1)-c(16)的下降量则输出头为3、0、1、2，LoRA为0、2、3、1。输出头轨迹2在k=1已接近零，阈值排序不能单独解释为退化速度一致。数值表见输出头与LoRA同段对照报告。

## v0-timer-XL-grad-4-change-velocity：两个随机16窗口区间

[两区间与全轨变化速度比较](results/v0-timer-XL-grad-4-two-random-16-blocks.md)：每轨随机两个连续16点区间，seed 0，共128个已有LoRA梯度。以k=1..15等k平均定义轨迹分数，两区间能找出方向/L2变化最快的轨迹2和最慢的轨迹1，但三项都把轨迹0、3排序对调，故未完整恢复全轨排序。方向曲线MAE 0.023–0.045；轨迹0相对模长分数由全轨0.492被估为0.315。单次随机抽样不能给出一般成功率。逐k CSV与四张分轨图见报告。

## v0-timer-XL-grad-64-change-velocity

[64轨变化速度与代理结果](results/v0-timer-XL-grad-64-change-velocity.md)：共1,980,371个合法窗口。轨迹间方差份额（方向/L2/模长）为0.798/0.805/0.487；输出头代理64轨排序Spearman为0.912/0.916/0.557，两段16窗口真实LoRA估计为0.909/0.880/0.609。完整结果和限制见独立报告；原始曲线、轨迹分数与段级数据在outputs/train/v0-timer-XL-grad-64-change-velocity/analysis/。

v0-timer-XL-grad-64-change-velocity 补充：每轨两段16窗口输出头梯度（共2048点）恢复全量LoRA的方向/L2/模长分数MAE为0.172/0.158/0.167，排序Spearman为0.868/0.829/0.294，前16名重合11/11/5。明显弱于同段真实LoRA，模长变化尤其差；详细对照及限制见[64轨报告](results/v0-timer-XL-grad-64-change-velocity.md)。


## v0-crb-final-test-timerXL-64tr

按q为平均锚点间距、K=max(2,ceil(N/q))完成64轨BF16 LoRA投影梯度重放。Uniform在1:0:1与1:1:0的所有q均胜出；coverage-only 1:0:0仅在q=64优于Uniform。完整加权表、等轨迹均值、口径与限制见[结果报告](results/v0-crb-final-test-timerXL-64tr.md)。

## v0-timerxl-grad64-pca

TimerXL 64轨真实4096维LoRA投影的逐轨PC1解释率中位数为12.1%（第10–90百分位6.6%–32.4%）；64轨PC1两两夹角中位数87.3°，1,944/2,016对超过60°。以前后半PC1余弦至少0.8且窗口数不少于4096筛出的34条稳定轨迹中，夹角中位数仍为87.1°。18/64轨迹前后半PC1余弦低于0.5，故方向差异属于描述性结果，不能当成正式显著性检验。rank80方向数均值174、中位数137；窗口数至少4096的40轨均值/中位数为248/235；按完整合法起点长度L逐轨算v/L，均值4.07%、中位数1.63%，显示相对4096维有谱集中但非十几个方向的极低秩。逐轨数值、方法和限制见[详细报告](results/v0-timerxl-grad64-pca.md)。

## ChronosTiny 与 TimerXL LoRA 梯度关系：src-96 ordinal 0 单轨试验

使用原始预训练 ChronosTiny（非预热检查点），在src-96第0条轨迹的230,161个合法窗口逐窗计算LoRA梯度，投影为4096D Count-Sketch（seed42）并以BF16保存；基础模型冻结，LoRA r=8、alpha=16、dropout=0、q/v目标模块、初始化seed0。与同轨全量TimerXL 4096D投影梯度比较。两模型参数空间不同，投影坐标不对齐，因此主相关性定义为同一轨迹内窗口对余弦相似度的跨模型相关，而非两个模型梯度向量的直接余弦。

每模型沿全轨等距取128个相同窗口，保留起点相隔至少576的8,128个窗口对。TimerXL与ChronosTiny窗口对余弦的Spearman为0.0758、Pearson为0.0768；Top 10%相似窗口对重合率为14.4%。全量相邻窗口对余弦序列的Spearman为−0.1379。k=1平均指标：TimerXL与ChronosTiny方向余弦0.9314与0.9348，相对L2差0.4085与0.3574，相对模长差0.1799与0.0478。现有单轨数据不支持较强的窗口关系相关性；轨迹级排序因仅一条轨迹无法评估，也不能推断跨轨总体关系。

逐滞后指标、配对窗口对统计和完成标记位于 outputs/train/v0-chronostiny-timerxl-lora-track0/analysis/；全部投影梯度缓存位于同目录formal/。完整性标记已核验。


## v0 全部结果汇总（2026-11-12）

按研究问题归纳的主结论、限制与专题报告入口见 [v0 结果总览](results/v0-consolidated-summary-20261112.md)。所有已登记的 v0 专题报告均保留在 docs/results/；其中包括已完成结果、明确终止的批次、部分运行缓存和仅供诊断的 mock/oracle 分析，摘要不把后两类外推为正式正向结论。

ChronosTiny ordinal 0 全窗口梯度完成，方法与边界见 [专题报告](results/v0-chronostiny-timerxl-20261112.md)：非临近窗口对余弦 Spearman=0.0758，Top-10% 窗口对重合率 14.4%；全量相邻窗口对余弦 Spearman=-0.1379。单轨数据不足以推断跨轨相关。ChronosTiny 前 64 轨扩展按用户调整停止，仅保留部分梯度缓存，不形成全轨分析结果。TimerXL 64 轨结果及 CRB/Uniform 完整比较见 [64 轨变化报告](results/v0-timer-XL-grad-64-change-velocity.md)、[PCA 报告](results/v0-timerxl-grad64-pca.md)和[CRB 最终比较](results/v0-crb-final-test-timerXL-64tr.md)。

2026-11-12 归档了所有非缓存 v0/v00 生成输出和日志，文件位于 archive/v0-consolidated-20261112/；原输出和日志路径通过符号链接保持可访问。后续可能使用的梯度缓存仍在原路径。当前无 v0 计算任务运行。

最后修改日期：2026-11-12


2026-11-12 存储说明：为给之后的正式 v1 实验预留数据盘空间，已删除 v0-gradient-proxy-test 的十个未投影 FP32 逐窗口输出头梯度数组；派生结果保留，详情见 docs/results/v0-gradient-proxy-test.md。

## 2026-11-12 spacing-q-dir
最后修改日期：2026-11-12。四个CRB比例及匹配Uniform已全部完成，64轨共1,980,371个有效方向；全局模长第1百分位阈值0.020094235，置零19,804个方向。按长度加权时，四臂中最佳CRB仅在q=2、12、24、64胜Uniform；块等权时四臂全部q胜Uniform。完整数值、定义和数据源见 [spacing-q-dir结果](results/v0-crb-final-test-timerXL-64tr-spacing-q-dir.md)。

2026-11-12：spacing-q-dir追加q=128、256已完成。q=128长度加权Uniform最优（0.707035），q=256长度加权CRB 1:1:1最优（0.760005）；块等权两列CRB 1:0:1最优（0.618695、0.688974）。完整对比见 [spacing-q-dir结果](results/v0-crb-final-test-timerXL-64tr-spacing-q-dir.md)。

## 2026-11-12 23:38 CST v1-crb-redo TimerXL 两基线

最后修改日期：2026-11-12 23:38 CST。full 和 rand01 各完成 4,000 步及相同的 FEV 4 + TIME 8 任务评估。full 在 10/12 项、rand01 在 2/12 项的 MASE 更低；FEV 四项等权均值分别为 6.447586 / 6.703599，TIME 八项等权均值分别为 1.710087 / 1.736942。完整逐任务数值、计算口径及原始文件见 [TimerXL 两基线结果](results/v1-crb-redo-timerxl-baselines.md)。
