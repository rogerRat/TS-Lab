# v0-gradient-proxy-test: Local-Risk and Cross-Rank

Last modified: 2026-11-08

Status: Local-Risk and Cross-Rank complete. Local-Risk scope: 12,836 windows of the selected M5 tail sample at the fixed warmed Chronos-T5 checkpoint.

## Batch questions and conclusions

### Part Local-Risk

Question: Within nonoverlapping blocks of 32 consecutive window starts, do inexpensive proxy statistics preserve the ranking of full-gradient vector variance and full-gradient direction dispersion? Does that relationship persist when only a subset of proxy gradients is observed?

Result category: Other. The 32-point result is strongly positive; the sparse-sampling extension finds that variance ranking remains strong down to 4 points, while direction ranking degrades more at low counts. No budgeted interval-selection policy was evaluated. Output-head dispersion has equal-trajectory mean Spearman 0.9476 with full vector variance and 0.8161 with full direction dispersion. This supports using the head proxy to flag candidate high-variation blocks for follow-up, but does not show how many full-gradient calls such a policy saves at a chosen coverage level.

#### Local-Risk sampling extension: sparse proxy measurements

Question: If only k output-head gradients are computed in each 32-start block, with k in {32,24,16,8,4}, how strongly do sampled-head variance and direction-dispersion estimates correlate with the full-gradient statistics computed from all 32 points?

Result category: Other (the variance ranking stays strong with sparse samples, while direction ranking degrades more quickly). Across 396 complete blocks from ten tracks, each count below 32 was evaluated using 20 fixed-seed nested random subsets per block. The table gives equal-track Pearson and median equal-track Spearman; bracketed ranges are the 10th-90th percentiles across sampling replicates.

| Proxy points per block | Full vector variance Pearson | Full vector variance Spearman | Full direction dispersion Pearson | Full direction dispersion Spearman |
|---:|---:|---:|---:|---:|
| 32 | 0.944 | 0.948 | 0.854 | 0.816 |
| 24 | 0.941 | 0.942 (0.938-0.945) | 0.845 | 0.806 (0.801-0.813) |
| 16 | 0.935 | 0.935 (0.930-0.940) | 0.821 | 0.783 (0.776-0.798) |
| 8 | 0.921 | 0.916 (0.907-0.922) | 0.755 | 0.704 (0.688-0.737) |
| 4 | 0.878 | 0.867 (0.852-0.889) | 0.600 | 0.579 (0.538-0.619) |

Interpretation: For full vector variance, sampling 8 of 32 proxy points preserves most of the block-ranking association; 4 points still provide a useful but weaker signal. Direction dispersion is more sensitive: it remains fairly strong at 16 points, is moderate at 8, and drops more clearly at 4. At 4 points, the proxy still tends to distinguish more variable blocks, but is substantially less reliable for ranking direction changes.

This is a retrospective subsampling analysis of saved head Gram matrices: the selected-point statistics match what those same sampled gradients would produce, but all proxy gradients had already been computed and cached. It measures statistical information loss, not actual end-to-end runtime savings. The sampled variance uses the same population formula on the observed subset; correlation, rather than absolute variance calibration, is the target. Sampling ranges describe sensitivity to the fixed random-subset design, not confidence intervals over datasets. No interval-selection policy or downstream training outcome was tested.

Artifacts: outputs/train/v0-gradient-proxy-test/local_risk_sparse/summary.csv gives the summary table; replicate_correlations.csv gives per-track/per-replicate associations; sampled_block_statistics.csv contains sampled and full block statistics; input_manifest.json records the fixed sampling design.

### Local-Risk 64-point sampling extension

Question: In consecutive 64-start blocks, if only k output-head gradients are computed for k in {64,48,32,16,8,4}, how strongly do sampled proxy variance and direction dispersion correlate with the true full-gradient statistics computed from all 64 points?

Result category: Other (variance ranking remains strong at low counts; direction ranking degrades more). Across 194 complete blocks from ten tracks, each count below 64 was tested with 20 fixed-seed nested random subsets per block. Values are equal-track Pearson and median equal-track Spearman; Spearman ranges in parentheses are the 10th-90th percentiles over sampling replicates.

| Proxy points per 64-start block | Full vector variance Pearson | Full vector variance Spearman | Full direction dispersion Pearson | Full direction dispersion Spearman |
|---:|---:|---:|---:|---:|
| 64 | 0.936 | 0.935 | 0.858 | 0.808 |
| 48 | 0.934 | 0.931 (0.926-0.935) | 0.853 | 0.808 (0.800-0.816) |
| 32 | 0.931 | 0.924 (0.920-0.929) | 0.843 | 0.805 (0.786-0.812) |
| 16 | 0.918 | 0.914 (0.898-0.924) | 0.809 | 0.771 (0.741-0.789) |
| 8 | 0.896 | 0.886 (0.868-0.899) | 0.733 | 0.693 (0.651-0.741) |
| 4 | 0.845 | 0.837 (0.792-0.859) | 0.579 | 0.557 (0.512-0.650) |

Interpretation: With 8 proxy points out of 64, the block ranking by overall gradient variance remains strong (Spearman 0.886), while direction-dispersion ranking is moderate (0.693). With 4 points, variance ranking remains useful (0.837) but direction ranking is weaker (0.557). Using 16 points preserves stronger direction ranking (0.771); 32 points brings it close to the all-64 baseline (0.805 versus 0.808).

The all-64 baseline correlations (0.935 for variance and 0.808 for direction) are slightly below the 32-start-block baselines (0.948 and 0.816). This is an observed block-scale difference; this experiment does not identify its cause. There are fewer complete 64-start blocks (194 versus 396 32-start blocks).

This is retrospective subsampling of saved Gram matrices. It measures information loss in the block statistics, not actual runtime savings, online interval selection, or downstream training gains. Sampling ranges are design sensitivity summaries, not confidence intervals over trajectories. The sampled windows overlap in raw observations, and the ten tracks are the same selected M5 tail sample.

Artifacts: outputs/train/v0-gradient-proxy-test/local_risk_sparse_64/summary.csv, replicate_correlations.csv, sampled_block_statistics.csv, and input_manifest.json. The part .complete marker is present.

### Local-Risk 128-, 256-, and 576-start sampling extension

Question: As the interval grows to 128, 256, or 576 consecutive window starts, how do sparse output-head variance and direction-dispersion statistics correlate with the full-gradient statistics over every start in the interval?

Result category: Other. The 128- and 256-start results show useful within-track associations under sparse sampling. The 576-start case has only 19 blocks total and at most two per track; its correlations are pooled across tracks and are exploratory, not a reliable within-track estimate.

Proxy counts preserve the fractions 100%, 75%, 50%, 25%, 12.5%, and 6.25%: 128 blocks use {128,96,64,32,16,8}; 256 blocks use {256,192,128,64,32,16}; 576 blocks use {576,432,288,144,72,36}. Each count below the full block size uses 20 nested, fixed-seed samples without replacement. The following rows show Pearson r and median Spearman rho; bracketed rho ranges are the 10th-90th sampling-replicate percentiles. The 128 block correlations average ten track-level correlations; the 256 results average nine tracks with at least three blocks; 576 results pool all blocks.

| Block starts | Proxy points | Full variance r / rho | Full direction dispersion r / rho |
|---:|---:|---:|---:|
| 128 | 128 | 0.907 / 0.872 | 0.899 / 0.827 |
| 128 | 96 | 0.906 / 0.869 (0.850-0.877) | 0.897 / 0.830 (0.820-0.837) |
| 128 | 64 | 0.899 / 0.860 (0.843-0.875) | 0.893 / 0.823 (0.808-0.841) |
| 128 | 32 | 0.894 / 0.845 (0.832-0.866) | 0.875 / 0.803 (0.780-0.827) |
| 128 | 16 | 0.871 / 0.840 (0.819-0.859) | 0.838 / 0.770 (0.737-0.813) |
| 128 | 8 | 0.835 / 0.809 (0.778-0.854) | 0.763 / 0.702 (0.618-0.753) |
| 256 | 256 | 0.942 / 0.900 | 0.933 / 0.822 |
| 256 | 192 | 0.941 / 0.900 (0.889-0.900) | 0.933 / 0.839 (0.821-0.856) |
| 256 | 128 | 0.937 / 0.889 (0.877-0.911) | 0.934 / 0.839 (0.800-0.856) |
| 256 | 64 | 0.927 / 0.883 (0.867-0.911) | 0.918 / 0.800 (0.756-0.846) |
| 256 | 32 | 0.913 / 0.867 (0.832-0.891) | 0.901 / 0.811 (0.700-0.856) |
| 256 | 16 | 0.904 / 0.867 (0.778-0.890) | 0.866 / 0.772 (0.698-0.822) |
| 576 | 576 | 0.957 / 0.949 | 0.860 / 0.840 |
| 576 | 432 | 0.954 / 0.941 (0.931-0.955) | 0.855 / 0.846 (0.823-0.860) |
| 576 | 288 | 0.951 / 0.946 (0.936-0.964) | 0.851 / 0.839 (0.819-0.867) |
| 576 | 144 | 0.938 / 0.941 (0.922-0.960) | 0.816 / 0.825 (0.770-0.853) |
| 576 | 72 | 0.919 / 0.921 (0.900-0.946) | 0.763 / 0.767 (0.689-0.829) |
| 576 | 36 | 0.875 / 0.892 (0.822-0.926) | 0.705 / 0.704 (0.544-0.785) |

Interpretation: At 128 starts, 8 sampled proxy points (6.25%) retain variance rho 0.809 and direction rho 0.702. At 256 starts, 16 points (6.25%) retain variance rho 0.867 and direction rho 0.772; 32 points improve direction rho to 0.811. This suggests longer blocks can preserve the cross-block ranking with fewer proxy points in this sample, but block-level and track-level results have different numbers of observations and should not be read as a proven benefit of longer intervals.

The 576-start pooled numbers look high even at 36 points, but they come from only 19 blocks across ten tracks. No track has three such blocks, so within-track correlations cannot be calculated. The pooled result can reflect differences between tracks as well as true within-track proxy performance. Treat this as exploratory only.

All results are retrospective calculations from saved full/head Gram matrices. They do not measure actual runtime savings or test an online policy that selects intervals for full-gradient computation. Blocks do not overlap by starting index, but their underlying 576-point windows can overlap in raw observations.

Artifacts: outputs/train/v0-gradient-proxy-test/local_risk_sparse_long/summary.csv, replicate_correlations.csv, sampled_block_statistics.csv, and input_manifest.json. The part .complete marker is present.

### Local-Risk target variance spread

Question: Do the full-gradient variance targets actually vary enough across blocks to support screening, or do most blocks sit near one common value?

Result: The full-gradient variance targets show substantial within-track spread for 32- to 256-start blocks. To reduce between-track scale effects, each block variance was divided by its own track's median variance before pooling. The normalized p90/p10 ratio remains 7.44 at 32 starts, 5.25 at 64, 4.42 at 128, and 3.88 at 256. Thus the differences are not only due to some tracks having larger gradients overall.

| Block starts | Complete blocks | Full variance p10 | Median | p90 | Within-track-normalized p90/p10 | Median within-track CV |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 396 | 2.12 | 6.21 | 15.75 | 7.44 | 0.810 |
| 64 | 194 | 2.98 | 7.82 | 17.19 | 5.25 | 0.697 |
| 128 | 93 | 4.18 | 11.24 | 21.12 | 4.42 | 0.505 |
| 256 | 46 | 5.14 | 11.69 | 20.32 | 3.88 | 0.435 |
| 576 | 19 | 8.91 | 13.06 | 21.10 | 2.06 | 0.218* |

The median within-track coefficient of variation (CV) is calculated only for tracks with at least two blocks. The 576-start CV uses two blocks per track and is highly unstable; only 19 blocks exist overall. The 576 normalized spread is also pooled over those blocks and remains exploratory.

Interpretation: There are genuinely lower- and higher-variance intervals in this sample, especially at shorter block lengths. The spread shrinks as block length grows, consistent with averaging over more starts. This answers whether the target has useful variation, but not whether a chosen proxy budget reliably finds the highest-variance tail. The prior sparse-sampling correlations show the proxy tracks the broad variation; a next decision-quality check should measure top-quantile recall or enrichment at a fixed sampling budget.

Full direction-dispersion targets also vary, but their distribution moves toward the upper bound as block length increases: pooled p10-p90 is 0.391-0.706 at 32 starts, 0.502-0.803 at 64, 0.644-0.912 at 128, 0.716-0.920 at 256, and 0.875-0.957 at 576. The 576 result again has only 19 pooled blocks.

## Part Cross-Rank

Question: For each reference window from the two shortest retained tracks, do proxy scores rank nonoverlapping source windows from the other eight tracks in the same order as full-gradient cosine, and do proxy cosine values approximate the full-gradient cosine values?

Result category: Other (qualified positive ranking signal for output-head cosine on this fixed pool; per-reference ordering is moderate, so direct substitution for full gradients is not established).

The comparison contains 1,877 reference queries and 24 source candidates (three disjoint 576-point windows from each of eight source tracks), for 45,048 cross-track pairs. Output-head cosine has pair-weighted Pearson 0.786 and Spearman 0.795 across all pairs. For the decision-relevant per-reference rankings, query-weighted mean Spearman is 0.640, mean Kendall tau-b is 0.481, and mean top-5 overlap is 0.487. A random five-of-24 ranking has expected overlap 0.208. Per-reference mean Spearman is 0.620 for reference track 00 and 0.649 for track 07. Head cosine calibration has mean per-reference MAE 0.071, RMSE 0.091, and signed bias -0.016.

The embedding cosine is not useful for this cross-track ranking in this sample (mean per-reference Spearman -0.020; top-5 overlap 0.250). Negative absolute loss difference is weak (mean per-reference Spearman 0.222; top-5 overlap 0.285). The two reference tracks give similar average head-ranking strength, but there are only two reference providers and 24 candidates. The 1,877 queries overlap within their tracks and are dependent; these summaries are descriptive, not independent-query significance tests. The ten tracks are a nonrepresentative selected tail sample, so this does not establish performance on unrestricted M5 windows, other datasets, or other models.

Artifacts: outputs/train/v0-gradient-proxy-test/cross_rank/input_manifest.json records exact track identities and selected source starts; cross_track_pair_scores.csv has all 45,048 scores; query_ranking.csv has all 1,877 per-reference summaries; aggregate.json has aggregate and per-reference-track summaries. The Cross-Rank .complete marker is present.

Protocol: ../gradient-proxy-protocol.md. Direction is primary. Full reference gradients are BF16 copies of FP32 BP; every coordinate is included. Proxy head gradients are FP32 output-path derivatives with detached backbone states. Embeddings are valid-token means of final decoder states under teacher forcing. Loss is the original CE.

All reported summary correlations weight trajectories equally. Ranges are observed trajectory ranges, not confidence intervals. Overlapping pairs are dependent; no pair-independent p-values are used.

## Window norm associations

| Proxy | Mean Pearson | Mean Spearman | Spearman range |
|---|---:|---:|---:|
| head_norm | 0.9312 | 0.9249 | 0.7156 to 0.9765 |
| embedding_norm | 0.1606 | 0.1245 | -0.0877 to 0.3461 |
| loss | 0.4637 | 0.4778 | 0.3078 to 0.7287 |

## Direction associations at fixed lag

Head and embedding entries correlate pairwise cosine with full-gradient cosine. Loss entries correlate absolute loss difference with full-gradient angular distance (1-cosine). Larger positive values indicate better ordering in all columns. These measure association, not equality of cosine values.

| Lag | Proxy | Mean Pearson | Mean Spearman | Spearman range |
|---:|---|---:|---:|---:|
| 1 | head_cosine | 0.9483 | 0.9379 | 0.8961 to 0.9611 |
| 1 | embedding_cosine | 0.8838 | 0.8568 | 0.8302 to 0.8881 |
| 1 | loss_difference | 0.3730 | 0.4855 | 0.3135 to 0.6500 |
| 2 | head_cosine | 0.9285 | 0.8910 | 0.8380 to 0.9333 |
| 2 | embedding_cosine | 0.8423 | 0.7920 | 0.7704 to 0.8527 |
| 2 | loss_difference | 0.2496 | 0.2888 | 0.1950 to 0.5025 |
| 4 | head_cosine | 0.8981 | 0.8298 | 0.7406 to 0.8908 |
| 4 | embedding_cosine | 0.7598 | 0.6578 | 0.5950 to 0.7433 |
| 4 | loss_difference | 0.1421 | 0.1529 | 0.0282 to 0.3981 |
| 8 | head_cosine | 0.8449 | 0.7735 | 0.6755 to 0.8636 |
| 8 | embedding_cosine | 0.5935 | 0.4781 | 0.3915 to 0.5978 |
| 8 | loss_difference | 0.0905 | 0.0889 | -0.0929 to 0.3651 |
| 16 | head_cosine | 0.8155 | 0.7566 | 0.6099 to 0.8770 |
| 16 | embedding_cosine | 0.3698 | 0.3147 | 0.2213 to 0.4254 |
| 16 | loss_difference | 0.0822 | 0.0825 | -0.0715 to 0.4054 |
| 32 | head_cosine | 0.8052 | 0.7731 | 0.5971 to 0.8810 |
| 32 | embedding_cosine | 0.1599 | 0.1601 | 0.0350 to 0.3353 |
| 32 | loss_difference | 0.1339 | 0.1456 | 0.0008 to 0.4279 |
| 64 | head_cosine | 0.8297 | 0.8174 | 0.7086 to 0.8891 |
| 64 | embedding_cosine | 0.0135 | 0.0431 | -0.0882 to 0.2640 |
| 64 | loss_difference | 0.2115 | 0.2068 | -0.0120 to 0.4442 |
| 128 | head_cosine | 0.8176 | 0.8227 | 0.6780 to 0.8920 |
| 128 | embedding_cosine | 0.0321 | 0.0458 | -0.0663 to 0.1706 |
| 128 | loss_difference | 0.2601 | 0.2623 | -0.0419 to 0.5212 |
| 256 | head_cosine | 0.7833 | 0.7794 | 0.5301 to 0.8794 |
| 256 | embedding_cosine | 0.0165 | -0.0028 | -0.1967 to 0.0921 |
| 256 | loss_difference | 0.2819 | 0.2673 | 0.0402 to 0.5469 |
| 512 | head_cosine | 0.7408 | 0.7372 | 0.2565 to 0.9144 |
| 512 | embedding_cosine | 0.0181 | 0.0117 | -0.1655 to 0.1139 |
| 512 | loss_difference | 0.2759 | 0.2835 | -0.0307 to 0.6138 |
| 575 | head_cosine | 0.7938 | 0.7946 | 0.5319 to 0.9110 |
| 575 | embedding_cosine | 0.0079 | -0.0157 | -0.1785 to 0.2577 |
| 575 | loss_difference | 0.3280 | 0.3300 | 0.1086 to 0.6829 |

## Within-block variation

Nonoverlapping blocks of 32 starts; partial terminal blocks are excluded from these correlations. Vector variance includes magnitude and direction. Direction dispersion uses normalized gradients and therefore isolates directional spread.

| Target | Proxy | Mean Pearson | Mean Spearman | Spearman range |
|---|---|---:|---:|---:|
| full_variance | head_variance | 0.9436 | 0.9476 | 0.8514 to 0.9810 |
| full_variance | embedding_variance | 0.3754 | 0.4268 | 0.0083 to 0.7668 |
| full_variance | loss_variance | 0.0783 | 0.2301 | -0.1729 to 0.5134 |
| full_direction_dispersion | head_direction_dispersion | 0.8540 | 0.8161 | 0.7129 to 0.9089 |
| full_direction_dispersion | embedding_direction_dispersion | 0.3366 | 0.3301 | 0.0355 to 0.5329 |
| full_direction_dispersion | loss_variance | 0.0124 | 0.0162 | -0.3381 to 0.3787 |

## Checks and artifacts

Three full FP32 BP cache-alignment checks per trajectory and one analytic-head/autograd comparison per trajectory passed. See per-trajectory validation_and_timing.json for numerical errors. Cache checks use relative L2 <=0.01 and cosine >=0.9999; head derivative checks use relative L2 <=0.0001 for FP32 batched accumulation differences.

Artifacts under outputs/train/v0-gradient-proxy-test: correlations.csv (per trajectory), trajectory_equal_weight_summary.csv, cosine_calibration.csv (absolute cosine means/spread and proxy MAE), blocks.csv, and per-trajectory head_gradients.npy, window_features.npz, gram_matrices.npz, pair_metrics.npz, manifests and validation records.

## Direct compute cost

Matched batch-size-one measurements on 80 windows (8 from each of the ten trajectories), three repeats per window, give median wall times of 42.80 ms for full BP, 14.21 ms for output-head gradients, 11.35 ms for mean decoder embeddings, and 11.28 ms for forward loss. Paired median speedups over full BP are 2.93x, 3.72x, and 3.71x. Timing includes tokenization, model computation, CPU result transfer, and CUDA synchronization. Full protocol and per-track ranges: docs/results/v0-gradient-proxy-cost.md.

## Interpretation limits

Full and proxy dimensions differ: direct cross-space cosine was not computed. A strong correlation does not imply calibrated absolute cosine. The head proxy shares output-path information with the full gradient. Teacher-forced features require future labels. This study does not fit a mapping or evaluate CoVar allocation, held-out training outcomes, or matched proxy speed benchmarks. Any proxy selection from these trajectories needs held-out confirmation.
