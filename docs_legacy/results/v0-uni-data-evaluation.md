# v0-uni-data-evaluation: lag differences in reference-gradient scores

Last modified: 2026-11-08

## Protocol

The inputs are the same ten retained M5 trajectories and complete BF16 gradients at the fixed warmed Chronos-T5 Tiny checkpoint. Each trajectory contributes ten seeded random legal starts, yielding 100 references total. Starts within a trajectory are separated by at least two legal-start indices; the draw uses seed 0. All 100 selected starts are removed from the query set and from either endpoint of every lag pair. The remaining 12,736 legal starts are scored against all 100 references, including cross-trajectory pairs. For each query-reference pair, compute full-vector cosine and unnormalized dot product using all 8,394,496 coordinates. No output-head proxy is used.

For lag k, compare query starts i and i+k from the same trajectory against the same reference. The two statistics are absolute score difference |a-b| and relative score difference |a-b|/((|a|+|b|)/2). The mean absolute magnitude in the denominator is used because cosine and dot products can be negative. This ratio lies in [0,2] for finite nonzero scores; a sign flip gives 2. Record zero denominators and sign flips. Aggregate each statistic within a trajectory over all valid lag pairs and references, then average the ten trajectory summaries equally. Save per-trajectory results for lags 1-575, full score arrays, and lag curves. Pair counts describe dependent observations, not independent statistical replicates.

## Results

| Lag | Cosine mean absolute difference | Cosine mean relative difference | Dot mean absolute difference | Dot mean relative difference | Sign flips |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0402 | 0.4856 | 0.6100 | 0.5372 | 9.40% |
| 2 | 0.0518 | 0.5924 | 0.7804 | 0.6581 | 12.04% |
| 4 | 0.0618 | 0.6761 | 0.9172 | 0.7537 | 14.45% |
| 8 | 0.0725 | 0.7541 | 1.0750 | 0.8415 | 17.17% |
| 16 | 0.0872 | 0.8479 | 1.2752 | 0.9437 | 21.25% |
| 32 | 0.1088 | 0.9734 | 1.6026 | 1.0817 | 27.36% |
| 64 | 0.1411 | 1.1524 | 2.1143 | 1.2668 | 36.91% |
| 128 | 0.1562 | 1.2365 | 2.3815 | 1.3498 | 41.20% |
| 256 | 0.1666 | 1.2775 | 2.4590 | 1.3860 | 43.86% |
| 575 | 0.1732 | 1.2965 | 2.6590 | 1.3962 | 44.70% |

## Score distribution

The following statistics use all 1,273,600 valid query-reference scores (12,736 non-reference queries times 100 references). Means are signed; variance is population variance with denominator equal to the score count. The pooled column weights every score equally. The equal-track column averages ten within-track statistics equally.

| Score | Pooled signed mean | Pooled signed median | Pooled variance | Pooled mean absolute value | Pooled median absolute value | Equal-track mean | Equal-track mean within-track variance |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full-gradient cosine | 0.0284 | 0.0284 | 0.0283 | 0.1378 | 0.1207 | 0.0293 | 0.0277 |
| Full-gradient dot product | 0.6101 | 0.1635 | 9.1331 | 1.8782 | 1.0334 | 0.6300 | 9.0371 |

The cosine signed mean and median are both about 0.028; the signed mean near zero partly reflects cancellation between positive and negative scores. The dot signed median of 0.163 is substantially below its mean of 0.610, showing that its positive tail raises the mean. The cosine mean absolute value of 0.138 and median absolute value of 0.121 show that its typical magnitude is also small. Thus a lag-1 mean absolute cosine change of 0.040 is small on the global cosine scale but material relative to typical score magnitude. The dot product has a signed mean of 0.610 and a much wider scale (population variance 9.133); it should be interpreted on its own scale. The mean lag-1 relative difference is larger than the ratio of global means because the relative ratio is computed separately for each pair before averaging, and small-denominator pairs and sign flips contribute strongly.

The lag-1 line contains 1,262,600 query-reference score comparisons, corresponding to 12,626 valid within-trajectory query-start pairs. Across the ten trajectories, lag-1 mean relative difference ranges from 0.364 to 0.600 for cosine and 0.418 to 0.642 for dot product. The median relative difference averaged across tracks is 0.205 for cosine and 0.268 for dot product at lag 1. No zero denominators occurred in the reported lag summary.

The scores change less at lag 1 than at longer lags, but relative changes are not negligible even between adjacent legal starts. The sign-flip rate is about 9.4% at lag 1. Thus the empirical score sequences show local association but do not support treating adjacent score values as approximately constant under this metric. Absolute differences are also relevant: median absolute score magnitude across all valid query-reference pairs is 0.121 for cosine and 1.033 for dot product. The relative ratio is sensitive to low-magnitude scores; the absolute-difference and sign-flip curves should be read alongside it.

A score sequence against one reference is not the same as the full-gradient vector sequence. This experiment does not establish whether a particular downstream ranking or top-k decision remains stable between adjacent starts. References and queries can have overlapping 576-observation windows, and pair observations are strongly dependent. The ten trajectories are the fixed selected late-manifest sample, not a population-wide estimate.

## Validation and artifacts

Validated ten completed BF16 caches and file sizes, ten references per track, no adjacent reference starts, no reference starts among valid queries or lag endpoints, finite scores for all valid queries, and 11,500 per-trajectory/lag/metric rows. The complete 1-575 lag table is in outputs/train/v0-uni-data-evaluation/lag_summary.csv; per-track values are in trajectory_lag.csv. References are in references.json, per-track score matrices in trajectory-NN-scores.npy, and the four lag curves in lag_difference.png. Config: configs/rage/v0-uni-data-evaluation.json. Entry: scripts/v0_uni_data_evaluation.py. Log: logs/v0-uni-data-evaluation/. The successful replay completed in about 11.8 minutes; the first launch stopped after track 00 scoring due to a shape-unpacking bug, which was repaired before the successful restart.
