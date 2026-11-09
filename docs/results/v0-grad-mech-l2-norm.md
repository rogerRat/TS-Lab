# Relative lag-L2 gradient distance

Last modified: 2026-11-08

Status: completed. This is the v0-grad-mech-l2-norm supplement on the ten retained M5 trajectories only: source manifest ordinals 973, 980, 981, 984, 985, 987, 988, 996, 998, and 999. It covers 12,836 legal windows at the same fixed warmed ChronosTiny checkpoint. The original 1,000-trajectory grad-mech result remains separate.

## Metric and source

For each pair of gradients g_i and g_(i+k), R = ||g_i - g_(i+k)||_2 / ((||g_i||_2 + ||g_(i+k)||_2)/2). R=0 means identical gradients; R=1 means their difference equals the average gradient norm; R is at most 2. Pairs are within the same trajectory, for every lag from 1 to 575. Lag 1 means 575/576 = 99.83% window-span overlap.

All 8,394,496 gradient coordinates are included. The source gradients were computed in FP32 and retained as BF16. The supplement uses the existing full-gradient Gram matrices derived from those BF16 caches. The Gram construction used FP32 tiled matrix products with FP64 accumulation across coordinate tiles. Squared L2 is G_ii + G_jj - 2G_ij, and the denominator uses sqrt(G_ii) and sqrt(G_jj). No new model gradient was computed.

## Results

The primary lag-1 result has 12,826 pairs. Its pair-weighted mean R is 0.6372, median 0.7620, RMS 0.7771, p90 1.1571, and p95 1.2411. The mean of the ten trajectory means is 0.6462; individual trajectory means range from 0.4801 to 0.8052.

Descriptively, 1563/12826 (12.1862%) lag-1 pairs have R<0.1, while 3951/12826 (30.8046%) have R>=1. These are descriptive reference levels, not pre-registered hypothesis thresholds.

Selected pair-weighted lag summaries:

| Lag | Overlap | Pairs | Mean R | Median R | P90 R | P95 R | Equal-track mean R |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 99.83% | 12826 | 0.6372 | 0.7620 | 1.1571 | 1.2411 | 0.6462 |
| 2 | 99.65% | 12816 | 0.8194 | 0.9658 | 1.2151 | 1.2971 | 0.8258 |
| 4 | 99.31% | 12796 | 0.9671 | 1.0502 | 1.2707 | 1.3475 | 0.9709 |
| 8 | 98.61% | 12756 | 1.0813 | 1.1130 | 1.3332 | 1.4156 | 1.0834 |
| 16 | 97.22% | 12676 | 1.1790 | 1.1823 | 1.4302 | 1.5193 | 1.1804 |
| 32 | 94.44% | 12516 | 1.2894 | 1.2755 | 1.5468 | 1.6198 | 1.2890 |
| 64 | 88.89% | 12196 | 1.4106 | 1.4054 | 1.6451 | 1.7137 | 1.4074 |
| 128 | 77.78% | 11556 | 1.4505 | 1.4432 | 1.6668 | 1.7595 | 1.4452 |
| 256 | 55.56% | 10276 | 1.4807 | 1.4698 | 1.7125 | 1.8059 | 1.4712 |
| 512 | 11.11% | 7716 | 1.4869 | 1.4778 | 1.6920 | 1.7638 | 1.4707 |
| 575 | 0.17% | 7086 | 1.4906 | 1.4870 | 1.6764 | 1.7609 | 1.4684 |

## Conclusion

The empirical local relative-smoothness hypothesis is contradicted on these ten retained trajectories. Even at lag 1, adjacent windows share 575 of 576 time points (99.83% span overlap), yet the mean of the pairwise relative full-gradient L2 distances is 0.6372: the gradient-difference norm is, on average in this pairwise normalized sense, about 64% of the two gradients' mean norm. The median is 0.7620; 55.01% of lag-1 pairs have R>=0.6 and 30.80% have R>=1. This is a common large relative change in this selected sample, although the ten trajectory means vary from 0.4801 to 0.8052.

For the same ten trajectories and the same 12,826 lag-1 pairs, the pair-weighted cosine mean reconstructed from their cached full-gradient Gram matrices is 0.7215 (equal-trajectory mean 0.7173). The original 1,000-trajectory grad-mech cosine mean was 0.7539 on a different scope. A cosine around 0.7 describes partial directional alignment; it does not establish a small L2 change of the complete gradients. The earlier interpretation that lag-1 directional similarity supports local relative smoothness is therefore not supported by the direct relative L2 measurement.

This conclusion concerns small relative gradient changes under the fixed model, checkpoint, loss, preprocessing, and ten selected late-manifest trajectories. It does not prove that the gradient map is mathematically non-Lipschitz: such a claim needs a specified input-space distance and bound. The windows share raw time points, but the shifted context and target can still produce substantially different gradients.

## Checks and scope

All ten source manifests, BF16 cache completion markers and file sizes, and Gram matrix shapes, finite entries, symmetry, and positive diagonals passed validation. Thirty sampled pair distances were recomputed directly from the BF16 caches (three per trajectory); the maximum absolute difference from the Gram-derived L2 was 0.00001966. No negative squared distances required clipping. The 5750 trajectory-lag rows and 575 pooled lag rows have finite metrics; lag-1 pair count matches the 12,826 saved values.

The ten trajectories were selected from the unprocessed tail of the original fixed 1,000-trajectory manifest for gradient retention. They are not a representative rerun of all 1,000 trajectories. Window pairs overlap and are dependent, so the pair count must not be read as an independent sample size or used for pair-independent p-values. BF16 retention and Gram multiplication introduce limited numerical error; the direct-pair checks assess this computation but do not restore the original FP32 gradients. Window overlap fraction is not an input-space distance, so R by itself does not prove Lipschitz continuity.

Archived result artifacts: archive/v0-grad-mech-test/outputs/train/v0-grad-mech-l2-norm/ (lag1_pairs.csv, lag_summary.csv, trajectory_lag.csv, summary.json, audit.json, config.json, and .complete). The original outputs/train/v0-grad-mech-l2-norm/ path is a symbolic link to this archive. The ten BF16 gradient caches remain in outputs/train/v0-crb-reconst-test/gradients/ and were not archived. Execution logs: logs/v0-grad-mech-l2-norm/. Protocol: docs/EXP.md. Implementation: scripts/v0_grad_mech_l2_norm.py.
