# v0-var-proxy-estimate

Last modified: 2026-11-08

## Question and protocol

For each of ten retained continuous trajectories, divide the legal window-start sequence into consecutive nonoverlapping complete 256-start blocks. The ten terminal fragments are excluded from correlation analysis. In each complete block, use eight evenly spaced output-head gradient proxies at local positions 0, 36, 73, 109, 146, 182, 219, and 255. The proxy statistic is the population vector variance, mean squared norm minus squared norm of the mean, computed from the saved head-gradient Gram matrix.

Use four evenly spaced full-gradient anchors at positions 0, 85, 170, and 255. Reconstruct every one of the 256 full gradients by linear interpolation between neighboring anchors; anchor gradients are exact. For each block, average over all 256 positions: (1) L2 norm of reconstructed minus true gradient, (2) directional cosine, and (3) absolute difference between reconstructed and true gradient norms. Also record mean 1-cosine as a direction-error reading of the same cosine measure. The cached all-coordinate full-gradient Gram matrices provide exact inner products for this retrospective replay of BF16 gradient copies; no new model or gradient calls were made. Both proxy and full-gradient values were originally computed for all legal starts, so the replay measures association under the stated sparse budgets, not measured runtime savings.

Compute Pearson and Spearman correlations across 46 complete blocks and separately within each trajectory. The equal-track mean averages defined within-track correlations; trajectory 00 has only two complete blocks, so its within-track correlations are undefined and excluded from that mean. No pair-independent significance test is used.

Config: `configs/rage/v0-var-proxy-estimate.json`. Entry: `scripts/v0_var_proxy_estimate.py`. Output: `outputs/train/v0-var-proxy-estimate/`. Log: `logs/v0-var-proxy-estimate/`. Environment: `rage` Conda. Run command: `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src /root/autodl-tmp/rage/envs/rage/bin/python -u -m scripts.v0_var_proxy_estimate --config configs/rage/v0-var-proxy-estimate.json`.

## Observed associations

| Block target | Pooled Pearson | Pooled Spearman | Equal-track Pearson | Equal-track Spearman |
|---|---:|---:|---:|---:|
| Mean L2 error | 0.8568 | 0.8538 | 0.9170 | 0.8444 |
| Mean directional cosine | 0.1767 | 0.3977 | 0.3443 | 0.3444 |
| Mean 1-cosine | -0.1767 | -0.3977 | -0.3443 | -0.3444 |
| Mean absolute norm difference | 0.5905 | 0.4970 | 0.5900 | 0.5111 |

The sampled head-gradient variance strongly ranks mean full-vector L2 reconstruction error in these blocks. It has a moderate positive association with absolute norm difference. Directional cosine instead has a weak-to-moderate positive association, opposite to the prediction that larger proxy variance always implies worse directional reconstruction. L2 combines magnitude and direction, so the strong L2 association does not establish directional error prediction. Within-track estimates use only four or five blocks for nine tracks and are unstable; the pooled result can also reflect between-track differences. These ten selected M5 tail trajectories do not establish generalization to other trajectories, models, or datasets.

Raw block-level values, per-track correlations, exact positions, and excluded terminal lengths are in `outputs/train/v0-var-proxy-estimate/`. The output `.complete` marker is present.

## Complete full-gradient variance supplement

Last modified: 2026-11-08

For the identical 46 complete blocks, calculate the population vector variance from all 256 cached full gradients: mean squared norm minus squared norm of the block mean. Reuse the original three reconstruction metrics without recalculating or changing the four anchor positions. This is an oracle descriptive comparison because the variance requires all 256 true gradients. The 46 values matched the prior 256-point full-variance rows in `outputs/train/v0-gradient-proxy-test/local_risk_sparse_long/sampled_block_statistics.csv` exactly.

| Target correlated with full variance | Pooled Pearson | Pooled Spearman | Equal-track Pearson | Equal-track Spearman |
|---|---:|---:|---:|---:|
| Mean L2 error | 0.9291 | 0.9463 | 0.9596 | 0.9111 |
| Mean directional cosine | 0.1216 | 0.3240 | 0.3726 | 0.4444 |
| Mean 1-cosine | -0.1216 | -0.3240 | -0.3726 | -0.4444 |
| Mean absolute norm difference | 0.7015 | 0.6381 | 0.7145 | 0.6222 |
| Eight-proxy variance | 0.9191 | 0.9169 | 0.9288 | 0.8556 |

Full variance has a stronger association than eight-proxy variance with mean L2 error and absolute norm difference. It also has a positive association with directional cosine, so even true vector variance does not imply worse directional reconstruction here. The L2 and variance statistics both depend on full-gradient scale and share the same 256 target vectors, which may contribute to their association. The oracle full variance is unavailable at the sparse cost of eight proxies. Within-track estimates still have only four or five blocks for nine trajectories; trajectory 00 has two and is excluded from the equal-track mean.

Entry: `scripts/v0_true_var_reconstruction.py`. Output: `outputs/train/v0-var-proxy-estimate/true_variance/`, containing block metrics, correlations, input manifest, and `.complete`.

## Four-anchor mean as one constant block representative

Last modified: 2026-11-08

Use the same four full-gradient anchors at local positions 0, 85, 170, and 255, but take their arithmetic vector mean and apply that single vector to all 256 starts in the block. The four anchor starts are evaluated against the mean as well; they are no longer exact reconstruction points. Keep the same 46 blocks, eight-proxy variance, and all-256 true variance. Average each of the same three metrics within each block, then correlate them with both variance predictors. This is a cached full-gradient Gram replay, with no new model gradient calls.

| Variance predictor | Target | Pooled Pearson | Pooled Spearman |
|---|---|---:|---:|
| Eight-proxy variance | Mean L2 error | 0.8946 | 0.9126 |
| Eight-proxy variance | Mean directional cosine | 0.3097 | 0.3374 |
| Eight-proxy variance | Mean absolute norm difference | 0.6899 | 0.6444 |
| All-256 true variance | Mean L2 error | 0.9623 | 0.9873 |
| All-256 true variance | Mean directional cosine | 0.2344 | 0.2670 |
| All-256 true variance | Mean absolute norm difference | 0.8125 | 0.8078 |

Equal-block mean L2 is 3.5273 for the constant four-anchor mean versus 3.5127 for the prior linear reconstruction. Mean directional cosine is 0.2519 versus 0.3618, and mean absolute norm difference is 1.6817 versus 1.0832. These aggregate values describe the same 46 blocks; the four sampled anchors are scored as approximations in the constant-mean method, whereas linear reconstruction keeps them exact.

The very high association between true variance and L2 error is partly structural. For any fixed block representative `m`, the average squared L2 error equals the true population vector variance plus the squared distance between `m` and the true block mean. The reported L2 target is the average unsquared norm, so this identity does not make its correlation exactly one, but it explains why a strong association is expected. Directional cosine remains weakly positively associated with both variance predictors. Variance does not by itself isolate directional variation. Per-track correlations use only four or five blocks for nine tracks and are unstable; trajectory 00 has two blocks and is omitted from the equal-track mean.

Entry: `scripts/v0_four_anchor_mean.py`. Output: `outputs/train/v0-var-proxy-estimate/four_anchor_mean/`, containing block metrics, per-track and pooled correlations, summary, input manifest, and `.complete`.
