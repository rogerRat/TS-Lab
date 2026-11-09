# v0-directional-marginal-gain

Last modified: 2026-11-08

## Question and fixed replay

Can block direction dispersion predict the actual reduction in directional reconstruction error from one additional full-gradient anchor, beyond predicting the starting error? Reuse the same 46 complete, nonoverlapping 256-start blocks from ten retained trajectories. For each block, compute true full-gradient direction dispersion over all 256 unit gradients, complete-gradient vector variance, and relative vector variance from the completed v0-gradient-variance-decomposition metrics. Calculate a sparse output-head proxy direction dispersion from eight evenly spaced head gradients at local positions 0,36,73,109,146,182,219,255. The eight proxy vectors are in a different parameter space; this statistic is an estimate of directional heterogeneity, not the complete-gradient direction dispersion itself.

For each budget n in {2,4,8,16}, place n uniformly spaced true full-gradient anchors including endpoints, using the existing `spaced` rule. Add one anchor at the integer midpoint of the largest consecutive anchor gap, with the leftmost gap winning ties; the midpoint uses floor division. The added positions are 127, 42, 54, and 8 for n=2,4,8,16. The selection rule is fixed independently of both direction dispersions and the resulting gain. Reconstruct all 256 gradients by linear interpolation before and after insertion. Define directional error per position as `1-cos(true,reconstructed)`, force observed anchor error to zero, and average over all 256 starts for `E_h(n)` and `E_h(n+1)`. The actual gain is `Delta E_h(n)=E_h(n)-E_h(n+1)`. A negative gain means the new anchor worsened the block average. Also report gain excluding only the new anchor, averaged over the other 255 positions, and gain over positions unqueried both before and after insertion.

## Meaning of the direction error and gain

For a start t between consecutive anchors a and b, linear reconstruction is `ghat_t=(1-alpha)g_a+alpha*g_b`, where `alpha=(t-a)/(b-a)`. At a queried anchor, `ghat_t=g_t`. The per-start direction error is

`e_t(n)=1-(g_t dot ghat_t)/(||g_t||_2 ||ghat_t||_2)`.

All cached true and reconstructed gradients have positive norms. The error is dimensionless and lies in [0,2]: zero means the two vectors point in the same direction, one means they are orthogonal, and two means they point in opposite directions. It measures directional alignment only; it does not measure the difference in gradient lengths or the full-vector L2 error. The block error is `E_h(n)=(1/256) sum_{t in h} e_t(n)` and includes the n queried anchors, each with zero error.

After adding one anchor at position c, the total marginal gain is `Delta E_h(n)=E_h(n)-E_h(n+1)`. Positive gain means the block-average direction error decreased, zero means no net change, and negative gain means it increased. This is an absolute reduction in the dimensionless error, not a percentage. The direct contribution of making c exact is `e_c(n)/256`. To separate that mechanical improvement from changes elsewhere, the nonnew-point gain is `(1/255) sum_{t in h, t != c} [e_t(n)-e_t(n+1)]`. A second diagnostic averages only positions that remain unqueried after insertion. A negative nonnew-point gain means the interpolation got worse on average away from the new anchor even if the total gain is positive.

Use the saved all-coordinate BF16 full-gradient and FP32 output-head Gram matrices; no model or gradient recomputation occurs. Compute pooled Pearson and Spearman correlations separately for each n across 46 blocks. Compute Spearman within each trajectory and average defined track values equally; track 00 has only two blocks and is excluded from that mean. Do not pool budgets for the primary correlations, since their gain scales differ. No independent-block p-values or causal claims are made.

Entry: `scripts/v0_directional_marginal_gain.py`. Environment: `rage` Conda. Command: `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src /root/autodl-tmp/rage/envs/rage/bin/python -u -m scripts.v0_directional_marginal_gain`. Output: `outputs/train/v0-directional-marginal-gain/`. Logs: `logs/v0-directional-marginal-gain/`.

## Starting error and gain

| Initial anchors n | New position | Mean E(n) | Mean E(n+1) | Mean total gain | Mean gain excluding new point | Blocks with negative nonnew gain |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 127 | 0.8176 | 0.7104 | 0.1072 | 0.1040 | 2/46 |
| 4 | 42 | 0.6382 | 0.5976 | 0.0407 | 0.0378 | 1/46 |
| 8 | 54 | 0.5066 | 0.4922 | 0.0144 | 0.0122 | 3/46 |
| 16 | 8 | 0.4003 | 0.3928 | 0.0075 | 0.0055 | 5/46 |

Mean total gain declines with larger starting budgets. The added point itself contributes mean `old_error[new]/256` of 0.0036, 0.0030, 0.0022, and 0.0020 in budget order. Most average improvement at n=2 and n=4 therefore occurs away from the newly queried point. Some individual blocks get worse even away from that point; at n=16 this occurs in five of 46 blocks. These are gains from one fixed largest-gap split, not optimized additions.

## Does high starting error itself predict gain?

| Initial anchors n | Pooled Spearman: E(n) versus total gain | Pooled Spearman: E(n) versus nonnew gain | Blocks with negative total gain | Blocks with negative nonnew gain |
|---:|---:|---:|---:|---:|
| 2 | 0.461 | 0.448 | 1/46 | 2/46 |
| 4 | 0.280 | 0.273 | 0/46 | 1/46 |
| 8 | 0.254 | 0.234 | 1/46 | 3/46 |
| 16 | 0.021 | -0.025 | 2/46 | 5/46 |

Higher current error has at most moderate association with this fixed one-anchor gain, and almost none at n=16. In the n=16 replay, trajectory 07 block 0 has the highest starting error among all 46 blocks (`E=0.5299`), yet the fixed new anchor increases its average error: total gain -0.00587 and nonnew-point gain -0.00860. This directly demonstrates that even the currently hardest block in this budget need not benefit from this particular fixed anchor. It does not show that every possible anchor choice in the block would fail.

## Does dispersion predict difficulty or gain?

Values below are pooled Spearman coefficients across 46 blocks at each fixed budget. The final column is the coefficient for gain averaged over the 255 positions other than the new anchor.

| Initial anchors n | Predictor | Starting error E(n) | Total marginal gain | Gain excluding new point |
|---:|---|---:|---:|---:|
| 2 | Full vector variance | -0.298 | -0.075 | -0.079 |
| 2 | Relative vector variance | 0.585 | -0.017 | -0.031 |
| 2 | True direction dispersion | 0.616 | -0.074 | -0.088 |
| 2 | Eight-head-proxy direction dispersion | 0.472 | -0.002 | -0.012 |
| 4 | Full vector variance | -0.324 | -0.175 | -0.183 |
| 4 | Relative vector variance | 0.639 | 0.147 | 0.148 |
| 4 | True direction dispersion | 0.701 | 0.142 | 0.137 |
| 4 | Eight-head-proxy direction dispersion | 0.489 | 0.380 | 0.370 |
| 8 | Full vector variance | -0.073 | -0.271 | -0.286 |
| 8 | Relative vector variance | 0.542 | 0.230 | 0.222 |
| 8 | True direction dispersion | 0.547 | 0.256 | 0.232 |
| 8 | Eight-head-proxy direction dispersion | 0.276 | 0.235 | 0.212 |
| 16 | Full vector variance | 0.069 | -0.046 | -0.061 |
| 16 | Relative vector variance | 0.392 | 0.050 | 0.022 |
| 16 | True direction dispersion | 0.395 | 0.060 | 0.034 |
| 16 | Eight-head-proxy direction dispersion | 0.131 | 0.056 | 0.032 |

The true direction dispersion ranks starting difficulty substantially better than marginal benefit at all four budgets. At n=4, its starting-error rho is 0.701 but its total-gain rho is only 0.142; the corresponding equal-track rhos are 0.689 and 0.211. At n=2, both true and sparse proxy dispersion have nearly zero or negative association with gain. At n=8 the associations become weakly positive, and by n=16 they are near zero again. A high-dispersion block can remain hard to reconstruct while a fixed extra anchor offers little improvement.

The sparse proxy direction dispersion correlates with true direction dispersion across the 46 blocks at Pearson 0.867 and Spearman 0.703. Its best observed total-gain association is at n=4 (rho 0.380, equal-track rho 0.344); for nonnew-point gain, rho is 0.370. Leave-one-track-out pooled nonnew-gain rho remains 0.303-0.447 at n=4. This is a budget-specific moderate signal, not consistent evidence across n=2,8,16. At n=8, sparse-proxy total-gain rho is 0.235 and equal-track rho 0.311; at n=16 it is 0.056 and 0.089. The preselected largest-gap insertion might interact with the proxy sampling positions, so the n=4 result should not be interpreted as proof that a proxy-guided allocation policy improves reconstruction.

## Interpretation and artifacts

For these fixed ten trajectories, direction dispersion describes current reconstruction difficulty better than the measured one-anchor gain. High current error also does not guarantee benefit: the n=16 block with the highest starting error had a negative total gain under the fixed insertion rule. The evidence does not yet justify allocating more anchors solely by direction dispersion. The n=4 sparse proxy result is a candidate for a separate fixed-budget selection test, with matched uniform controls and held-out evaluation; this replay does not test such a policy. The 46 blocks come from overlapping raw windows within a selected set of tracks, and nine within-track correlations have only four or five blocks each.

`block_budget_metrics.csv` has all 184 block-budget rows, exact anchor positions, E(n), post-error, direct new-point gain, nonnew gain, and unqueried-position gain. `correlations.csv` has 64 budget-predictor-target associations, including pooled Pearson, pooled Spearman, and equal-track Spearman. `budget_summary.csv`, `proxy_direction_dispersion.csv`, `proxy_association.json`, `leave_one_track_out.csv`, and `input_manifest.json` provide the remaining checks. The output `.complete` marker is present.
EOF'