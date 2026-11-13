# v0-vage: VAGE-Static reconstruction replay

Last modified: 2026-11-08

## Protocol

Ten retained M5 trajectories contain 12,836 legal starts at the fixed warmed Chronos-T5 Tiny checkpoint. Each window spans 576 observations (512 context plus 64 prediction), so the legal start count is N=T-575. The replay uses retained BF16 full-gradient caches and their full-coordinate and output-head Gram matrices. Each trajectory is independent.

Both full-gradient and output-head proxy budgets are ceil(N/64). Nonoverlapping blocks contain at most 512 consecutive starts; terminal partial blocks remain. The proxy budget is allocated approximately by block length with at least two points per block. Proxy points are geometrically spaced and unique. The proxy score is unbiased sample trace variance from the output-head Gram submatrix. The prior is the length-weighted average proxy variance of all blocks in the trajectory. For p proxy points, the final score is w times the block variance plus (1-w) times the prior, with w=min(1,(p-1)/7). This shrinks blocks with fewer than eight proxy points. Scores remain fixed during full-gradient allocation.

The configurable n_min is two in this run, so initial full-gradient anchors are both endpoints of each block. Remaining anchors go to the non-full block maximizing L_h V_h/[n_h(n_h+1)]. Ties go to the lower block index. Within that block, the midpoint of the largest uncovered anchor gap is chosen; ties go to the leftmost gap. The exact full-gradient budget is used. Uniform uses the same number of anchors evenly spaced over the complete trajectory, including its endpoints. Both methods use linear interpolation between anchors and reconstruct all legal starts. Primary metrics are mean L2 and mean cosine; secondary metrics include relative RMSE, tails, and non-anchor means.

Selection reads only sampled proxy Gram entries. Full-gradient Gram entries are accessed only for evaluation. This is retrospective replay: the call counts are prospective costs, while no new model forward or backward calls occurred. Uniform pays no proxy calls, so this is not a matched wall-clock comparison. The ten selected late-manifest trajectories and overlapping windows limit generalization.

## Results

| Equal-trajectory mean metric | VAGE-Static | Uniform | VAGE minus Uniform | VAGE wins / 10 |
|---|---:|---:|---:|---:|
| Mean L2, lower better | 3.5437 | 3.4824 | +0.0613 | 2 |
| Mean cosine, higher better | 0.3651 | 0.3971 | -0.0319 | 1 |
| Relative RMSE, lower better | 1.0079 | 0.9953 | +0.0126 | 2 |
| P95 L2, lower better | 6.3402 | 6.4168 | -0.0765 | 5 |
| P05 cosine, higher better | -0.0754 | 0.0150 | -0.0905 | 2 |
| Non-anchor mean L2, lower better | 3.6011 | 3.5388 | +0.0623 | 2 |
| Non-anchor mean cosine, higher better | 0.3549 | 0.3873 | -0.0324 | 1 |

VAGE-Static accounts for 204 full-gradient and 204 output-head proxy calls. Uniform accounts for 204 full-gradient and zero proxy calls. Under this fixed configuration, VAGE-Static does not improve the primary pointwise reconstruction metrics. Its modest average P95 L2 advantage is inconsistent across tracks and accompanies worse mean error and lower-tail cosine. This tests the complete static allocation and longest-gap placement policy, not variance-guided methods generally.

Proxy variance ranks block dispersion, but may not predict the improvement from one more interpolation anchor. Forced endpoints at 512-start boundaries also change spacing relative to whole-trajectory Uniform. These are hypotheses, not established causes.

## Validation and artifacts

Ten proxy markers, Gram shapes and finite values, full-gradient cache markers, and exact budgets were checked. Three direct BF16-cache interpolation errors on trajectory 00 were compared with Gram-based errors; absolute differences were 0.00239, 0.00295, and 0.00247 L2 on errors from 3.78 to 4.75. The Gram matrices derive from tiled products of BF16 vectors, so minor numerical differences from direct calculations are expected.

Config: configs/rage/v0-vage.json. Entry: scripts/v0_vage_static.py. Raw outputs: outputs/train/v0-vage/trajectory_metrics.csv, block_allocation.csv, anchors.json, summary.json, and .complete. Logs: logs/v0-vage/.

## n_min=1 sensitivity replay

Last modified: 2026-11-08

A separate replay used configs/rage/v0-vage-nmin1.json and preserved the n_min=2 artifacts. Each block initially receives one center anchor. If a selected block still has only one anchor, the next point is the farther block endpoint, with the lower endpoint on a tie. Subsequent points split the largest internal gap. For starts outside the global anchor range, reconstruction uses the nearest observed anchor. All budgets, proxy scores, and the Uniform baseline are otherwise identical.

| Equal-trajectory mean metric | VAGE n_min=1 | VAGE n_min=2 | Uniform |
|---|---:|---:|---:|
| Mean L2, lower better | 3.8747 | 3.5437 | 3.4824 |
| Mean cosine, higher better | 0.3156 | 0.3651 | 0.3971 |
| Relative RMSE, lower better | 1.1267 | 1.0079 | 0.9953 |
| P95 L2, lower better | 7.2953 | 6.3402 | 6.4168 |
| P05 cosine, higher better | -0.1724 | -0.0754 | 0.0150 |

All ten n_min=1 trajectories lack the first global start among final anchors; four also lack the last global start. This produces nearest-anchor extrapolation at uncovered ends and is a material difference from n_min=2. The result does not isolate the effect of reducing the per-block minimum from the effect of losing endpoint coverage. The n_min=1 design is worse than both n_min=2 and Uniform on the primary metrics. Raw outputs: outputs/train/v0-vage-nmin1/.

## Additional n_min=1 methods

Last modified: 2026-11-08

All methods use Bg=Bp=ceil(N/64), 512-start blocks, one initial anchor per block, and the same whole-trajectory linear interpolation evaluator. Proxy-based methods use the same sparse proxy samples and shrinkage rule as VAGE-Static. Quota allocates integer counts using weights L_h sqrt(V_h+1e-12) and places the resulting anchors evenly within each block. Fusion uses kappa=8 and a median true-to-proxy block-variance ratio fitted on the other nine trajectories for each held-out track; it updates the block sample variance using only selected full-gradient anchors. Oracle uses the population full-gradient block variance from all cached starts and is nondeployable. The unchanged Static and Uniform anchor sets match the preceding replay exactly.

| Method | Mean L2, lower better | Mean cosine, higher better | Relative RMSE |
|---|---:|---:|---:|
| Uniform | 3.4824 | 0.3971 | 0.9953 |
| VAGE-Quota | 3.5031 | 0.3675 | 0.9967 |
| VAGE-Residual diagnostic | 3.5159 | 0.3801 | 1.0035 |
| VAGE-Static | 3.8747 | 0.3156 | 1.1267 |
| VAGE-Fusion | 3.8762 | 0.3163 | 1.1273 |
| Oracle-VAGE | 3.8774 | 0.3163 | 1.1275 |

No method exceeds Uniform on either primary equal-track mean. Quota is nearest on L2. Oracle uses true block variance but performs similarly to Static under the same one-anchor and gap-splitting policy. This shows that better block variance alone does not rescue this particular reconstruction policy, although missing endpoint coverage remains a confounder.

Residual has an important protocol limitation. To avoid uncounted proxy calls, candidate anchors and initial anchors were restricted to the Bp observed proxy locations. Because Bp=Bg in every trajectory, all proxy locations necessarily become full-gradient anchors by the end. Its final anchor set therefore equals the sparse proxy sample set regardless of the residual acquisition order. The Residual row is a stratified-proxy-anchor diagnostic, not a valid test of adaptive residual selection. A meaningful Residual comparison needs a candidate proxy pool larger than Bg, a lower Bg, or a surrogate that scores unobserved candidate positions without extra proxy calls.

All proxy-based methods account for 204 full-gradient and 204 proxy calls in total. Uniform accounts for 204 full-gradient and no proxy calls. Oracle reads all cached full gradients and has no deployable cost interpretation. Full-gradient truth in the Fusion calibration comes only from the other nine tracks for each held-out trajectory; the ten-track set was already used to develop these hypotheses, so this is exploratory rather than independent confirmation. Raw artifacts: outputs/train/v0-vage-methods-nmin1/. Entry: scripts/v0_vage_methods.py.

## Pair-mean reconstruction supplement

Last modified: 2026-11-08

This supplement freezes every previously selected anchor set and every budget. It changes only the reconstruction rule. For an unobserved start strictly between adjacent full-gradient anchors a and b, predict (g_a+g_b)/2 at every start in that gap. Queried anchors return their observed true gradient exactly. Outside the global anchor range, as in the n_min=1 replay, use the nearest anchor. This is one constant representative per anchor gap, not one representative per fixed 512-start VAGE allocation block. The same full-coordinate BF16-derived Gram matrices evaluate both reconstruction rules. No model or proxy calls were added.

| Fixed anchor selection | Pair-mean L2 | Linear L2 | Pair-mean cosine | Linear cosine |
|---|---:|---:|---:|---:|
| Uniform, 204 anchors | 3.5445 | 3.4824 | 0.3498 | 0.3971 |
| VAGE-Static n_min=2 | 3.5880 | 3.5437 | 0.3205 | 0.3651 |
| VAGE-Static n_min=1 | 3.9042 | 3.8747 | 0.2923 | 0.3156 |
| VAGE-Quota n_min=1 | 3.5464 | 3.5031 | 0.3240 | 0.3675 |
| VAGE-Fusion n_min=1 | 3.9030 | 3.8762 | 0.2944 | 0.3163 |
| Oracle-VAGE n_min=1 | 3.9064 | 3.8774 | 0.2928 | 0.3163 |
| VAGE-Residual diagnostic n_min=1 | 3.5699 | 3.5159 | 0.3338 | 0.3801 |

Values are equal-trajectory means on the same ten retained trajectories. The duplicate Uniform and Static anchor sets from the earlier n_min=1 files produce identical rows and are deduplicated above. At this budget, the pair-mean rule worsens both primary metrics for every listed allocation on average. Quota remains closest to Uniform in L2 but does not exceed it. The original endpoint-coverage confound for n_min=1 and the Residual candidate-pool degeneracy remain.

Raw per-trajectory and summary tables: outputs/train/v0-vage/pairmean/. Shared config: configs/rage/v0-pairmean-supplements.json. Entry: scripts/evaluate_pairmean_reconstruction.py. The pair-mean output has its own .complete marker.
