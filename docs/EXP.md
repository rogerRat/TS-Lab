# 实验

最后修改日期：2026-11-09

## Experiment and batch overview

This table covers every maintained experiment and batch in this document. Completed rows report observed outcomes; terminated and unstarted rows remain visible. Detailed protocols and limitations are in the sections below and in docs/RESULTS.md.

| Experiment | Batch | Work | Status | Actual result |
|---|---|---|---|---|
| v0-crb-var | 1-ten-trajectory-replay | Compare CRB adaptive intervals with matched uniform intervals at one gradient per about 64 starts | Completed | 10/10 budget-verified; relative variance 0.6410 vs 0.6802; direction dispersion 0.4051 vs 0.4402 |
| v0-active-value-rank | 1-expanded-cross-rank | Expanded cross-track proxy ranking | Completed | 100 references; 714-766 candidates each; mean rho 0.7104 |
| v0-active-value-rank | 2-active-correction | Budgeted active value-ranking correction | Completed | At 10% and B=64, boundary regret 0.04192 vs 0.05148 highest-proxy |
| v0-active-value-rank | 3-proxy-coverage | Diagnose true top-k proxy positions across 100 references | Completed | Mean recall at k/2k/3k: 0.4023/0.6480/0.7950; 65 references have true top-k beyond 5k |
| v0-active-value-rank | 4-equal-budget-query | Compare pure boundary, proxy high, random, existing complex at B=0,8,16,32,64 | Completed | At B=64, recall: boundary 0.5064, high 0.4776, random 0.4295, complex 0.5415 |
| v0-directional-marginal-gain | 1 | Validate 46 blocks, full/head Gram caches, and prior dispersion | Completed | 46 fixed blocks; true dispersion matched prior results |
| v0-directional-marginal-gain | n=2 | Add fixed largest-gap anchor and measure direction-error gain | Completed | 46 block rows; mean total gain 0.1072 |
| v0-directional-marginal-gain | n=4 | Add fixed largest-gap anchor and measure direction-error gain | Completed | 46 block rows; mean total gain 0.0407 |
| v0-directional-marginal-gain | n=8 | Add fixed largest-gap anchor and measure direction-error gain | Completed | 46 block rows; mean total gain 0.0144 |
| v0-directional-marginal-gain | n=16 | Add fixed largest-gap anchor and measure direction-error gain | Completed | 46 block rows; mean total gain 0.0075 |
| v0-directional-marginal-gain | 6 | Compare four predictors with difficulty and actual gains; report | Completed | 64 correlation rows, 160 leave-one-track-out rows, .complete |
| v0-gradient-variance-decomposition | 1 | Validate 46 fixed blocks, prior metrics, full Gram caches | Completed | All blocks matched prior full variance and reconstruction metrics |
| v0-gradient-variance-decomposition | 2 | Compute magnitude, direction, and relative variance statistics | Completed | 46 block rows; exact amplitude/direction component decomposition |
| v0-gradient-variance-decomposition | 3 | Correlate dispersion and reconstruction error; exclude four anchors | Completed | 30 association rows; 252 non-anchor starts per block |
| v0-gradient-variance-decomposition | 4 | Control mean norm and trajectory, draw track-colored scatter plots | Completed | Two six-panel PNG/SVG figures; full-variance/cosine controlled rank r=-0.431 |
| v0-gradient-variance-decomposition | 5 | Validate and report | Completed | Result report, docs, and .complete present |
| v0-grad-traj-level-norm-gap | 1 | Validate ten full/head Gram cache pairs and all legal windows | Completed | Ten pairs validated; 12,836 windows total |
| v0-grad-traj-level-norm-gap | 2 | Compute track-level norms of mean gradient and mean per-window norms | Completed | Twenty track-kind rows; full span 1.9345x, head span 2.4960x for primary norm |
| v0-grad-traj-level-norm-gap | 3 | Compare spreads and full/head trajectory ranks | Completed | Primary full/head Spearman 0.8788; report and .complete written |
| v0-var-proxy-estimate | 1 | Validate ten retained full/head Gram caches and fixed block positions | Completed | Ten trajectory caches validated; 46 complete 256-start blocks |
| v0-var-proxy-estimate | 2 | Replay eight-proxy variance and four-anchor reconstruction | Completed | 46 block metrics written; ten terminal fragments excluded |
| v0-var-proxy-estimate | 3 | Correlate three reconstruction measures with proxy variance and report | Completed | Pooled Spearman: L2 0.8538, cosine 0.3977, norm difference 0.4970 |
| v0-var-proxy-estimate | 4 | Correlate complete 256-gradient variance with the same reconstruction measures | Completed | Pooled Spearman: L2 0.9463, cosine 0.3240, norm difference 0.6381 |
| v0-var-proxy-estimate | 5 | Replace linear reconstruction with the four-anchor vector mean | Completed | True-variance Spearman: L2 0.9873, cosine 0.2670, norm difference 0.8078 |
| v0-uni-data-evaluation | 1 | Validate caches and draw 100 nonadjacent references | Completed | Ten references per trajectory; 100 total |
| v0-uni-data-evaluation | 2 | Score all non-reference starts against references | Completed | 12,736 queries and 1,273,600 scores per metric |
| v0-uni-data-evaluation | 3 | Aggregate lag 1-575 differences and plot | Completed | 11,500 trajectory-lag-metric rows; lag-1 relative cosine/dot differences 0.4856/0.5372 |
| v0-uni-data-evaluation | 4 | Validate and report results | Completed | Score distribution, lag curves, report, and .complete recorded |
| v0-uni-reconst-scale | s=1 | Uniform linear reconstruction | Completed | 12,836 anchors; mean L2 0.0000; cosine 1.0000 |
| v0-uni-reconst-scale | s=2 | Uniform linear reconstruction | Completed | 6,420 anchors; mean L2 0.9650; cosine 0.8952 |
| v0-uni-reconst-scale | s=4 | Uniform linear reconstruction | Completed | 3,213 anchors; mean L2 1.6969; cosine 0.7941 |
| v0-uni-reconst-scale | s=8 | Uniform linear reconstruction | Completed | 1,609 anchors; mean L2 2.2654; cosine 0.6956 |
| v0-uni-reconst-scale | s=16 | Uniform linear reconstruction | Completed | 808 anchors; mean L2 2.7382; cosine 0.6002 |
| v0-uni-reconst-scale | s=32 | Uniform linear reconstruction | Completed | 406 anchors; mean L2 3.1250; cosine 0.5063 |
| v0-uni-reconst-scale | s=64 | Uniform linear reconstruction | Completed | 204 anchors; mean L2 3.4824; cosine 0.3971 |
| v0-uni-reconst-scale | s=128 | Uniform linear reconstruction | Completed | 103 anchors; mean L2 3.8809; cosine 0.2563 |
| v0-uni-reconst-scale | pair-mean supplement | Keep eight Uniform anchor sets; replace linear reconstruction | Completed | s=64 L2 3.5445 vs linear 3.4824; cosine 0.3498 vs 0.3971 |
| v0-vage | Static n_min=2 | Proxy-guided allocation versus Uniform | Completed | Mean L2 3.5437 vs 3.4824; cosine 0.3651 vs 0.3971 |
| v0-vage | Static n_min=1 | One initial anchor per block sensitivity | Completed | Mean L2 3.8747; cosine 0.3156 |
| v0-vage | n_min=1 method comparison | Quota, Fusion, Oracle, Residual diagnostic, Static, Uniform | Completed | Best VAGE mean L2: Quota 3.5031; Uniform 3.4824; Residual adaptive test degenerate |
| v0-vage | pair-mean supplement | Keep Static, Quota, Fusion, Oracle, Residual, Uniform anchors; replace linear reconstruction | Completed | Uniform L2 3.5445, Static n_min=2 L2 3.5880; both worse than linear |
| v0-gradient-proxy-test | Local-Risk core | Complete 32-start block proxy association | Completed | Head variance/full variance Spearman 0.9476 |
| v0-gradient-proxy-test | Local-Risk sparse 32 | Sample 4-32 proxies per 32-start block | Completed | Variance Spearman 0.867 at 4 proxies |
| v0-gradient-proxy-test | Local-Risk sparse 64 | Sample 4-64 proxies per 64-start block | Completed | Variance Spearman 0.837 at 4 proxies |
| v0-gradient-proxy-test | Local-Risk long blocks | Sparse proxy sampling in 128, 256, 576 starts | Completed | Variance Spearman 0.809 at 8/128 and 0.867 at 16/256; 576 pooled exploratory |
| v0-gradient-proxy-test | Local-Risk target spread | Check within-track true variance spread | Completed | Normalized p90/p10 7.44, 5.25, 4.42, 3.88 at 32, 64, 128, 256 starts |
| v0-gradient-proxy-test | Cross-Rank | Score fixed cross-track source pool | Completed | Head-cosine mean per-reference Spearman 0.640; top-5 overlap 0.487 |
| v0-gradient-proxy-cost | Direct cost benchmark | Compare full BP and three proxies | Completed | Median full BP 42.80 ms; head proxy 14.21 ms per window |
| v0-grad-mech-l2-norm | 1 | Validate ten BF16 caches and Gram matrices | Completed | 30 direct pair checks passed |
| v0-grad-mech-l2-norm | 2 | Compute relative L2 for lags 1-575 | Completed | 5,750 trajectory-lag rows; lag-1 mean 0.6372 |
| v0-grad-mech-l2-norm | 3 | Report and archive supplement | Completed | Report and .complete present; artifacts archived |
| v0-crb-reconst-test | 1 | Freeze ten source trajectories and manifests | Completed | Fixed source ordinals and cache identities |
| v0-crb-reconst-test | 2 | Prepare data-disk space and resumable cache paths | Completed | Storage prepared; retained caches accessible |
| v0-crb-reconst-test | 3 | Retain ten BF16 full-gradient caches | Completed | Ten complete cache files retained |
| v0-crb-reconst-test | trajectory 00 | Eight budget groups and CRB configurations | Completed | All eight q groups complete |
| v0-crb-reconst-test | trajectory 01 | Eight budget groups and CRB configurations | Completed | All eight q groups complete |
| v0-crb-reconst-test | trajectory 02 | Eight budget groups and CRB configurations | Completed | All eight q groups complete |
| v0-crb-reconst-test | trajectory 03 | Replay scheduled configurations | Terminated after partial work | q=1 complete; q=4 at 13/16 |
| v0-crb-reconst-test | trajectory 04 | Replay scheduled configurations | Terminated after partial work | q=1 complete only |
| v0-crb-reconst-test | trajectory 05 | Replay scheduled configurations | Not started; user terminated | No replay result |
| v0-crb-reconst-test | trajectory 06 | Replay scheduled configurations | Not started; user terminated | No replay result |
| v0-crb-reconst-test | trajectory 07 | Replay scheduled configurations | Not started; user terminated | No replay result |
| v0-crb-reconst-test | trajectory 08 | Replay scheduled configurations | Not started; user terminated | No replay result |
| v0-crb-reconst-test | trajectory 09 | Replay scheduled configurations | Not started; user terminated | No replay result |
| v0-crb-reconst-test | extreme-track extension | Subsequent threshold-selected trajectories | Not started; user terminated | No result |
| v0-grad-mech-test | 1 | Register and validate ChronosTiny checkpoint | Completed | Pinned checkpoint loaded; 8,394,496 parameters |
| v0-grad-mech-test | 2 | Load M5 and fix 1,000 trajectories | Completed | 30,490 decoded records; first 1,000 eligible selected |
| v0-grad-mech-test | 3 | Implement warmup and gradient probe | Completed | Probe entry and cache logic implemented |
| v0-grad-mech-test | 4 | Run warmup and full-gradient probe | Completed | 1,000/1,000 trajectories; ten BF16 caches retained |
| v0-grad-mech-test | 5 | Summarize and archive results | Completed | Results in docs/RESULTS.md; probe artifacts archived |
| rage-smoke-static-cycle | 1 | Install and verify six Conda environments | Completed | All six profiles passed checks |
| rage-smoke-static-cycle | 2 | Verify data preparation, selection, training, evaluation | Completed | Integrated cycle passed |
| rage-smoke-static-cycle | 3 | Run full regression suite | Completed | 784 passed, 3 skipped |
| rage-smoke-static-cycle | 4 | Verify one-command installer | Completed | Installer verification passed |

### v0-active-value-rank

First proposed: 2026-11-09. Last modified: 2026-11-09. Status: completed. Goal: test whether decision-aware full-gradient queries improve reference-conditioned top-k selection over output-head proxy selection at a fixed gradient-call budget. Reuse ten retained M5 trajectories, the fixed warmed Chronos-T5 Tiny checkpoint, existing 100-reference full-gradient cosine score matrices, and saved output-head gradients. Do not recompute full gradients or restart terminated CRB.

Batch 1-expanded-cross-rank uses the 100 seeded references in v0-uni-data-evaluation, ten per trajectory. For each reference, candidate starts are 0,16,32,... in each of the other nine trajectories, excluding every start reserved as one of the 100 references. Compute head cosine from saved head vectors and align with existing full cosine. Report per-reference and equal-reference-track proxy Spearman, top-k recall, true selected value, and oracle regret for top fractions 0.05, 0.10, 0.20. The thinning limits candidate count but does not remove raw-window overlap.

Batch 2-active-correction masks unqueried full values. Every policy receives the same eight proxy-quantile initial labels. Fit a ridge residual model for full-minus-head cosine with intercept, standardized head cosine, standardized squared head cosine, standardized head norm, embedding cosine, and absolute loss difference. Non-intercept feature statistics use unlabeled proxy features. Refit after each query. Compare random, highest-proxy, uncertainty-only, and boundary-uncertainty selection. Query budgets are 8,16,32,64 candidate full-gradient labels per reference. The boundary policy uses predictive uncertainty divided by distance to the predicted top-10-percent cutoff. Queried scores are exact; unqueried scores use the corrected model. Report final top fractions 0.05,0.10,0.20. Seed=0, deterministic tie breaking. Per-reference models cannot use another reference's full labels. The direct-proxy B=0 and full-gradient oracle are controls. Estimate call cost from measured medians: head 14.21 ms, full 42.80 ms per candidate; include one shared reference head and full call. Separately report replay CPU time. These are modeled call costs, not measured online speedups.

| Batch | Work | Dependency | Completion condition | Status |
|---|---|---|---|---|
| 1-expanded-cross-rank | Score expanded candidate pool | Existing head and full score caches | Aligned scores, proxy metrics, .complete | Completed |
| 2-active-correction | Replay four query policies at four budgets | Batch 1 | Ranking/cost tables, report, .complete | Completed |

### v0-directional-marginal-gain

First proposed: 2026-11-08. Last modified: 2026-11-08. Status: completed. Full protocol and result: docs/results/v0-directional-marginal-gain.md.

Goal: test whether complete-gradient or sparse-proxy direction dispersion predicts the actual one-anchor reduction in block direction reconstruction error, not merely the starting error. Reuse the ten retained trajectories, the same 46 complete nonoverlapping 256-start blocks, cached full/head Gram matrices, and prior full vector variance, relative vector variance, and full-gradient direction dispersion. Use eight evenly spaced output-head proxy positions 0,36,73,109,146,182,219,255 to calculate sparse unit-vector direction dispersion. For each n in {2,4,8,16}, place n evenly spaced full-gradient anchors including block endpoints. Add one anchor at floor midpoint of the largest consecutive anchor gap, resolving equal gaps leftmost. New positions for n={2,4,8,16} are {127,42,54,8}. Reconstruct all 256 full gradients by linear interpolation before and after insertion. At each start, direction error is 1-cos(g_t,ghat_t), with 0 for aligned directions, 1 for orthogonal directions, and 2 for opposite directions; it ignores gradient-length mismatch. E_h(n) averages this error over all 256 starts including n exact anchors. Gain is E_h(n)-E_h(n+1): positive means lower block-average direction error, negative means worse reconstruction, and the units are absolute 1-cosine error rather than percent. Also compute gain excluding the new anchor and over all positions still unqueried after insertion. Calculate pooled Pearson/Spearman and equal-track Spearman for four predictors against starting difficulty and three gain definitions, separately by n. Track 00 has only two blocks and is omitted from equal-track correlations. Save leave-one-track-out sensitivity for nonnew gain. No model calls, gradient recomputation, or pair-independent significance tests.

| Batch | Work | Dependency | Completion condition | Status |
|---|---|---|---|---|
| 1 | Validate 46 blocks and compute sparse proxy dispersion | Completed v0-gradient-variance-decomposition | 46 full/head block pairs and prior true dispersion match | Completed |
| n=2 | Replay largest-gap insertion from two anchors | Batch 1 | 46 block metrics; pre/post and nonnew gain | Completed |
| n=4 | Replay largest-gap insertion from four anchors | Batch 1 | 46 block metrics; pre/post and nonnew gain | Completed |
| n=8 | Replay largest-gap insertion from eight anchors | Batch 1 | 46 block metrics; pre/post and nonnew gain | Completed |
| n=16 | Replay largest-gap insertion from sixteen anchors | Batch 1 | 46 block metrics; pre/post and nonnew gain | Completed |
| 6 | Correlate predictors with starting error and gains; report | Four budget replays | 64 association rows, 160 sensitivity rows, docs and .complete | Completed |

Result: true direction dispersion versus starting direction error has pooled Spearman {0.616,0.701,0.547,0.395} at n={2,4,8,16}; versus actual total gain it has {-0.074,0.142,0.256,0.060}. Sparse proxy direction dispersion versus total gain has {-0.002,0.380,0.235,0.056}. At n=4 the sparse proxy has a moderate budget-specific signal, but no predictor consistently ranks marginal gain across all budgets. Thus the current evidence does not support immediate proxy-dispersion-driven anchor allocation. Full caveats and per-budget results are in the report.

### v0-gradient-variance-decomposition

First proposed: 2026-11-08. Last modified: 2026-11-08. Status: completed. Full protocol and results: docs/results/v0-gradient-variance-decomposition.md.

Goal: identify which aspects of the complete gradient field drive the observed 256-start block variance and reconstruction relations. Reuse exactly the 46 complete blocks from v0-var-proxy-estimate, the all-coordinate full-gradient Gram matrices, prior eight-point output-head proxy variances, and four-anchor linear reconstruction at local positions 0,85,170,255. Define per-block mean norm E||g||, population amplitude variance Var(||g||), full-gradient direction dispersion D_h=1-||E[g/||g||]||^2, total population vector variance V_g=E||g||^2-||Eg||^2, and relative variance V_g/E||g||^2. Record exact amplitude fraction Var(||g||)/V_g and the remaining direction-related component. Recompute average L2, absolute norm difference, cosine, and 1-cosine over all 256 starts and separately over 252 non-anchor starts. Compare V_g with mean norm, amplitude variance, D_h, and reconstruction errors; compare D_h and relative variance with direction error. Use pooled Pearson/Spearman, equal-track Spearman where defined, and residualized Pearson/rank associations after controlling mean norm and ten track fixed effects. Plot 46 color-coded blocks before and after controls. No model or gradient recomputation and no pair-independent significance tests.

| Batch | Work | Dependency | Completion condition | Status |
|---|---|---|---|---|
| 1 | Validate fixed block identity and cached Gram inputs | Completed v0-var-proxy-estimate | 46 blocks and prior variance/reconstruction values match | Completed |
| 2 | Compute block variance decomposition and both reconstruction subsets | Batch 1 | 46 complete block rows, all derived values finite | Completed |
| 3 | Calculate requested raw and controlled associations | Batch 2 | Pooled, equal-track, and partial tables for all/non-anchor subsets | Completed |
| 4 | Plot by track and check sensitivity | Batch 3 | Raw and adjusted rank scatter panels, leave-one-track-out ranges | Completed |
| 5 | Record results | Batch 4 | Report, RESULTS, ORCHE, LOG, and .complete agree | Completed |

Result: median amplitude fraction of total vector variance is 0.176, and only one of 46 blocks exceeds 0.5. Pooled Spearman V_g versus mean norm is 0.970, versus amplitude variance 0.569, and versus D_h -0.222. D_h versus direction error is 0.701 pooled and 0.713 after mean-norm/track rank controls. The prior positive V_g versus cosine pooled Spearman 0.324 changes to controlled rank association -0.431; proxy variance versus cosine changes from 0.398 to -0.202. Excluding four exact anchors changes metric levels but not correlations because each block has the same four zero-error anchor positions. All results are descriptive for the fixed ten-track sample.

### v0-grad-traj-level-norm-gap

First proposed: 2026-11-08. Last modified: 2026-11-08. Status: completed. Full protocol and results: docs/results/v0-grad-traj-level-norm-gap.md.

Goal: determine whether the ten retained trajectories differ in the norm of their mean full-gradient vector and mean output-head proxy-gradient vector. Reuse all legal windows and saved all-coordinate full/head Gram matrices, without model calls. For each track and kind calculate the primary `||mean_i g_i||=sqrt(mean(G))` and secondary `mean_i ||g_i||=mean(sqrt(diag(G)))`, plus RMS norm, population vector variance, and the primary-to-secondary ratio. Compare per-kind cross-track minimum, maximum, max/min, median, mean, and CV. Correlate full versus head metrics across ten paired tracks using Pearson and Spearman. Raw full/head magnitudes are in different parameter spaces, so compare relative spread and ordering rather than treating absolute magnitudes as equal-scale measurements.

| Batch | Work | Dependency | Completion condition | Status |
|---|---|---|---|---|
| 1 | Validate ten full/head Gram caches, manifests, and markers | Completed gradient-proxy extraction | Shapes, finite entries, positive norms, and symmetry pass | Completed |
| 2 | Calculate primary and secondary norms for both gradient kinds | Batch 1 | Twenty track-kind rows and spread summary | Completed |
| 3 | Compare track ordering and report | Batch 2 | Paired correlations, report, and output .complete | Completed |

Result: primary full norm of mean ranges 0.7780-1.5050 (1.9345x, CV 0.2291); output-head ranges 0.0607-0.1515 (2.4960x, CV 0.3040). Their ten-track Spearman is 0.8788. Secondary mean-of-window-norm spreads are narrower at 1.4057x and 1.4433x, respectively. These are descriptive selected-track differences, not population-level statistical significance.

### v0-var-proxy-estimate

First proposed: 2026-11-08. Last modified: 2026-11-08. Status: completed. Full protocol and result: docs/results/v0-var-proxy-estimate.md.

Goal: determine whether eight output-head gradient proxies in each 256-start block estimate a variance associated with four-anchor linear-interpolation reconstruction error of the true full-gradient sequence. Reuse the fixed ten continuous retained trajectories and their complete full/head gradient Gram caches. Partition legal window starts into consecutive nonoverlapping 256-start blocks; exclude terminal fragments. Proxy positions are 0,36,73,109,146,182,219,255; full-gradient anchor positions are 0,85,170,255. Compute population vector variance from eight proxy gradients. Reconstruct all 256 full gradients by linear interpolation of four anchors and average the L2 difference, directional cosine, and absolute difference in norms over all positions. Record 1-cosine as a direction-error view. Correlate block values with Pearson and Spearman across all blocks and within each track. Do not recompute gradients or infer runtime savings from cache replay.

| Batch | Work | Dependency | Completion condition | Status |
|---|---|---|---|---|
| 1 | Validate fixed ten cache pairs and block positions | Completed proxy extraction | All ten cache markers, Gram shapes, symmetry, and finite entries pass | Completed |
| 2 | Compute proxy variance and reconstruction measures | Batch 1 | One row per complete 256-start block; terminal fragments counted | Completed: 46 rows |
| 3 | Compute correlations and report findings | Batch 2 | Per-track and pooled Pearson/Spearman; report and .complete | Completed |
| 4 | Compare complete full-gradient variance with the same metrics | Batch 3; full Gram caches | True variance for 46 blocks, per-track and pooled correlations, supplement .complete | Completed |
| 5 | Use the mean of four full anchors as a constant block representative | Batch 4; same blocks and anchors | Three metrics, both variance correlations, summary and .complete | Completed |

Result: pooled Spearman with mean L2 is 0.8538, with mean directional cosine 0.3977, and with mean absolute norm difference 0.4970. In the complete-variance supplement, corresponding pooled Spearman values are 0.9463, 0.3240, and 0.6381; full variance correlates with eight-proxy variance at 0.9169. The positive cosine association does not support using variance to predict worse directional reconstruction. Equal-track mean correlations exclude track 00, which has two blocks. A follow-up four-anchor constant-mean representative gives true-variance pooled Spearman 0.9873/0.2670/0.8078 for L2/cosine/absolute norm difference, and proxy-variance Spearman 0.9126/0.3374/0.6444. All four anchor starts are scored against the mean. See the result report for interpretation and limits.

### v0-uni-data-evaluation

First proposed: 2026-11-08. Last modified: 2026-11-08. Status: completed. Full report: docs/results/v0-uni-data-evaluation.md.

Goal: test whether full-gradient similarity scores against fixed reference windows vary smoothly along each source trajectory. Reuse the same ten retained BF16 full-gradient caches and fixed model; do not recompute gradients. Draw ten legal starts per trajectory with seed 0, rejecting adjacent starts within a trajectory, for exactly 100 reference windows. Exclude every selected reference start from query scoring and from either endpoint of every within-trajectory lag pair. Compute each remaining start against each of the 100 references using the complete-gradient dot product and cosine. Save the full per-track score arrays. For lags 1 through 575, compare score pairs from the same trajectory and same reference, separated by exactly that many legal starts. Record absolute difference, relative difference |a-b|/((|a|+|b|)/2), and sign-flip rate for both scores. The absolute-magnitude denominator is required because cosine and dot product can be negative; preserve zero-denominator counts. Average per-track statistics equally across ten tracks. Generate lag curves. Query and reference gradients can share raw observations; only exact reference starts are excluded.

| Batch | Work | Dependency | Completion condition | Status |
|---|---|---|---|---|
| 1 | Validate caches and select 100 nonadjacent references | Ten full BF16 caches | Ten references per track, no adjacent starts | Completed |
| 2 | Compute full-gradient dot and cosine score matrices | Batch 1; RTX 4080 | All non-reference query-reference pairs saved | Completed: 12,736 queries by 100 refs |
| 3 | Compute lag 1-575 differences and plot | Batch 2 | Per-track and equal-track tables, plot, .complete | Completed: 11,500 rows |
| 4 | Record results | Batch 3 | EXP, RESULTS, ORCHE, LOG, report agree | Completed |

Config: configs/rage/v0-uni-data-evaluation.json. Entry: scripts/v0_uni_data_evaluation.py. Output: outputs/train/v0-uni-data-evaluation/. Logs: logs/v0-uni-data-evaluation/.

Result: at lag 1, mean relative score difference is 0.4856 for cosine and 0.5372 for dot product; sign-flip fraction is 9.40%. Cosine absolute difference is 0.0402. Across 1,273,600 valid query-reference scores, pooled signed mean/population variance are 0.0284/0.0283 for cosine and 0.6101/9.1331 for dot product. Pooled mean absolute values are 0.1378 and 1.8782, respectively. Pooled signed medians are 0.0284 for cosine and 0.1635 for dot product; pooled absolute-value medians are 0.1207 and 1.0334. The small cosine signed mean partly reflects sign cancellation, while the small mean absolute cosine confirms typically low magnitude. Both difference measures rise with lag. These scores are locally associated but not approximately constant under the chosen relative metric. All summaries are equal-trajectory means and are limited to the fixed ten-track sample.

### v0-uni-reconst-scale

First proposed: 2026-11-08. Last modified: 2026-11-08. Status: completed. Full protocol and results: docs/results/v0-uni-reconst-scale.md.

Goal: measure Uniform linear-interpolation reconstruction quality as the permitted average spacing varies. Reuse the fixed ten retained trajectories and their full-gradient Gram matrices without recomputing model gradients. For each trajectory with N=T-575 legal starts, run s in {1,2,4,8,16,32,64,128}, K=ceil(N/s), and K unique whole-trajectory evenly spaced anchors including both endpoints. Reconstruct all legal starts from true anchor gradients using linear interpolation. Report mean L2 and mean cosine per trajectory, then equal-trajectory and window-weighted aggregates. Include anchors in the metric average; exact anchor errors are zero and cosine one. No proxy calls or tunable parameters.

| Batch | Work | Dependency | Completion condition | Status |
|---|---|---|---|---|
| 1 | Validate ten gradient and Gram caches | Existing completed caches | All markers, sizes, shapes, and finite entries pass | Completed |
| 2 | Replay eight Uniform budgets on ten trajectories | Batch 1 | 80 metric rows, exact anchor counts, .complete | Completed |
| 3 | Record scale results | Batch 2 | Result report, RESULTS, ORCHE, and LOG agree | Completed |

Equal-trajectory mean cosine is 1.0000, 0.8952, 0.7941, 0.6956, 0.6002, 0.5063, 0.3971, and 0.2563 in ascending s order. Mean L2 is 0.0000, 0.9650, 1.6969, 2.2654, 2.7382, 3.1250, 3.4824, and 3.8809. The s=64 row reproduces the prior Uniform baseline. A completed reconstruction-only supplement keeps all eight Uniform anchor sets but replaces each unobserved gradient in an anchor gap by the mean of its two endpoint gradients. At s=64, mean L2 changes from 3.4824 to 3.5445 and mean cosine from 0.3971 to 0.3498. Results and exact rules: docs/results/v0-uni-reconst-scale.md.

### v0-vage

First proposed: 2026-11-08. Last modified: 2026-11-08. Status: VAGE-Static completed. Full protocol and result: docs/results/v0-vage.md.

Goal: compare static proxy-guided block allocation with whole-trajectory Uniform for linear-interpolation reconstruction of every legal window gradient. Reuse the fixed ten BF16 full-gradient caches and existing full/head Gram matrices; perform no new model gradient calls. Per trajectory set N=T-575, Bg=ceil(N/64), Bp=ceil(N/64), and nonoverlapping block length 512 with the terminal block retained. Allocate proxy points approximately by block length, with at least two per block. Compute unbiased output-head vector trace variance and shrink blocks with fewer than eight proxy points toward the length-weighted trajectory-wide proxy prior using weight min(1,(p-1)/7). Keep proxy estimates static. Set n_min=2 full-gradient endpoints per block. Allocate remaining anchors by descending L_h V_h/[n_h(n_h+1)], respecting block capacity and lower-index tie-breaking. Insert at the midpoint of the largest uncovered within-block gap, with leftmost tie-breaking. Uniform uses Bg globally even anchors. Evaluate mean L2 and cosine, relative RMSE, tails, and non-anchor means. No per-track tuning is permitted.

| Batch | Work | Dependency | Completion condition | Status |
|---|---|---|---|---|
| 1 | Validate caches and compute static proxy scores and anchors | Ten full/head Gram and BF16 caches | Exact Bg/Bp counts for all ten tracks | Completed: 204 calls each in replay accounting |
| 2 | Evaluate VAGE-Static and matched Uniform reconstruction | Batch 1 | Ten paired metric rows and complete marker | Completed |
| 3 | Record results and limitations | Batch 2 | EXP, RESULTS, ORCHE, LOG, result report agree | Completed |

Additional n_min=1 comparison: completed Uniform, Static, Quota, Fusion, Residual diagnostic, and Oracle on the same ten caches. Quota uses L_h sqrt(V_h+1e-12) integer allocation and even within-block anchors. Fusion uses kappa=8 and leave-one-track-out median true-to-proxy block variance scale, updating with selected full-gradient anchor variance. Oracle uses all cached true gradients to calculate population block variance and is nondeployable. Residual restricts candidate anchors to the Bp measured proxy positions to avoid unbudgeted proxy calls. Since Bg=Bp, it necessarily ends with all proxy points as anchors and is not a meaningful adaptive-selection test. Equal-track mean L2: Uniform 3.4824, Quota 3.5031, Residual diagnostic 3.5159, Static 3.8747, Fusion 3.8762, Oracle 3.8774. No variant beats Uniform on mean L2 or mean cosine. Details: docs/results/v0-vage.md.

n_min=1 sensitivity: completed separately with the same Bg/Bp, proxy scores, and Uniform baseline. Initial block anchor is its center; the next selected point is the farther endpoint, then internal largest gaps are split. Starts outside the final anchor range use nearest-anchor extrapolation. Equal-track mean L2 is 3.8747 and mean cosine 0.3156, both worse than n_min=2 and Uniform. All ten tracks lack the first global start among final anchors; four lack the last. This comparison does not isolate minimum count from endpoint coverage. Details: docs/results/v0-vage.md.

Primary result: VAGE-Static equal-track mean L2 3.5437 versus Uniform 3.4824 and mean cosine 0.3651 versus Uniform 0.3971. VAGE wins mean L2 on two tracks and mean cosine on one. The static design is worse on the primary reconstruction metrics. Uniform also avoids all 204 proxy calls. This is retrospective replay, not a direct runtime benchmark. A completed reconstruction-only supplement freezes all VAGE and matched Uniform anchor sets and uses a constant average of adjacent anchor gradients for each gap, keeping observed anchors exact. At n_min=2, Static L2/cosine change from 3.5437/0.3651 to 3.5880/0.3205; matched Uniform changes from 3.4824/0.3971 to 3.5445/0.3498. The n_min=1 methods are also reported in docs/results/v0-vage.md.

## 当前实验

### v0-gradient-proxy-test

This batch has two parts with separate questions and result summaries. Part codes are Local-Risk and Cross-Rank. The original proxy-exploration stage is Local-Risk; the appended cross-trajectory experiment is Cross-Rank. The fixed model, checkpoint, ten selected trajectories, and saved gradient caches are shared.

#### Part Local-Risk: local dispersion indication (completed)
Question: Within nonoverlapping blocks of 32 consecutive window starts on the ten retained tracks, do inexpensive proxy statistics preserve the ranking of full-gradient vector variance and full-gradient direction dispersion?

Result: Other (positive descriptive association, not a binary selection-policy result). Output-head dispersion has equal-trajectory mean Spearman 0.9476 with full vector variance and 0.8161 with full direction dispersion. The study does not test whether proxy-selected blocks capture high-variance intervals under a fixed full-gradient budget. Full results: docs/results/v0-gradient-proxy-test.md.

#### Local-Risk sampling extension (completed)
Question: If only k output-head gradients are computed in each 32-start block, with k in {32,24,16,8,4}, how strongly do the sampled-head variance and direction-dispersion estimates correlate with the full-gradient variance and direction dispersion computed from all 32 points?

Use the existing complete 32-start blocks and saved head/full Gram matrices. For each k below 32, perform 20 seeded uniform samples without replacement per block. A permutation is nested within each replicate, so the k=4 set is contained in k=8, k=16, and k=24. The k=32 result uses all points once as the exact baseline. Across blocks within each trajectory, compute Pearson and Spearman correlations against both full-gradient targets; average trajectory correlations equally. Report the median and 10th-90th percentile across sampling replicates. This is a sampling-efficiency extension of Local-Risk, not a third top-level part.

Status: completed on 2026-11-08. Result: Other (variance ranking remains strong down to 4/32 sampled points; direction-dispersion ranking weakens more, especially at 4/32). The analysis is retrospective subsampling of saved exact head Gram matrices, so it measures information loss rather than runtime savings. Config: configs/rage/v0-gradient-proxy-local-risk-sparse.json. Entry: scripts/analyze_sparse_proxy_risk.py. Output: outputs/train/v0-gradient-proxy-test/local_risk_sparse/.

#### Local-Risk 64-point sampling extension (completed)
Question: In each consecutive 64-start block, if only k output-head gradients are computed for k in {64,48,32,16,8,4}, how strongly do sampled proxy variance and direction dispersion correlate with the true full-gradient variance and direction dispersion across all 64 points?

Use the saved full-coordinate and output-head Gram matrices to compute full 64-point targets and sampled proxy statistics. For k below 64, use 20 fixed-seed uniform samples without replacement per block; within each replicate the sample sets are nested. The k=64 result uses all points once. Report per-track Pearson and Spearman correlations, equal-track averages, and 10th-90th sampling-replicate ranges. This is retrospective subsampling and does not measure runtime savings or interval-selection performance.

Status: completed on 2026-11-08. Result: Other (variance ranking remains strong at 8 and 4 of 64 points; direction ranking is moderate at 8 and weaker at 4). This is retrospective Gram-matrix subsampling, not a runtime or selection-policy test. Config: configs/rage/v0-gradient-proxy-local-risk-64.json. Entry: scripts/analyze_sparse_proxy_risk_64.py. Output: outputs/train/v0-gradient-proxy-test/local_risk_sparse_64/.

#### Local-Risk long-block sampling extension (completed)
Question: In 128-, 256-, and 576-start blocks, how does the correlation between sparse output-head variance/direction dispersion and full-gradient variance/direction dispersion change as the proxy count is reduced?

Keep the same proxy sampling fractions as the 64-start experiment: 100%, 75%, 50%, 25%, 12.5%, and 6.25%. This gives counts {128,96,64,32,16,8}, {256,192,128,64,32,16}, and {576,432,288,144,72,36}. Use 20 fixed-seed nested uniform samples without replacement for each count below the full block size. Use saved full/head Gram matrices; do not recompute gradients.

For 128 starts, compute equal-track correlations across complete blocks. For 256 starts, only tracks with at least three complete blocks can contribute per-track correlations. For 576 starts, there are only 19 complete blocks total and at most two per track, so report only pooled-block correlations as exploratory and do not treat them as an equal-track or inferential result. All blocks are disjoint by window-start index; their underlying 576-point windows can still overlap in raw observations.

Status: completed on 2026-11-08. Result: Other (128- and 256-start blocks retain useful ranking under sparse sampling; 576-start results are pooled exploratory only because there are just 19 blocks). Config: configs/rage/v0-gradient-proxy-local-risk-long.json. Entry: scripts/analyze_sparse_proxy_risk_long.py. Output: outputs/train/v0-gradient-proxy-test/local_risk_sparse_long/.

#### Local-Risk target variance spread (completed)
Question: Do the full-gradient variance targets vary enough across blocks to support screening, or are they concentrated near a common value?

Result: Other (substantial within-track spread for 32-256 start blocks). After normalizing each block by its track median, p90/p10 is 7.44, 5.25, 4.42, and 3.88 for 32, 64, 128, and 256 starts. The 576-start ratio is 2.06 over only 19 blocks and is exploratory. Results: docs/results/v0-gradient-proxy-test.md.

#### Part Cross-Rank: cross-track nonoverlap ranking (new appended experiment)
Question: For each full-gradient reference window from the two shortest retained tracks, do proxy scores rank a fixed pool of pairwise nonoverlapping source windows from the other eight tracks in the same order as full-gradient cosine similarity? Also measure absolute cosine calibration for vector proxies.

Reference tracks: track 00 (607 windows) and track 07 (1,270 windows), the two shortest by window count. Use every window in each reference track. Source tracks: 01-06, 08, and 09. Select three deterministic, evenly placed source windows per source track; their 576-point spans must not overlap. This gives 24 source candidates and 45,048 reference-source pairs. Full-gradient cosine uses all 8,394,496 coordinates from the retained BF16 copies of the FP32 gradients. Primary proxy: output-head-gradient cosine. Secondary proxies: decoder token-mean embedding cosine and negative absolute CE-loss difference. Per-reference outcomes: Spearman, Kendall tau-b, and top-5 overlap against full-gradient ranking. Also report head and embedding cosine MAE, RMSE, and bias.

This protocol can answer ranking over the fixed source pool only. It does not estimate unrestricted M5 performance: only eight source tracks and 24 disjoint candidates are available in the ten-track cache. Track and window dependence must be stated. Result: Other (qualified positive ranking signal for output-head cosine on the fixed candidate pool, with only moderate average per-reference rank agreement; not enough to claim direct replacement of full gradients). Pooled pair Pearson/Spearman are 0.786/0.795; mean per-reference Spearman/Kendall are 0.640/0.481; mean top-5 overlap is 0.487 versus 0.208 expected for a random five-of-24 list. Head cosine MAE is 0.071 and bias is -0.016. Embedding cosine ranks poorly and loss difference is weak. This is a fixed ten-track-tail sample and not unrestricted M5 evidence. Full results: docs/results/v0-gradient-proxy-test.md. Protocol and entry: docs/gradient-proxy-protocol.md and scripts/cross_rank_gradient_proxies.py.

#### Cost benchmark amendment (2026-11-08)

Completed 2026-11-08. Results: docs/results/v0-gradient-proxy-cost.md. Measure wall-clock computation cost directly on the same fixed checkpoint and selected ten trajectories, using 8 evenly spaced windows per trajectory and 3 timed repeats per window. Batch size is 1 for every method. Compare full BP, output-head-only gradient, final decoder token-mean embedding, and CE loss. Include tokenizer/window preparation, model forward, method-specific backward/proxy operation, copying the requested result to CPU, and CUDA synchronization. Warm each method first; randomize method order per window. Validate sampled proxy outputs against the completed feature caches before timing. Save per-call timings, per-track summaries, medians, means, 90th percentiles, speedups and descriptive call-bootstrap intervals. Protocol and implementation: docs/gradient-proxy-protocol.md and scripts/benchmark_gradient_proxy_cost.py. Output: outputs/train/v0-gradient-proxy-cost/. This measures compute cost under this implementation and device; it does not claim a platform-independent FLOP ratio.


First proposed: 2026-11-08. Last modified: 2026-11-08.
Status: base Local-Risk and Cross-Rank completed on 2026-11-08; the 32-point and 64-point Local-Risk sampling extensions are complete; 128-, 256-, and 576-start sampling is complete. Local-Risk and Cross-Rank answers are summarized separately in docs/results/v0-gradient-proxy-test.md. Results: docs/results/v0-gradient-proxy-test.md. Protocol: docs/gradient-proxy-protocol.md. The source proxy extraction, alignment checks, lag analysis, and block-32 dispersion are complete for all ten tracks. Cross-Rank reuses retained full-gradient and proxy caches and computes cross-track scores without new model backward passes.


## 历史记录

### v0-grad-mech-l2-norm

First proposed: 2026-11-08. Last modified: 2026-11-08.
Status: completed and result artifacts archived on 2026-11-08; results in docs/results/v0-grad-mech-l2-norm.md. This is a supplement to the completed v0-grad-mech-test, restricted to the ten retained trajectories. It does not reopen the original 1,000-trajectory run. The ten BF16 source-gradient caches remain unarchived in outputs/train/v0-crb-reconst-test/gradients/.

Goal: measure the full-vector change between overlapping window gradients, including direction and magnitude in one scalar. For each within-trajectory pair at lag k, define a = ||g_i||_2, b = ||g_(i+k)||_2, D = ||g_i - g_(i+k)||_2, and R = D / ((a+b)/2). The primary outcome is the lag-1 distribution of R; secondary outcomes are the lag-conditioned distributions for every integer lag 1 through 575. No pair crosses trajectory boundaries. Window span is 576 (512 context plus 64 prediction), stride 1, and overlap fraction is (576-k)/576.

Inputs: use only the already retained BF16 full-gradient caches for source manifest ordinals 973, 980, 981, 984, 985, 987, 988, 996, 998, and 999 (12,836 windows total). Their coordinates are BF16 copies of FP32 gradients from the fixed warmed ChronosTiny checkpoint. Reuse the full-coordinate Gram matrices already derived from those caches by v0-gradient-proxy-test; these matrices used FP32 tiled GEMM with FP64 accumulation across coordinate tiles. Do not recalculate model gradients, sample new trajectories, change the checkpoint, or create another full-gradient copy. Verify the source manifest, cache completion markers, file sizes, Gram shape, finite entries, symmetry, and three direct BF16-cache pair distances per trajectory. Compute D from the Gram diagonal and pair dot product, clipping only small negative squared distances attributable to numerical roundoff and recording the clip count. Reject any substantial negative squared distance, zero denominator, nonfinite R, or R above the triangle-inequality limit beyond numerical tolerance.

Outputs: configs/rage/v0-grad-mech-l2-norm.json fixes scope and paths. Entry: scripts/v0_grad_mech_l2_norm.py in the rage Conda environment, with two CPU threads and no model or GPU allocation. Write outputs/train/v0-grad-mech-l2-norm/trajectory_lag.csv (count, mean, RMS, median, p90, p95, maximum for each trajectory and lag), lag_summary.csv (pair-weighted pooled summaries and equal-trajectory mean at each lag), lag1_pairs.csv (all lag-1 pair values), audit.json, summary.json, and .complete. Write logs under logs/v0-grad-mech-l2-norm/. Report results in docs/results/v0-grad-mech-l2-norm.md and link from docs/RESULTS.md. Pair values overlap heavily and are dependent; do not treat pair counts as independent replicates. This is an empirical relative gradient-distance analysis on the selected ten late-manifest trajectories, not a global Lipschitz proof or a replacement for the 1,000-trajectory direction/norm results.

| Batch | Work | Dependency | Completion condition | Status |
|---|---|---|---|---|
| 1 | Validate ten caches, source manifests, Gram matrices, and direct pair checks | Completed grad-mech and gradient-proxy artifacts | All ten sources pass the audit | Completed: 30 direct pair checks passed |
| 2 | Compute R for all within-trajectory pairs at lags 1-575 and aggregate | Batch 1 | Ten trajectory-lag tables and pooled lag summary have the expected pair counts and finite values | Completed: 5750 trajectory-lag rows and 575 pooled rows |
| 3 | Record results and limitations | Batch 2 | Result report, RESULTS index, LOG, ORCHE, and .complete agree with the generated data | Completed |

Conclusion: In the ten retained trajectories, the practical hypothesis of small relative full-gradient changes at lag 1 is contradicted. The 12,826 lag-1 pairs have mean R=0.6372, median R=0.7620, and 55.01% have R>=0.6, despite 575/576 raw window-span overlap. For these same pairs, mean cosine is 0.7215; a cosine around 0.7 does not establish small full-vector L2 change. This is a restricted empirical conclusion, not a proof of mathematical non-Lipschitzness or a claim about all 1,000 original trajectories. Result artifacts are archived at archive/v0-grad-mech-test/outputs/train/v0-grad-mech-l2-norm/, with the original outputs/train/v0-grad-mech-l2-norm/ path retained as a symbolic link. The ten BF16 gradient caches, code, config, and logs remain at their existing project paths.

### v0-crb-reconst-test（terminated by user）

#### 2026-11-07 gradient reuse amendment

This amendment supersedes the earlier instruction to use the first ten trajectories. The user authorized a random subset because trajectory identity is not scientifically constrained. At probe ordinal 972, select ten not-yet-processed manifest ordinals uniformly without replacement using seed 0: 973, 980, 981, 984, 985, 987, 988, 996, 998, and 999. Their series IDs are recorded in the CRB input manifest.

During the existing v0-grad-mech-test pass, retain BF16 copies of the full gradients for these ten trajectories at outputs/train/v0-crb-reconst-test/gradients/trajectory-00.bf16 through trajectory-09.bf16. The source probe computes FP32 gradients and converts the retained vectors to BF16 for storage. The CRB stage must reuse these files and must not recompute their gradients. Completion markers and the CRB input manifest identify each source trajectory and cache ordinal. The formal queue processes one selected trajectory per batch under outputs/train/v0-crb-reconst-test/batches/trajectory-NN/. It starts each batch as soon as that trajectory's cache marker exists, uses CPU-only replay, and does not load a GPU model. Each batch runs one initial search round over all eight budget groups, with the uniform baseline and 16 CRB configurations per group.

The selected ten contain 12,836 legal windows and require about 215.5 GB (200.7 GiB). Current free space was about 502 GiB, leaving about 300 GiB after the cache. A seeded sample of twenty remaining trajectories was estimated at 407.5 GB, leaving about 122 GiB; that would consume most available space, so retain ten. This is a recorded storage-driven scope change. Update the result summary to disclose random selection from the not-yet-processed tail of the fixed 1,000-trajectory manifest. The ten per-trajectory queue batches preserve the selected order and each batch writes its own input manifest, progress, metrics, and completion marker.

- 首次提出：2026-11-07
- 最后更新：2026-11-08
- 状态：用户于 2026-11-08 终止；结果已大概明确且受资源限制，复用 gradient reuse amendment 中固定的 10 个随机 manifest ordinals 和探测阶段提前保存的 BF16 全梯度。每条轨迹独立完成 8 个预算组及每组 16 个 CRB 比例配置。
- 目标：检验 CRB 自适应重建能否恢复每个合法窗口起点的梯度，以及在相同实际探测预算下是否优于均匀锚点。
- 数据与梯度：读取 v0-grad-mech-test 的 input_manifest.json，并按 configs/rage/v0-crb-reconst-test.json 中固定的 10 个 manifest ordinals 选择轨迹；使用其预热后的固定检查点、Chronos 损失及预处理。窗口参数为上下文长度 512、预测长度 64、步长 1。复用探测阶段落盘的 BF16 全梯度，不在 CRB 阶段重复计算这些梯度。完整向量由 FP32 探测梯度转换为 BF16 后持久化，保留所有 8,394,496 个参数坐标及轨迹/窗口/检查点元数据，不生成 FP32 全量梯度副本。每条轨迹写完并落盘后记录完成标记；中断时从最后保存的窗口续算。
- 总预算组：预算比 q 定义为合法起点总数 N / 允许计算梯度的点数。标准预算区间固定为 q∈{1,4,16,64,128,256,512,576}。每条含 N 个合法起点的轨迹允许 K=ceil(N/q) 次梯度计算；首尾合法起点必须计算，故当 K<2 时提升为 2。q=1 表示全部合法起点均计算。q 不表示随机跳过概率，也不要求每隔恰好 q 个起点探测一次；它定义每条轨迹的总计算次数，CRB 按内部调度决定探测位置。
- 均匀基准：对每个 q 组必须先运行一次 baseline-uniform。按 K 个唯一锚点在该轨迹全部合法起点范围内均匀布置，并强制包含首尾；每点保存 L2 误差和余弦相似度。CRB 若实际计算数低于 K，则额外运行相同实际计算数的均匀锚点基准，确保比较预算一致。
- CRB 比例与预算分配：三类 CRB 预算依次为覆盖 cover_density、风险驱动误差 error_density、分裂/抵消检测 split_density。覆盖预算至少包括两个端点；每个总预算 K 先预留这两个必需端点锚点，再将 K-2 个剩余预算按比例分配给三类预算，使用最大余数法转成整数预算，并记录实际预算。比例为 cover:error:split，三项非负且和为 1。对每个 q∈{1,4,16,64,128,256,512,576}，每条轨迹均运行 1 个 baseline-uniform 和 16 个 CRB 配置。16 个比例固定为分母 4 的三维单纯形格点（15 个组合，含 1:0:0）及中心点 (1/3,1/3,1/3)；所有参数都在 [0,1] 且和为 1。
- 参数探索算法：使用确定性均匀单纯形设计，不做随机采样或 Pareto 自适应追加轮次。分母 4 的 15 个格点按 cover、error、split 的整数权重枚举，再加入中心点 (1/3,1/3,1/3)，保证 16 个唯一比例配置。每个候选按每条轨迹的实际整数预算三元组去重；若配置重合则复用既有重放和指标，但仍记录参数尝试。每个新候选都在该批次对应轨迹的全部合法起点上精确评估，不用抽样窗口代替最终指标。q=1 的所有起点都已探测，采用精确恒等结果（L2=0、余弦=1），不复制全量梯度到 CRB 内存结构。
- CPU 重放与逐点输出：全梯度阶段完成后释放 GPU 模型，使用 CPU 和 BF16 只读内存映射逐轨迹读取数据。CRB 回调只从缓存按合法起点取向量，不再运行模型；每个非锚点估计仅使用其最近的左右两个相邻锚点完整梯度做线性插值，不拟合或混入其他锚点。每个 CRB 参数配置以及对应的 baseline-uniform 都为每个窗口起点保存两项标量：L2 向量误差 ||g_est-g_true||_2（越低越好）和余弦相似度（越高越好）。另外输出逐轨迹汇总、全轨迹差值、配置比例、请求与实际探测数、候选去重/复用信息及 Pareto 前沿。不中止或清理全量 BF16 梯度文件。
- 执行范围与停止：每条轨迹只执行一轮，依次完成 8 个预算组；每组包含 1 个均匀基准及 16 个确定性单纯形比例配置。共 10 个单轨迹批次，输出位于 outputs/train/v0-crb-reconst-test/batches/trajectory-NN/。最多并行 2 个批次，每批限制为 2 个 CPU 线程；异常退出由队列检查并续跑。每个候选结果先原子写入后追加 evaluations.jsonl，每轮和每个候选均更新进度及心跳。磁盘剩余空间达到 1 GiB 安全余量时停止并收尾，保留所有梯度和已完成指标。
- 存储核算：固定抽样的 10 条轨迹共 12,836 个合法窗口，参数维度为 8,394,496。BF16 每参数 2 字节，完整矩阵约 215,500,000,000 字节，即约 215.5 GB（200.7 GiB），另有少量索引、逐点指标和日志。检查时数据盘可用 538.61 GB（501.6 GiB），缓存写入后名义余量约 322.6 GB（300.5 GiB）；估计会占用当时可用空间的 40.0%。按轨迹写入并用内存映射分块读取；全量梯度不驻留于 RAM，不创建完整 FP32 副本。
- 后续极端轨迹扩展：主分析完成后，另行按预先登记的梯度范数变化和相邻方向余弦阈值筛选剧烈波动轨迹；筛选规则不得参考 CRB/均匀方法的表现。沿用同一两项逐点误差、同预算对照和差距报告。
- 完成标准：10 条固定轨迹的全部 BF16 梯度可续读且未被删除；每个 q∈{1,4,16,64,128,256,512,576} 组都有 baseline-uniform 及包含 1:0:0 的 16 个参数尝试记录；每个候选在全部合法起点产生两项逐点指标，重复预算配置明确标记复用；停止原因、磁盘余量及完整输出均已记录后再写入 .complete。

| 批次 | 工作内容 | 依赖条件 | 状态 |
|---|---|---|---|
| 1 | 等待 v0-grad-mech-test 成功完成，冻结 amendment 指定的 10 条轨迹、检查点和窗口清单 | v0-grad-mech-test 的 .complete 标记 | 已完成；10 条轨迹和缓存索引固定 |
| 2 | 复核数据盘余量，准备 BF16 分轨迹梯度文件及断点记录 | 批次 1；约 215.5 GB 梯度缓存空间 | 已完成；缓存复用，余量约 301 GiB |
| 3 | 复用探测阶段保存的 10 条轨迹完整 BF16 梯度 | 批次 2 | 已完成；10 条缓存均保留 |
| 4 | 对每条轨迹完成 q∈{1,4,16,64,128,256,512,576} 的均匀基准和 16 个单纯形比例配置 | 批次 3 | 用户终止；轨迹 00-02 完成，03-04 部分完成，05-09 未开始 |
| 5 | 对每条轨迹各完成一次 8 预算组 × (1 个基准 + 16 个比例配置)，最多 2 批并发、每批 2 线程 | 批次 4 | 用户终止；已提交结果及状态见 docs/results/v0-crb-reconst-test.md |
| 6 | 按预先固定的极端波动阈值开展后续轨迹扩展 | 主搜索停止并完成结果记录 | 未开始；无后续任务排队 |

#### Termination and retained artifacts (2026-11-08)

The user terminated further replay because the available results were already broadly clear and container resources were constrained. At termination, trajectories 00-02 completed all eight q groups; trajectory 03 completed q=1 and 13/16 configurations at q=4; trajectory 04 completed q=1; trajectories 05-09 had not started. The replay supervisor tmux session was stopped after the workers had exited. This is a partial experiment, not a completed ten-trajectory run. Raw batch outputs and the final regenerated visualization are archived at archive/v0-crb-reconst-test/. The ten BF16 full-gradient trajectory caches under outputs/train/v0-crb-reconst-test/gradients/ are intentionally retained unchanged.



### v0-grad-mech-test (completed)

- Status: Completed on 2026-11-07; 1,000/1,000 trajectories. Results: docs/RESULTS.md. Probe results and visualization are archived under archive/v0-grad-mech-test/; original result paths remain available through symbolic links.

- 首次提出：2026-11-07
- 最后更新：2026-11-07
- 目标：检验同一条连续时间序列轨迹中，相邻且高度重叠的窗口是否具有高度相似的全参数梯度，并评估窗口重叠程度与梯度方向、幅值之间的关联。
- 模型：ChronosTiny 预训练检查点 amazon/chronos-t5-tiny，固定版本 29d808298f1a62493e7b9a5e08529d0d930fa189，架构 T5ForConditionalGeneration，参数量 8,394,496。已在 configs/pretrained_models/chronos.yaml 中登记为 chronos-t5-tiny-8m；检查点位于 cache/pretrained_models/ 下。
- 数据配置：configs/rage/v0-grad-mech-test.json；使用 autogluon/chronos_datasets 中现有的 Chronos 数据集配置 m5，源版本固定为 eeecad0b82a8c237e212ce6f8d1abecb513e2cec。上游元数据显示共有 30,490 条源记录，Parquet 文件约 78 MB。将原始配置下载到 data/src/raw/chronos/m5，不抽样地加载全部记录至 data/src/shard_dataset/v0-grad-mech-test/m5/load，然后按稳定的源记录顺序选取最先出现的 1,000 条符合条件的轨迹（数值均有限、非恒定，且长度至少为 576 个点）。不得随机抽样。在 outputs/ 下的输入清单中保存准确的原始行 ID 和选中轨迹长度。此前选用的 Boomlet 数据配置无法提供 1,000 条轨迹，因此改用当前方案。
- 窗口划分：上下文长度 512，预测长度 64，窗口总跨度 576，步长 1。对每条轨迹，将每个窗口与其后所有存在跨度重叠的窗口比较（滞后量 1 至 575）。主要科学结果为滞后量 1，此时重叠比例为 575/576（99.83%）。滞后量 576 时没有重叠，因此排除。每条轨迹边界处都要重置窗口和梯度状态；不同轨迹之间绝不配对。
- 预热：从登记的 ChronosTiny 预训练检查点开始，在梯度分析前，使用选定的同一批 1,000 条轨迹进行微调。采用 100 个优化器步数、批大小 32、学习率 1e-6；该学习率与批大小取自 configs/rage/development_chronos_source_0914.json 中的一组候选值。为每条选定轨迹确定性地构造 4 个均匀分布的预热窗口，再使用现有的带种子训练采样器（种子 0）从这 4,000 个窗口中采样。此方案固定为 v0 预热协议。将预热后的检查点保存至 outputs/train/v0-grad-mech-test/，并在不再更新参数的该检查点上计算所有探测梯度。如果数据加载器无法在不改变已记录采样语义的前提下组成批大小为 32 的批次，则停止并记录问题，不得擅自更改协议。
- 梯度定义：对每个窗口分别计算所有可训练参数相对于已配置 Chronos 训练损失的梯度，所有计算使用相同模型状态和预处理。不得投影、草图化或缩减梯度维度，比较完整梯度向量。记录每个窗口的损失、梯度范数、轨迹 ID、窗口起点及检查点 ID。
- 成对分析：枚举滞后量 1 至 575 的全部配对（所有且仅有窗口跨度存在非零重叠的配对）。使用余弦相似度衡量方向，使用范数比和对称相对范数差衡量幅值。按滞后量和轨迹报告结果，并以滞后量 1 的分布作为主要结论。连续指标为主要证据；只有在余弦相似度 >= 0.9 且相对范数差 <= 0.1 时，才可将配对标记为接近。若显存允许，使用全精度模型和梯度计算；在 GPU 上分批计算完整向量点积，并以 FP32 累积点积和平方范数。不得物化包含所有配对梯度的张量。
- 梯度缓存与存储：每次只处理一条轨迹。固定大小的环形缓冲区最多保留 576 个完整梯度（当前梯度及最多 575 个前序梯度）；某个槽位完成最后一次非零重叠比较后，应立即覆盖最旧槽位；轨迹结束时释放缓冲区。如果主机 RAM 余量不足，将环形缓冲区存为固定大小的 FP32 内存映射文件，路径为 /root/autodl-tmp/rage/outputs/cache/v0-grad-mech-test/；绝不为每个窗口追加一个梯度文件。按轨迹和滞后量在线汇总配对指标（数量、均值、方差、最小值/最大值和固定分箱直方图），并逐步将摘要写入 outputs/；不得逐配对写入记录，因为全量配对表本身可能非常庞大。运行日志写入 logs/。如果 GPU 显存不足以容纳 FP32 梯度或比较批次，则将完整梯度缓存和 GPU 比较操作数切换为 BF16，同时保留所有参数坐标；范数和点积仍以 FP32 累积，并记录精度模式。发生显存不足时，不得采用投影或降维作为应对方式。
- 资源估算：当前数据盘剩余空间约 503 GB，主机可用 RAM 约 351 GiB；GPU 为 RTX 4080，显存 32 GiB。参数量 8,394,496 时，576 个梯度组成的环形缓冲区在 FP32 下约占 18.02 GiB，在 BF16 下约占 9.01 GiB，尚未计入文件系统开销。缓存保留在主机 RAM 或固定大小的数据盘内存映射文件中，不驻留于 GPU。分批将梯度块传至 GPU 进行成对计算；若 FP32 传输或模型反向传播超出显存，则模型参数和完整梯度值改用 BF16。此方案限制了缓存空间，但所有配对的计算量仍然很大：对于具有 W 个窗口的轨迹，配对数量为 sum(W - lag for lag in 1..min(575, W-1))。
- 完成标准：检查点和输入清单均不可变且有记录；恰好处理 1,000 条符合条件的轨迹（若不足则记录并停止）；所有非零重叠配对均产生有限值指标；缓存峰值不超过 576 个梯度，过期槽位均已覆盖；只有在输出和摘要全部完成后，才写入 .complete 标记文件。

| 批次 | 工作内容 | 依赖条件 | 状态 |
|---|---|---|---|
| 1 | 登记固定版本的 ChronosTiny 检查点，并验证模型加载和准确参数量 | Hugging Face 可访问，且指定不可变版本 | 检查点已下载，注册表校验通过；已添加 chronos-tiny 学生模型条目 |
| 2 | 获取并加载 Chronos m5；枚举稳定轨迹 ID 和长度；固定最先出现的 1,000 个符合条件的 ID | HF 数据集及固定版本可访问 | 原始数据获取及完整数据包加载已完成；已解码 30,490 条轨迹，并验证最先出现的 1,000 条符合条件的记录 |
| 3 | 实现并检查 100 步预热及 GPU 完整梯度环形缓冲区成对探测 | 批次 1-2 | 已在 scripts/exp/v0_grad_mech_test.py 中实现 |
| 4 | 执行预热，然后比较所有非零重叠梯度 | 批次 3；GPU 和数据盘 | 已完成：1,000 条轨迹与 10 份 CRB BF16 缓存完成，源实验 .complete 已写入 |
| 5 | 汇总方向、幅值、重叠关联、资源使用及局限 | 批次 4 | 已完成；结果见 docs/RESULTS.md，探测产物已归档 |


### rage-smoke-static-cycle

- 首次启动：2026-11-07
- 最后更新：2026-11-07
- 目标：验证从已准备资源到静态选择，再到 rr 和 topk 训练及评估的最小完整流程。
- 数据：3 条确定性合成源序列，每条包含 20 个 float64 数值。冒烟测试使用上下文长度 4、预测长度 4、选择比例 25%。
- 数据准备：通过 BundleLoadWriter 写入源 ShardDatasetBundle，然后生成符合条件的长度为 8 的窗口及稳定源 ID。
- 方法：rr 按首次出现的顺序轮流处理序列 ID，并贪心选取每个序列组中剩余窗口里分数最高的一个。topk 按每个窗口提供的有限标量分数降序排序；分数相同时保留输入顺序。
- 训练：使用 CPU 上的 torch.nn.Linear(4, 4) 测试模型，训练 1 个 epoch，批大小为 1，学习率为 0.001，随机种子为 7。此测试通过现有的 run_arm_training 训练及评估路径运行，不下载预训练检查点。
- 评估：使用 2 个留出的冒烟测试窗口、实体标签 a 和 b、上下文长度 4、单位测试尺度，并输出 MASE。
- 完成标准：准备数据包通过校验且可读取；两种选择器均返回预期预算；两次训练均成功完成并得到有限的 MASE；选中产物和日志均位于项目标准路径下。
- 脚本与测试：scripts/select_static.py、src/rage/static_selection.py、tests/test_static_selection.py、tests/test_minimal_cycle.py。

| 批次 | 工作内容 | 依赖条件 | 状态 |
|---|---|---|---|
| 1 | 安装并验证全部 6 个 Conda 环境 | Conda 和软件包索引 | 已通过；全部 6 个环境均通过 pip check、导入检查和 CUDA 冒烟测试 |
| 2 | 验证数据准备、选择、训练及评估流程 | 批次 1 | 已通过集成测试 |
| 3 | 运行完整测试套件并修复迁移回归问题 | 批次 1 | 已通过：784 项通过，3 项跳过 |
| 4 | 验证所有配置档的一键安装程序 | 批次 1 | 已通过 bash _install.sh --verify-only |

- 2026-11-07：首次 CPU 冒烟运行使用 12 个已准备的 NPZ 窗口，测试了 rr、topk 和 run_arm_training。rr 选中 3 行，完成 3 个优化器步数，MASE 为 4.5501。topk 选中 3 行，完成 3 个优化器步数，MASE 为 4.2771。之后添加了数据包到选择器再到训练的集成测试，并已通过（见批次 2）。

### v0-active-value-rank diagnostic extension

First proposed: 2026-11-09. Last modified: 2026-11-09. Status: completed. These two batches reuse the same 100 reference pools and cached full/head cosine values as batches 1 and 2. No gradient, model, or downstream training calls are made.

Batch 3-proxy-coverage uses k=ceil(0.10n) per reference, with n from 714 to 766. Stable descending score order breaks ties by existing candidate order. Save the proxy rank of every true top-k window, coverage at every 10 proxy candidates plus k,2k,3k,4k,5k,7k,n, and six rank bins (1-k, k+1-2k, 2k+1-3k, 3k+1-4k, 4k+1-5k, and the remainder). Produce one two-panel coverage and rank-location figure per reference. The full cached scores are used only as the diagnostic oracle.

Batch 4-equal-budget-query uses B={0,8,16,32,64}, k=ceil(0.10n), and Recall@k. At B=0 all policies use the raw head proxy. At B>=8, all methods share the exact eight proxy-quantile labels from batch 2; these count toward B. Pure boundary queries unselected candidates in proxy-rank order k,k+1,k-1,k+2,k-2 and so on. Proxy high queries unselected candidates in descending proxy score order. Random queries use 30 fixed seeds, independently per reference. These three simple policies use exact cached full cosine only for queried candidates and raw head cosine for all others. The existing complex boundary policy reuses the batch-2 ridge correction, uncertainty, and boundary query replay, including its same eight initial labels and exact total budget. No policy may consult unqueried full scores when choosing candidates. Both raw scores are cosine values; their observed mean true-minus-proxy offset is 0.0176 and mean per-reference standard deviations are 0.1484/0.1565. The complex method has the correction mechanism as its tested component, so its unqueried-score rule differs by design. This comparison is a retrospective same-pool replay, not an independent held-out evaluation.

| Batch | Work | Dependency | Completion condition | Status |
|---|---|---|---|---|
| 3-proxy-coverage | True top-k rank, coverage, and rank bins per reference | Batch 1 caches | 100 reference figures, rank and coverage tables, .complete | Completed |
| 4-equal-budget-query | Four methods at shared budgets, 30 random seeds | Batches 1-3 and existing batch 2 replay | Per-reference and summary tables, query manifest, comparison figure, .complete | Completed |

Detailed results and limitations: docs/results/v0-active-value-rank-diagnostics.md.


## v0-crb-var

First proposed: 2026-11-09. Last modified: 2026-11-09. Status: completed. Goal: test whether native CRB adaptive partitioning places more boundaries in regions with higher within-block gradient variance or direction dispersion than a matched uniform partition. Reused the ten completed BF16 full-gradient caches from v0-crb-reconst-test; no gradients were recomputed. For each trajectory with N legal starts, B=ceil(N/64), giving approximately one allowed gradient computation per 64 starts. The frozen CRBEngine charged B probes across coverage, error-risk and recursive minimum-kappa stages; two endpoint probes were included. The baseline used B evenly spaced anchors including endpoints. All block metrics use every cached full-gradient vector. Results: equal-track mean relative variance 0.6410 CRB vs 0.6802 Uniform, lower for CRB on 10/10 tracks; mean direction dispersion 0.4051 vs 0.4402, lower on 10/10. Pooled block-size Spearman association with relative variance and direction dispersion was 0.6027/0.5830 for CRB and 0.0876/0.0602 for Uniform; within-track CRB correlations were positive on all ten tracks, meaning longer intervals retained greater local dispersion. Raw absolute variance was 10.1357 vs 9.8196 and did not consistently favor CRB. Full report: docs/results/v0-crb-var.md.

## v0-proxy-dp-segmentation

First proposed: 2026-11-09 22:19 CST. Last modified: 2026-11-09 22:19 CST. Status: running. Goal: test whether an exact DP partition minimizing the unweighted sum of interval trace variances in reconstructed output-head proxy gradients improves true-gradient interval metrics over an equal-count uniform partition.

Use ten fixed trajectories and existing proxy-gradient arrays at , with true full-gradient BF16 caches at . No gradients are recomputed. For each trajectory with N starts, set K=ceil(N/64). Sample proxy gradients every four starts and include the terminal start when needed to permit interpolation through the full trajectory. Linearly reconstruct the proxy vector at every start. Define a vector interval variance as trace of the population covariance, equal to mean squared Euclidean deviation from the interval vector mean. Optimize the unweighted sum of interval variances, not weighted by interval length.

Solve exact DP: DP[k,i]=min_j(DP[k-1,j]+V(j,i)) subject to interval length at least 16. Lmax is unrestricted, effectively N. Check K*16<=N. Save backpointers and recover K+1 contiguous boundaries. The same K intervals form the evenly spaced baseline. Evaluate both partitions against every true full-gradient vector using equal-interval mean relative variance, equal-interval mean direction dispersion (mean 1-cos), interval-length-weighted direction dispersion, and equal-interval mean absolute variance. Preserve per-interval boundaries, lengths, and metrics. Config: configs/rage/v0-proxy-dp-segmentation.json. Entry point: scripts/temp/v0_proxy_dp_segmentation.py. Output: outputs/train/v0-proxy-dp-segmentation/.

| Batch | Work | Status |
|---|---|---|
| 1 | Run ten cached proxy reconstructions, exact DP, and matched uniform partitions; evaluate true-gradient metrics | Running; first trajectory completed; resume checkpoints enabled |
| 2 | Validate K, minimum interval length, complete coverage, finite outputs, and completion marker; summarize results | Pending |
