# v0-active-value-rank: expanded ranking and active correction

Last modified: 2026-11-09

Status: Both requested batches complete. This is a retrospective replay on ten selected M5 trajectories at one warmed Chronos-T5 Tiny checkpoint. It does not measure downstream model performance.

## Protocol

Batch 1-expanded-cross-rank used the 100 existing reference windows, ten per trajectory. Each reference was scored against every 16th legal start in each of the other nine trajectories, excluding starts reserved as references. Candidate counts ranged from 714 to 766 (mean 721.8), versus 24 in the earlier Cross-Rank test. The full-gradient cosine targets came from the existing 100-reference score arrays; output-head cosine was computed from the saved head-gradient vectors. Embedding cosine and absolute loss difference were retained as candidate features for batch 2. Source windows 16 starts apart still overlap in raw observations.

Batch 2-active-correction hid all full scores except queried candidate labels. Every method began with the same eight proxy-quantile labels, then selected up to 64 labels by random, highest proxy score, predictive uncertainty, or boundary uncertainty. A six-feature per-reference ridge model predicted full-minus-head score. Queried candidates used exact full scores; unqueried candidates used model-corrected scores. The boundary policy targeted the predicted top-10-percent cutoff. Budgets were 8, 16, 32, and 64 total candidate labels. Final top fractions were 5, 10, and 20 percent. No full labels from one reference were used to train another reference's model.

The primary selection metrics are true top-k recall and oracle regret. Oracle regret is the mean true full-gradient cosine of the oracle top-k set minus that of the selected top-k set. All means below average the 100 reference results equally (ten per reference track). Because windows and references within a track are dependent, these are descriptive summaries, not independent-sample significance tests.

## Batch 1 results

The mean per-reference Spearman correlation between full and head cosine is 0.7104. Per-reference-track mean rho ranges from 0.6133 to 0.7768. Direct proxy selection gives:

| Top fraction | Mean top-k recall | Mean oracle regret |
|---:|---:|---:|
| 5% | 0.3050 | 0.09509 |
| 10% | 0.4023 | 0.07525 |
| 20% | 0.5565 | 0.05126 |

These values describe the expanded, stride-16 pool and cannot be directly compared as a matched improvement over the prior 24-candidate pool.

## Batch 2 results

At the pre-registered 10-percent decision fraction:

| Candidate full-label budget | Random regret | Highest-proxy regret | Uncertainty regret | Boundary regret | Boundary recall |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.08775 | 0.08775 | 0.08775 | 0.08775 | 0.3617 |
| 16 | 0.07996 | 0.07337 | 0.07387 | 0.06536 | 0.4293 |
| 32 | 0.06965 | 0.06244 | 0.06981 | 0.05454 | 0.4764 |
| 64 | 0.06249 | 0.05148 | 0.06441 | 0.04192 | 0.5415 |

The direct proxy (zero candidate full labels) has regret 0.07525 and recall 0.4023. At budget 8, the common calibration model worsens the direct-proxy result; all policies are identical because no policy-specific queries have yet been made. At budgets 16, 32, and 64, the boundary policy lowers mean regret relative to direct proxy by 0.00989, 0.02071, and 0.03333. The corresponding improvements occur in 9, 10, and 10 of the ten reference-track means, respectively; 70, 92, and 99 of 100 individual references improve.

At 5-percent selection, highest-proxy querying is better than boundary querying at budget 64 (regret 0.04801 versus 0.05138). The boundary policy was designed around a 10-percent cutoff, so this is an important objective-specific limitation. At 20-percent selection, boundary querying has lower mean regret than highest-proxy querying at all three post-initial budgets.

## Call-cost interpretation

The prior matched batch-size-one cost benchmark measured medians of 14.21 ms for one head-gradient proxy and 42.80 ms for one full gradient. The modeled per-reference cost includes a proxy for every candidate, one reference proxy, the budgeted candidate full calls, and one shared reference full call. It averages about 10.66, 11.00, 11.68, and 13.05 seconds at budgets 8, 16, 32, and 64. The direct-proxy row includes the same shared full-reference setup for comparison, although direct proxy alone would not require that call. These are per-reference cost estimates from separate calls with no sharing across references, not a measured online pipeline. The cached embedding and loss features may be reusable from the proxy forward pass, but their incremental fused cost was not benchmarked. The measured CPU replay model time is recorded separately in summary.csv and is small here.

## Validation and limits

Both batch .complete markers are present. The new track-01/start-0 versus reference track-00/start-36 pair agrees with the earlier independently generated Cross-Rank values to float rounding for full cosine, head cosine, embedding cosine, and loss difference. All cross-track full scores and all proxy features in the selected pools were finite. The input manifests record reference identities, candidate stride, source manifests, budgets, and fixed parameters. Query order for each reference and policy is preserved.

The ten trajectories are a selected late-manifest sample from one dataset and one checkpoint. The 100 references are dependent within tracks. The target is cosine to a reference gradient, not an observed training benefit. The present result supports further testing of boundary-aware querying for this target and 10-percent selection, but does not establish downstream training improvement, cross-dataset generalization, or measured end-to-end cost savings. The initial eight-label degradation and 5-percent exception are reasons to test calibration stability and cutoff-specific policies next.

Outputs: outputs/train/v0-active-value-rank/batch-1/ and batch-2/. Logs: logs/v0-active-value-rank/. Entry: scripts/v0_active_value_rank.py.
