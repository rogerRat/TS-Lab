# v0-active-value-rank diagnostic extension

Last modified: 2026-11-09

Status: batches 3-proxy-coverage and 4-equal-budget-query complete. All outputs are retrospective replays of cached full-gradient cosine and output-head cosine scores. No new model or gradient was trained or computed.

## Inputs and decision rule

The 100 fixed references and ten source trajectories are the same as the completed v0-active-value-rank batches 1 and 2. Each reference has 714-766 cross-track stride-16 candidates; the mean is 721.8. Reserved reference starts are excluded. For each reference, k=ceil(0.10n), so k varies with candidate count rather than being fixed at 72. Stable descending score order resolves exact ties by existing candidate order. Recall@k is the intersection of the selected and full-gradient oracle top-k sets divided by k. Results average reference recalls equally; ten references come from each reference track.

## Batch 3: where true top-k windows appear in proxy order

| Proxy pool | Mean true top-k recall | Minimum reference recall | Median reference recall |
|---:|---:|---:|---:|
| k | 0.4023 | 0.0411 | 0.4156 |
| 2k | 0.6480 | 0.0685 | 0.6552 |
| 3k | 0.7950 | 0.1233 | 0.8132 |
| 4k | 0.8744 | 0.1781 | 0.9263 |
| 5k | 0.9265 | 0.3014 | 0.9733 |
| 7k | 0.9754 | 0.5753 | 1.0000 |

Across references, a mean 29.22 true top-k windows lie at proxy ranks 1-k, 17.87 at k+1-2k, 10.70 at 2k+1-3k, 5.79 at 3k+1-4k, 3.79 at 4k+1-5k, and 5.33 beyond 5k. Sixty-five references have at least one true top-k window beyond 5k; fourteen have at least ten. Reference 70 is an extreme case: coverage is 0.0411 at k, 0.1233 at 3k, and 0.5753 at 7k. Thus many missed windows are near the front of the proxy list, but substantial severe under-ranking remains. A boundary-only query rule cannot recover those distant windows within a small budget.

The rank of each of the 7,270 true top-k candidates is in batch-3/true_top_proxy_ranks.csv. Batch-3/coverage.csv contains every 10-candidate step plus k multiples; batch-3/rank_bins.csv contains the six bins. The 100 figures in batch-3/figures/reference-NN.png show a separate coverage curve and rank distribution for each reference. Batch-3/coverage_summary.csv records cross-reference coverage without hiding the minimum and median.

## Batch 4: equal-budget query comparison

At B=0, all methods rank the raw head proxy and have mean recall 0.4023. For B>=8, all methods begin with the same eight proxy-quantile labels already used in batch 2; those labels count against B. Pure boundary visits proxy ranks k,k+1,k-1,k+2,k-2 and so forth, skipping labels already chosen. Proxy high visits ranks 1,2,3 and so forth, also skipping initial labels. Random visits one of 30 seeded permutations per reference. These three methods replace queried proxy scores with exact cached full scores and leave unqueried scores raw. The existing complex method uses the batch-2 ridge residual correction, predictive uncertainty, and boundary utility, with exact scores for queried points. Its saved per-reference batch-2 recall rows are reused exactly. All four methods use the same candidate pools, targets, evaluation, budgets, and initial eight labels.

| Budget B | Pure boundary | Proxy high | Random, 30 seeds | Existing complex |
|---:|---:|---:|---:|---:|
| 0 | 0.4023 | 0.4023 | 0.4023 | 0.4023 |
| 8 | 0.4048 | 0.4048 | 0.4048 | 0.3617 |
| 16 | 0.4214 | 0.4130 | 0.4083 | 0.4293 |
| 32 | 0.4482 | 0.4336 | 0.4152 | 0.4764 |
| 64 | 0.5064 | 0.4776 | 0.4295 | 0.5415 |

Complex minus pure-boundary recall is -0.0431, +0.0078, +0.0282, and +0.0352 at B=8,16,32,64. The corresponding reference wins/ties/losses for complex are 31/9/60, 47/8/45, 59/4/37, and 58/3/39. At B=64, complex beats pure boundary on 9 of 10 reference-track means, with one negative track mean. Its average gain is real on this fixed replay but does not dominate every reference. At B=8 the common calibration model is worse than the simple raw-score substitution.

Random seed means at B=16,32,64 have sample standard deviations 0.00076, 0.00137, and 0.00227 across the 30 seeds. At B=64 the worst and best seed means are 0.4252 and 0.4343. The B=8 random repeats are identical because the entire budget is the shared initial set. Random per-reference averages can still be poor despite small cross-reference seed-mean variability.

## Interpretation and limits

The proxy is useful as a candidate screen but misses distant true top-k windows for many references. Pure boundary is a meaningful simple baseline and improves over proxy high and random at B=16-64. The existing complex method is worse at eight labels, marginally better at sixteen, and more clearly better at 32-64 on this same-pool replay. The evidence supports retaining the complex method for further held-out testing at moderate budgets, while preserving the simple boundary baseline and investigating proxy under-ranking before relying on a narrow candidate pool. It does not justify a new model or new query strategy now.

Both raw scores are cosine similarities on the same numerical range. Across references the mean full-minus-head score offset is 0.0176; mean per-reference score standard deviations are 0.1484 for full and 0.1565 for head. Raw score substitution for the three simple policies is therefore interpretable, but the proxy and full distributions are not identical. The complex method is allowed its existing learned correction as the component under test; it uses different unqueried-score estimates by design. This comparison does not isolate the correction model from its query policy. The 100 references share ten trajectories, and some stride-16 windows overlap in raw observations. No held-out references were reserved before the earlier method was developed. These are descriptive same-sample results, not independent significance or downstream training evidence.

Output markers and artifacts: outputs/train/v0-active-value-rank/batch-3/ and batch-4/. The method comparison curve is batch-4/method_comparison.png; the per-reference metrics, policy query manifest, and summary are adjacent. Entry point: scripts/v0_active_value_diagnostics.py. Logs: logs/v0-active-value-rank/.

The combined 100-reference coverage curves and mean rank-bin chart are in batch-3/coverage_and_rank_summary.png. Batch-4/random_seed_summary.csv retains each of the 120 budget-seed aggregates. The one-time summary entry is scripts/temp/v0_active_value_summary.py.
