# Exploratory REF CRB phase-budget sweep (2026-09-26)

This extends the two quick Chronos / `time_cphl_h` REF checks without loading a model or computing any new gradients. All 1,939 full-window truth rows are read from the previously validated production/private caches. The three chosen short trajectories retain their original total BP budgets, 4/9/18. We replayed every feasible integer cover/error/split allocation with cover≥2 (both endpoints) and error,split≥1: 1, 21, and 120 allocations respectively. Ratios use deterministic nearest-feasible integer apportionment; on the very short trajectory every ratio collapses to 2/1/1. We searched positive integer ratio weights ≤24, then compared all resulting curves to the same truth. This is *in-sample exploratory tuning*, not held-out validation or a V7 protocol change.

| Ratio | Actual phase BP (REF IDs 1 / 3 / 5) | Pooled relative reconstruction RMSE |
|---:|---|---:|
| Native 1:1:1 | 2/1/1 · 3/3/3 · 6/6/6 | 0.9174 |
| 3:1:1 | 2/1/1 · 5/2/2 · 11/4/3 | 0.8972 |
| 3:2:1 | 2/1/1 · 5/3/1 · 9/6/3 | 0.8909 |
| 4:1:1 | 2/1/1 · 6/2/1 · 12/3/3 | 0.8916 |
| 9:1:1 | 2/1/1 · 7/1/1 · 15/2/1 | **0.8809** |
| Uniform anchors | Same 4/9/18 total BP | 0.8888 |

The best observed common ratio plateau starts at 9:1:1: it improves pooled reconstruction error by about 4.0% relative to native 1:1:1 and 0.9% relative to uniform anchors. However, the shortest trajectory is unchanged and uniform is better on it. The coverage-heavy ratio benefits the longest trajectory enough to win the window-weighted pooled metric. The equal-trajectory average of per-trajectory RMSE remains *slightly* better for uniform (0.8795) than 9:1:1 (0.8800). For the equal-trajectory **reference mean gradient**, a complementary target used by V7 gradient baselines rather than the full RAGS block matcher, uniform has relative vector error 0.6533 versus 0.7813 for 9:1:1 (native 1:1:1: 0.9109). Thus 9:1:1 is a promising reconstruction *candidate*, not evidence that it is the better V7 reference representation or improves MASE.

Interpretation: 1:1:1 spends too much of this small budget on later adaptive stages for these selected curves; increasing coverage helps. A less aggressive candidate is 3:2:1 (0.8909 pooled), with 9:1:1 as a high-coverage diagnostic. Freeze neither on this tuning set. A held-out task/reference check and downstream scoring/training assessment would be needed to choose a new scientific protocol.

Reproducible script: `scripts/quick_crb_budget_sweep.py`; raw ratio/allocation scores and report: `ssh exp:/root/autodl-tmp/TS3/outputs/quick-crb-budget-sweep-v4-20260926/`. The earlier v1/v2/v3 sweep outputs remain intact. No V7 process, cache identity, selection or result was changed.
