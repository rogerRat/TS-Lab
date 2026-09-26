# Quick REF CRB 3:1:1 phase-budget ablation (2026-09-26)

This follows `quick_crb_uniform_ref_20260926.md` using exactly the same Chronos checkpoint, `time_cphl_h` reference fragments 1/3/5, BF16-decoded full-gradient truth, and total BP budget of 4/9/18 per trajectory. The only changed mechanism is the CRB cover:error:split allocation. Exact integer budgets were assigned by largest remainder with ties ordered cover, error, split: 4→(2,1,1), 9→(5,2,2), 18→(11,4,3). Thus the shortest trajectory is unchanged. The original 1:1:1 allocations were (2,1,1), (3,3,3), (6,6,6). Other CRB settings and within-trajectory linear reconstruction are identical. The adaptive CRB replay was given only the requested anchor gradients, despite the full truth being available behind a read-only lookup. No model was loaded and no new BP was run.

| REF fragment | Legal windows | BP each | Native 1:1:1 relative RMSE | CRB 3:1:1 relative RMSE | Uniform relative RMSE |
|---:|---:|---:|---:|---:|---:|
| 1 | 185 | 4 | 0.8765 | 0.8765 | 0.8501 |
| 3 | 511 | 9 | 0.9281 | 0.8950 | 0.8860 |
| 5 | 1,243 | 18 | 0.9194 | 0.9053 | 0.9026 |
| All windows | 1,939 | 31 | 0.9174 | 0.8972 | 0.8888 |

3:1:1 improves pooled relative Frobenius RMSE over native 1:1:1 by 0.0201 absolute (2.2% relative), but remains 0.0084 worse than uniform. On the common non-anchor windows of all three methods, errors are 0.9243, 0.9047, and 0.8958 respectively, so the ordering is not caused by exact anchor rows. This is a three-trajectory projected-gradient reconstruction check, not evidence for final selection/training MASE. The result suggests more coverage is helpful here, while the specific CRB adaptive placement still does not beat uniform in this small sample.

Reproduce with `scripts/quick_crb_ratio_ref.py`. Machine-readable result: `ssh exp:/root/autodl-tmp/TS3/outputs/quick-crb-ratio311-ref-20260926/result.json`; readable report is adjacent. The script verifies the old full-truth cache identity and replays the original 1:1:1 and uniform errors before accepting the 3:1:1 result.
