# Quick V7 reference-gradient reconstruction check (2026-09-26)

This is a small, post hoc mechanism check, not a new V7 selection/training cell. On `chronos-base` / `time_cphl_h`, reference fragment IDs 1, 3, 5 have 185, 511, and 1,243 legal windows. Chronos was chosen before the full scan because its already-cached nearby gradient directions looked relatively stable among the three V7 models; this screening was small and does not establish global smoothness.

For each fragment, the native CRB algorithm was replayed against its *existing production gradient rows* using the frozen V7 configuration (C=512, H=64; cover/error/split each 0.005). The actual distinct BP counts were 4, 9, 18, exactly matching the production asset receipt. The comparator places the same number of anchors uniformly including both endpoints. Both sides reconstruct every window's 4,096-dimensional projected gradient by within-trajectory piecewise-linear interpolation. The ground truth is each window's BF16-stored projected gradient decoded to FP32 under the same checkpoint/precision/loss identity. The original cache was read-only; the 1,886 missing truth rows were computed once in an isolated cache on GPU1, while 53 existing rows were reused. This truth scan is an audit cost, not charged to either sparse method.

| Reference fragment | Legal windows | BP per method | Native CRB relative RMSE | Uniform relative RMSE |
|---:|---:|---:|---:|---:|
| 1 | 185 | 4 | 0.8765 | 0.8501 |
| 3 | 511 | 9 | 0.9281 | 0.8860 |
| 5 | 1,243 | 18 | 0.9194 | 0.9026 |
| All windows | 1,939 | 31 | 0.9174 | 0.8888 |

The uniform rule reduces pooled relative Frobenius RMSE by 0.0286 absolute (3.1% relative). Excluding the union of both methods' anchors, the corresponding errors are 0.9245 and 0.8955, so this is not just an anchor-exactness artifact. Uniform also has higher pooled mean per-window cosine, 0.5284 versus 0.4871. However both reconstructions remain poor in absolute terms at such a sparse budget. These three short, deliberately selected REF trajectories are not an unbiased cross-task sample. No downstream selection-probability, training, or MASE superiority is established.

Reproducible source: `scripts/quick_crb_uniform_ref.py`, with `tests/test_quick_crb_uniform_ref.py`. Full machine-readable result and private reusable truth cache are in `/root/autodl-tmp/TS3/outputs/quick-crb-uniform-ref-20260926/` on `ssh exp`. Do not delete or migrate the production gradient pool as part of this check.
