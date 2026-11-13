# v0-uni-reconst-scale: Uniform linear reconstruction by anchor spacing

Last modified: 2026-11-08

## Protocol

Use the same ten retained M5 trajectories and complete BF16 gradients as v0-vage, represented by validated full-coordinate Gram matrices. Each trajectory has N=T-575 legal starts for 512 context and 64 prediction observations. For each spacing denominator s in {1,2,4,8,16,32,64,128}, set K=ceil(N/s) unique complete-gradient anchors per trajectory. Place K anchors evenly over the entire legal-start range, including both endpoints. Reconstruct each unobserved gradient by linear interpolation between its neighboring true-gradient anchors. Queried anchors return their true gradient exactly. No proxy gradient is used.

For each trajectory, mean L2 is the average of the per-start Euclidean norm of the difference between the true and reconstructed full vectors. Mean cosine is the average per-start cosine between those vectors. The primary summary averages the ten trajectory means equally. A window-weighted summary is also saved. The same full-gradient Gram reconstruction formula used in the preceding VAGE replay evaluates the unqueried starts; queried anchor errors are set exactly to zero and their cosine to one. This is a retrospective budget sweep, not a measurement of new model runtime. All 12,836 legal starts contribute at every budget, including the queried anchors.

## Results

| Starts per anchor s | Total anchors, ten tracks | Mean L2, lower better | Mean cosine, higher better |
|---:|---:|---:|---:|
| 1 | 12,836 | 0.0000 | 1.0000 |
| 2 | 6,420 | 0.9650 | 0.8952 |
| 4 | 3,213 | 1.6969 | 0.7941 |
| 8 | 1,609 | 2.2654 | 0.6956 |
| 16 | 808 | 2.7382 | 0.6002 |
| 32 | 406 | 3.1250 | 0.5063 |
| 64 | 204 | 3.4824 | 0.3971 |
| 128 | 103 | 3.8809 | 0.2563 |

The q=64 row reproduces the Uniform mean L2 and cosine from v0-vage. Direction alignment declines substantially as anchors become sparse: approximately 0.90 at s=2, 0.70 at s=8, 0.51 at s=32, and 0.40 at s=64. These results describe this fixed model, checkpoint, selected ten M5 trajectories, BF16 gradient caches, and whole-trajectory linear interpolation. They do not establish performance on other datasets or a nonlinear reconstruction method.

## Validation and artifacts

All ten proxy-analysis cache completion markers, ten full-gradient cache markers and file sizes, Gram shapes, and finite values were checked. Exactly 80 trajectory-scale metric rows were written. The s=1 case was checked to have mean L2 zero and cosine one on every trajectory. The s=64 result matches the preceding Uniform replay. Config: configs/rage/v0-uni-reconst-scale.json. Entry: scripts/v0_uni_reconst_scale.py. Raw output: outputs/train/v0-uni-reconst-scale/trajectory_metrics.csv, summary.csv, config_snapshot.json, and .complete. Logs: logs/v0-uni-reconst-scale/.

## Pair-mean reconstruction supplement

Last modified: 2026-11-08

Keep the exact Uniform anchors and K=ceil(N/s) budgets of the original eight spacing groups. Between adjacent true-gradient anchors a and b, replace linear interpolation by the fixed representative (g_a+g_b)/2 for every unobserved start. Queried anchors return their true gradient exactly. There is no extrapolation because Uniform includes both global endpoints. No model or proxy calls were added.

| Starts per anchor s | Pair-mean L2 | Linear L2 | Pair-mean cosine | Linear cosine |
|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| 2 | 0.9652 | 0.9650 | 0.8952 | 0.8952 |
| 4 | 1.7804 | 1.6969 | 0.7836 | 0.7941 |
| 8 | 2.3777 | 2.2654 | 0.6785 | 0.6956 |
| 16 | 2.8264 | 2.7382 | 0.5786 | 0.6002 |
| 32 | 3.1827 | 3.1250 | 0.4754 | 0.5063 |
| 64 | 3.5445 | 3.4824 | 0.3498 | 0.3971 |
| 128 | 3.8954 | 3.8809 | 0.2146 | 0.2563 |

Values are ten-trajectory equal-weight means. At s=1, both methods return exact gradients. At s=2, most gaps contain no unobserved starts and results are nearly identical. At every other tested spacing, the constant pair mean has larger mean L2 and lower mean cosine than linear interpolation. This comparison isolates the reconstruction rule because anchors and budgets are identical. It does not evaluate a learned or variance-adaptive representative.

Raw per-trajectory and summary tables: outputs/train/v0-uni-reconst-scale/pairmean/. Shared config: configs/rage/v0-pairmean-supplements.json. Entry: scripts/evaluate_pairmean_reconstruction.py. The supplement has its own .complete marker.
