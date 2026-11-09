# v0-grad-traj-level-norm-gap

Last modified: 2026-11-08

## Question and method

Across the same ten retained continuous M5 trajectories, do trajectory-level mean-gradient norms differ appreciably for complete backpropagation gradients and output-head proxy gradients? For each track and each gradient kind, the primary quantity is the Euclidean norm of the vector average over all legal windows, `||mean_i g_i||`. To distinguish this from a different common reading, also report the arithmetic mean of per-window norms, `mean_i ||g_i||`. The full and output-head vectors occupy different parameter spaces, so their raw magnitudes are not directly comparable; compare relative spreads and trajectory ranks within each kind.

Use the saved complete all-coordinate full-gradient and output-head Gram matrices, with no new model calls. For N windows with Gram matrix G, calculate the squared norm of the mean vector as `mean(G)`, the mean per-window norm as `mean(sqrt(diag(G)))`, and the population vector variance as `mean(diag(G))-mean(G)`. Validate each track marker, manifest window count, Gram shape, finite values, positive diagonal, and symmetry. Report ten-track minimum, maximum, maximum/minimum, median, mean, and population coefficient of variation. Correlate the two kinds across ten paired trajectories using Pearson and Spearman. These are descriptive values for a fixed selected sample, not independent-window significance tests.

Entry: `scripts/v0_grad_traj_level_norm_gap.py`. Environment: `rage` Conda. Command: `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src /root/autodl-tmp/rage/envs/rage/bin/python -u -m scripts.v0_grad_traj_level_norm_gap`. Output: `outputs/train/v0-grad-traj-level-norm-gap/`. Log: `logs/v0-grad-traj-level-norm-gap/`.

## Per-trajectory observations

| Track | Windows | Full norm of mean | Head norm of mean | Full mean norm | Head mean norm |
|---|---:|---:|---:|---:|---:|
| 00 | 607 | 1.5050 | 0.1293 | 4.2293 | 0.5142 |
| 01 | 1388 | 0.8950 | 0.0897 | 3.3427 | 0.4041 |
| 02 | 1310 | 0.8107 | 0.0642 | 3.2879 | 0.3974 |
| 03 | 1383 | 0.8092 | 0.0607 | 3.0087 | 0.3563 |
| 04 | 1320 | 0.7780 | 0.0721 | 3.0925 | 0.3615 |
| 05 | 1393 | 1.0007 | 0.0730 | 3.3882 | 0.3850 |
| 06 | 1393 | 1.1034 | 0.1094 | 3.6179 | 0.4230 |
| 07 | 1270 | 1.3442 | 0.1515 | 4.0210 | 0.4918 |
| 08 | 1378 | 0.9556 | 0.1047 | 3.2492 | 0.4033 |
| 09 | 1394 | 0.8766 | 0.0791 | 3.3473 | 0.3829 |

The primary norm of the mean gradient varies from 0.7780 (track 04) to 1.5050 (track 00) for full gradients, a 1.9345-fold span, with population CV 0.2291. For output-head gradients it varies from 0.0607 (track 03) to 0.1515 (track 07), a 2.4960-fold span, with CV 0.3040. Full and head primary-norm rankings across tracks have Pearson 0.8835 and Spearman 0.8788. Tracks 00 and 07 are high in both kinds; tracks 03 and 04 are relatively low in both. The exact maximum and minimum tracks differ by gradient kind.

The secondary mean of per-window norms has a narrower spread: full 3.0087-4.2293 (1.4057-fold, CV 0.1073), output-head 0.3563-0.5142 (1.4433-fold, CV 0.1202). Its across-track full/head association is Pearson 0.9764 and Spearman 0.8061. The norm of a trajectory mean is smaller than the mean of its window norms because gradient directions can cancel; the ratios range from 0.2466 to 0.3558 for full gradients and 0.1615 to 0.3081 for output-head gradients. Thus the larger spread of mean-vector norms reflects both gradient scale and directional cancellation.

These are appreciable descriptive differences among the ten selected tracks. The tracks are not a random sample of all M5 series, windows within a track overlap, and there are only ten track-level observations. No formal population-level statistical significance or generalization claim is made. Raw metrics, spread statistics, paired correlations, and `.complete` are in the output directory.
EOF'