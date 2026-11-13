# v0-gradient-variance-decomposition

Last modified: 2026-11-08

## Question and fixed inputs

Does block-level vector variance mainly reflect gradient magnitude or directional change, and does controlling gradient scale and trajectory identity resolve the previously positive association between vector variance and linear-reconstruction cosine? Reuse exactly the 46 complete, nonoverlapping 256-start blocks from v0-var-proxy-estimate across the same ten retained trajectories. Reuse the complete full-gradient Gram matrices, eight-point output-head proxy variance, and four-anchor linear reconstruction at local positions 0, 85, 170, and 255. No model, gradient, or proxy calls were made.

For each block, let `r_t=||g_t||`, `u_t=g_t/r_t`, and use population averages over all 256 starts. Define mean gradient norm `E r_t`, amplitude variance `Var(r_t)=E r_t^2-(E r_t)^2`, full direction dispersion `D_h=1-||E u_t||^2`, total vector variance `V_g=E r_t^2-||E g_t||^2`, and relative vector variance `V_g/E r_t^2`. Here `D_h` is computed from complete full-gradient directions, not the output-head proxy. The exact directional variance component is `V_g-Var(r_t)=(E r_t)^2-||E g_t||^2`; it is not generally equal to `(E r_t)^2 D_h` when magnitudes and directions covary.

Recompute mean L2, absolute norm difference, directional cosine, and `1-cosine` from the same Gram blocks. Record both all 256 positions and the 252 non-anchor positions. Raw Pearson and Spearman correlations pool 46 blocks. Equal-track Spearman averages nine defined within-track correlations; track 00 has only two blocks. Controlled Pearson residualizes both variables on mean gradient norm and ten track fixed effects. Controlled Spearman ranks both variables and mean gradient norm globally, then residualizes the ranks on ranked mean norm and track fixed effects before correlating. These are descriptive partial associations, with no independent-window p-values. Leave-one-track-out ranges recalculate the controlled rank association after omitting each track in turn.

Entry: `scripts/v0_gradient_variance_decomposition.py`. Environment: `rage` Conda. Command: `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src /root/autodl-tmp/rage/envs/rage/bin/python -u -m scripts.v0_gradient_variance_decomposition`. Output: `outputs/train/v0-gradient-variance-decomposition/`. Logs: `logs/v0-gradient-variance-decomposition/`.

## Variance composition and scale

| Association across 46 blocks | Pooled Pearson | Pooled Spearman | Controlled rank association |
|---|---:|---:|---:|
| `V_g` versus mean gradient norm | 0.945 | 0.970 | Undefined: mean norm is a control |
| `V_g` versus amplitude variance | 0.690 | 0.569 | 0.663 |
| `V_g` versus direction dispersion `D_h` | -0.123 | -0.222 | 0.578 |
| `V_g` versus relative vector variance | -0.282 | -0.289 | 0.603 |

Mean gradient norm ranges 1.337-6.467 across blocks and strongly co-varies with absolute vector variance. Amplitude variance ranges 0.331-13.157. The exact amplitude-variance fraction `Var(r_t)/V_g` has median 0.176 and mean 0.202; it exceeds 0.5 in only one of 46 blocks. Thus amplitude variance does not numerically dominate the within-block vector-variance decomposition. The remaining component is direction-related but magnitude-weighted. The negative pooled `V_g` versus `D_h` association and positive controlled rank association show that block scale and track differences can mask directional relations. The term "scale driven" here describes cross-block variation in absolute `V_g`; it does not mean the within-block variance is mostly amplitude variance.

## Reconstruction associations

The following values use the 252 non-anchor positions per block. Correlations including anchors are identical to displayed precision because the same four exact anchors occur in every block: non-anchor mean errors equal all-position mean errors multiplied by `256/252`, while non-anchor mean cosine is an affine transform of all-position mean cosine. Exclusion changes metric levels but not Pearson, Spearman, or these residualized associations.

| Predictor versus reconstruction target | Pooled Pearson | Pooled Spearman | Equal-track Spearman | Controlled Pearson | Controlled rank association |
|---|---:|---:|---:|---:|---:|
| `V_g` versus magnitude error | 0.702 | 0.638 | 0.622 | 0.420 | 0.286 |
| Amplitude variance versus magnitude error | 0.707 | 0.541 | 0.567 | 0.560 | 0.373 |
| `V_g` versus direction error | -0.122 | -0.324 | -0.444 | 0.109 | 0.431 |
| `D_h` versus direction error | 0.776 | 0.701 | 0.689 | 0.842 | 0.713 |
| Relative vector variance versus direction error | 0.546 | 0.639 | 0.544 | 0.585 | 0.605 |
| Eight-point proxy variance versus direction error | -0.177 | -0.398 | -0.344 | -0.001 | 0.202 |

Direction error is `E[1-cos(g_t, ghat_t)]`, so its correlation signs are the opposite of those for mean cosine. The original positive pooled `V_g` versus cosine Spearman of 0.324 becomes a controlled rank association of -0.431 (equivalently, +0.431 with direction error) after controlling mean gradient norm and track. The eight-proxy variance versus cosine association changes from +0.398 pooled to -0.202 controlled. Direction dispersion is more directly associated with directional reconstruction error than absolute vector variance: pooled Spearman 0.701 versus -0.324, and controlled rank association 0.713 versus 0.431 for the positive error target. Relative vector variance also aligns positively with direction error after scale normalization.

Leave-one-track-out controlled rank ranges are -0.524 to -0.329 for `V_g` versus cosine, -0.349 to -0.057 for proxy variance versus cosine, and 0.617 to 0.756 for `D_h` versus direction error. Their signs persist across all ten omissions, although the proxy association is weak. This is a fixed-effects descriptive sensitivity check, not a causal result or a population-level inference.

The non-anchor mean magnitude error is 1.100 versus 1.083 with anchors; mean direction error is 0.648 versus 0.638. Every block uses 252 of 256 points after exclusion. The positive pooled variance-cosine association is therefore not caused by the four exact anchor positions. Its direction changes after controlling gradient scale and track identity.

## Figures and artifacts

`figures/scatter_all.png` and `figures/scatter_nonanchor.png` show 46 blocks colored by track. Each has raw and controlled-rank panels for full variance versus cosine, proxy variance versus cosine, and full direction dispersion versus direction error. SVG versions are also provided. `block_metrics.csv` contains all 46 derived block rows, `correlations.csv` gives pooled, equal-track, and controlled associations for all requested checks, and `input_manifest.json` records definitions and source paths. The `.complete` marker is present. The selected ten trajectories and overlapping raw windows limit generalization; the regression has only 46 block observations and ten track groups.
EOF'