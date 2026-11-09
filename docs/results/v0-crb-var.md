# v0-crb-var results

Last modified: 2026-11-09

Status: completed. Ten fixed trajectories were replayed from the retained BF16 full-gradient caches. No new gradients were computed.

## Protocol

For each trajectory, the gradient budget was ceil(number of legal starts / 64). Native CRB and Uniform used the same anchor count, including both endpoints. The CRB allocation used two endpoint probes and distributed discretionary probes as evenly as possible across coverage, interpolation-risk, and recursive minimum-kappa stages. All block metrics use every cached full-gradient vector. Relative variance is within-block vector variance divided by mean squared gradient norm. Direction dispersion is the mean of 1 minus cosine similarity to the block mean direction.

## Results

The ten-trajectory equal-track mean relative variance was 0.6410 for CRB and 0.6802 for Uniform; CRB was lower on 10/10 tracks. Mean direction dispersion was 0.4051 for CRB and 0.4402 for Uniform; CRB was lower on 10/10 tracks. Window-weighted direction dispersion was 0.3944 versus 0.4135, lower for CRB on 9/10 tracks. Absolute mean block variance was 10.1357 for CRB versus 9.8196 for Uniform, with CRB lower on 7/10 tracks; this scale-dependent measure did not consistently favor CRB.

Block size was positively associated with local relative variance and direction dispersion under CRB (Spearman rho 0.6027 and 0.5830). Under Uniform, the corresponding correlations were 0.0876 and 0.0602. The positive association means larger CRB blocks tend to retain greater within-block dispersion. Thus CRB did not preferentially place its shorter intervals in the regions with the highest measured raw local dispersion; the observed size-dispersion relationship is opposite to that proposed. This coexists with lower average relative variance and direction dispersion across the CRB blocks. Within each track, block-size Spearman rho averaged 0.5413 for relative variance and 0.5286 for direction dispersion, positive on all 10 tracks. Uniform within-track means were 0.0371 and 0.0114. Blocks within trajectories are dependent, so these correlations are descriptive and no inferential claim is made.

| Trajectory | Starts | Budget | CRB relative variance | Uniform relative variance | CRB direction dispersion | Uniform direction dispersion |
|---:|---:|---:|---:|---:|---:|---:|
| 00 | 607 | 10 | 0.7047 | 0.7163 | 0.4513 | 0.4689 |
| 01 | 1388 | 22 | 0.6301 | 0.7280 | 0.4110 | 0.4761 |
| 02 | 1310 | 21 | 0.6718 | 0.7010 | 0.4483 | 0.4713 |
| 03 | 1383 | 22 | 0.5970 | 0.6168 | 0.3502 | 0.3808 |
| 04 | 1320 | 21 | 0.5369 | 0.5902 | 0.3197 | 0.3662 |
| 05 | 1393 | 22 | 0.6550 | 0.6688 | 0.4160 | 0.4221 |
| 06 | 1393 | 22 | 0.6415 | 0.6588 | 0.3945 | 0.4152 |
| 07 | 1270 | 20 | 0.6478 | 0.7047 | 0.4130 | 0.4586 |
| 08 | 1378 | 22 | 0.6407 | 0.7038 | 0.3995 | 0.4507 |
| 09 | 1394 | 22 | 0.6846 | 0.7137 | 0.4478 | 0.4925 |

## Artifacts

- Per-block metrics: outputs/train/v0-crb-var/block_metrics.csv
- Per-trajectory summaries: outputs/train/v0-crb-var/trajectory_summary.json
- Completion marker: outputs/train/v0-crb-var/.complete
- Config: configs/rage/v0-crb-var.json
- Replay entry point: scripts/temp/v0_crb_var.py
