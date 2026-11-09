# CRB replay experiment results

Last modified: 2026-11-08

Status: terminated by the user on 2026-11-08 because the results were broadly clear and container resources were constrained. The ten-trajectory replay plan was not completed.

Protocol and full planned parameter details: `../EXP.md` under the history entry `v0-crb-reconst-test`. At termination, the run had 429 committed CRB configuration attempts, 326 unique candidate metric files, and 28 uniform baselines. The 326 metric files are stored as compressed per-window metric payloads. All ten BF16 full-gradient trajectory caches (about 201 GiB on disk) remain at `outputs/train/v0-crb-reconst-test/gradients/`.

## Batch status and observed results

The q cells report committed CRB configurations out of 16. `L2` and `cosine error` compare each configuration's median per-window error with its matched-budget uniform baseline. Counts are configuration attempts, not independent observations.

| Trajectory batch | CRB attempts | Unique metric files | Uniform baselines | q coverage | L2 better / tie / worse | Cosine error better / tie / worse |
|---:|---:|---:|---:|---|---:|---:|
| 00 | 128 | 79 | 8 | q=1,4,16,64,128,256,512,576: 16/16 each | 13 / 64 / 51 | 7 / 64 / 57 |
| 01 | 128 | 101 | 8 | q=1,4,16,64,128,256,512,576: 16/16 each | 3 / 16 / 109 | 5 / 16 / 107 |
| 02 | 128 | 101 | 8 | q=1,4,16,64,128,256,512,576: 16/16 each | 50 / 16 / 62 | 14 / 48 / 66 |
| 03 | 29 | 29 | 2 | q=1: 16/16; q=4: 13/16 | 1 / 16 / 12 | 0 / 16 / 13 |
| 04 | 16 | 16 | 2 | q=1: 16/16 | 0 / 16 / 0 | 0 / 16 / 0 |
| 05 | 0 | 0 | 0 | not started | - | - |
| 06 | 0 | 0 | 0 | not started | - | - |
| 07 | 0 | 0 | 0 | not started | - | - |
| 08 | 0 | 0 | 0 | not started | - | - |
| 09 | 0 | 0 | 0 | not started | - | - |
| **Total** | **429** | **326** | **28** | **3 batches complete; 2 partial; 5 not started** | **67 / 128 / 234** | **26 / 160 / 243** |

The tie counts include exact q=1 cases and configurations that reuse an already evaluated integer budget. The q=1 exact results are identities by design. Differences are descriptive, from the finite observed portion, and do not support a complete ten-trajectory conclusion.

## Regenerated visualization

The final snapshot was regenerated from committed metric files after termination. It records 429 attempts, 326 unique metric files, and 28 baselines. The overview and aggregate q-sweep use all available trajectory batches; the simplex and per-window panels use trajectory 00, q=4, S00.

- [Overview PNG](../../archive/v0-crb-reconst-test/visualization/preview_20261108_024904_utc/figure_1_overview.png)
- [Simplex PNG](../../archive/v0-crb-reconst-test/visualization/preview_20261108_024904_utc/figure_2_simplex_map.png)
- [Per-window profile PNG](../../archive/v0-crb-reconst-test/visualization/preview_20261108_024904_utc/figure_3_window_profile.png)
- [Aggregate q-sweep PNG](../../archive/v0-crb-reconst-test/visualization/preview_20261108_024904_utc/figure_4_q_sweep_trajectory_mean.png)
- [Snapshot manifest and CSV summaries](../../archive/v0-crb-reconst-test/visualization/preview_20261108_024904_utc/manifest.json)
- [Archived raw batches](../../archive/v0-crb-reconst-test/batches/)

The three earlier plot snapshots were removed from the active output tree after this archived snapshot was verified. The run output and the full-gradient caches were not removed.
