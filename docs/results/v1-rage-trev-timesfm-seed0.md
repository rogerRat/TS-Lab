# TimesFM seed 0 Trev results

Last modified: 2026-11-10

Both random-10% and full-pool phases completed 4,000 BF16 LoRA optimizer updates with frozen base weights. `Full` refers to the full eligible data pool. Each phase completed all eight fixed checkpoint evaluations, with four FEV and eight TIME tasks at each checkpoint. MASE is the sole metric used for comparison and reporting.

| Phase | Step | Hospital MASE | M_DENSE MASE | US consumption MASE | World CO2 MASE |
|---|---:|---:|---:|---:|---:|
| Random 10% | 0 | 0.6566 | 1.3960 | 8.8114 | 4.0807 |
| Random 10% | 4000 | 0.6547 | 1.3943 | 8.8224 | 3.9751 |
| Full pool | 0 | 0.6566 | 1.3960 | 8.8114 | 4.0807 |
| Full pool | 4000 | 0.6646 | 1.4042 | 8.8852 | 3.9794 |

Values are from each phase and step `fev_results.csv` under `outputs/eval/v1-rage-timesFM-trev-2.5-{rand,full}/`; phase completion markers are under `outputs/train/`. Lower MASE is better. At step 4000, the random phase is lower than the full phase on all four FEV tasks, but this is a single-seed observation. The eight TIME tasks are complete at each checkpoint; their MASE values are retained in task-level metrics files but are not aggregated here.
