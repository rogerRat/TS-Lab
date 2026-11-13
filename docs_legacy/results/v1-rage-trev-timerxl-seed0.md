# TimerXL seed 0 Trev results

Deprecated as a formal v1-rage answer (2026-11-10): the source data selection was unsuitable. Historical results are retained only as possible references.

Last modified: 2026-11-10

Both random-10% and full-pool phases completed 4,000 BF16 LoRA optimizer updates. `Full` refers only to the eligible data pool; the base model was frozen in both phases, and neither phase used full-parameter fine-tuning. Each phase has all eight fixed checkpoint evaluations, with four FEV tasks and eight TIME tasks at every checkpoint. The source artifacts are under `outputs/train/v1-rage-timerXL-trev-84m-{rand,full}/` and `outputs/eval/v1-rage-timerXL-trev-84m-{rand,full}/`.

| Phase | Step | Hospital MASE | M_DENSE MASE | US consumption MASE | World CO2 MASE |
|---|---:|---:|---:|---:|---:|
| Random 10% | 0 | 0.6688 | 2.0642 | 18.2748 | 7.2453 |
| Random 10% | 4000 | 0.6672 | 2.0354 | 13.4942 | 5.7754 |
| Full pool | 0 | 0.6688 | 2.0642 | 18.2748 | 7.2453 |
| Full pool | 4000 | 0.6675 | 2.0330 | 12.5499 | 5.2563 |

These values come directly from `fev_results.csv` for the corresponding phase and step. MASE is the sole metric used for inspection, comparison, and reporting in this Trev experiment. Lower MASE is better. The full-pool endpoint is lower on three of four FEV tasks than the random-10% endpoint, with hospital admissions nearly tied. These observations describe seed 0 only; no across-seed conclusion is available. All eight TIME task outputs and completion metadata are present at each checkpoint, but TIME metrics are not summarized here.
