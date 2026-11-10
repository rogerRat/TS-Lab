# Trev seed 0 final results

Last modified: 2026-11-10

All six Chronos, TimerXL, and TimesFM phases completed 4,000 BF16 LoRA optimizer updates and eight FEV/TIME checkpoint evaluations. The base weights were frozen in both phases; `full` means the full eligible data pool. MASE is the only metric used here. Lower is better.

## FEV MASE at step 4000

| Model | Pool | Hospital | M_DENSE | US consumption | World CO2 |
|---|---|---:|---:|---:|---:|
| Chronos | rand | 0.6776 | 1.4715 | 8.5522 | 3.7926 |
| Chronos | full | 0.6718 | 1.4705 | 8.6321 | 3.7830 |
| TimerXL | rand | 0.6672 | 2.0354 | 13.4942 | 5.7754 |
| TimerXL | full | 0.6675 | 2.0330 | 12.5499 | 5.2563 |
| TimesFM | rand | 0.6547 | 1.3943 | 8.8224 | 3.9751 |
| TimesFM | full | 0.6646 | 1.4042 | 8.8852 | 3.9794 |

## TIME MASE at step 4000

Task-level MASE is the arithmetic mean of the saved MASE array over series, windows, and variates for that task. No cross-task average is reported.

| Model | Pool | CPHL/15T | CPHL/30T | CPHL/H | ECDC/D | ECDC/W | epf/H | Import/M | Production/M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Chronos | rand | 1.3327 | 1.0446 | 1.7352 | 1.5273 | 1.1765 | 1.4267 | 1.9883 | 1.0170 |
| Chronos | full | 1.3407 | 1.0513 | 1.6985 | 1.5091 | 1.1744 | 1.4074 | 1.9957 | 1.0297 |
| TimerXL | rand | 0.6788 | 1.1598 | 2.0207 | 5.9848 | 1.8735 | 1.1044 | 2.5787 | 1.2820 |
| TimerXL | full | 0.7964 | 1.0559 | 1.9565 | 5.6738 | 2.0056 | 1.2530 | 2.4763 | 1.2334 |
| TimesFM | rand | 1.1080 | 0.8075 | 1.5769 | 2.6093 | 1.1632 | 1.0085 | 2.0019 | 1.0013 |
| TimesFM | full | 1.1167 | 0.8198 | 1.5936 | 2.4061 | 1.1735 | 1.0536 | 1.9866 | 1.0113 |

All values come from the corresponding `outputs/eval/<batch>/step-4000/` artifacts. These are one-seed descriptive results. The stopped seed 1 random Chronos run was incomplete and its temporary outputs were removed; no seed 1-3 result is reported.
