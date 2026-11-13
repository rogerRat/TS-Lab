# TimesFM seed 0 Trev 结果

作为正式 v1-rage 答案已于 2026-11-10 废弃：源数据选择不合适。历史结果仅作为可能的参考保留。

最后修改日期：2026-11-10

random-10% 与 full-pool 两个阶段均完成了 4,000 次 BF16 LoRA 优化器更新，基础权重冻结。`Full` 指完整的合格数据池。每个阶段均完成了全部八个固定检查点评估，每个检查点包含四个 FEV 与八个 TIME 任务。MASE 是用于比较与报告的唯一指标。

| Phase | Step | Hospital MASE | M_DENSE MASE | US consumption MASE | World CO2 MASE |
|---|---:|---:|---:|---:|---:|
| Random 10% | 0 | 0.6566 | 1.3960 | 8.8114 | 4.0807 |
| Random 10% | 4000 | 0.6547 | 1.3943 | 8.8224 | 3.9751 |
| Full pool | 0 | 0.6566 | 1.3960 | 8.8114 | 4.0807 |
| Full pool | 4000 | 0.6646 | 1.4042 | 8.8852 | 3.9794 |

数值取自各阶段与各 step 的 `fev_results.csv`，位于 `outputs/eval/v1-rage-timesFM-trev-2.5-{rand,full}/` 下；阶段完成标记位于 `outputs/train/` 下。MASE 越低越好。在 step 4000 时，random 阶段在全部四个 FEV 任务上均低于 full 阶段，但这只是单 seed 的观察。八个 TIME 任务在每个检查点均已完成；其 MASE 值保留在任务级指标文件中，此处未做汇总。

## v1-rage-trev TimesFM seed 0

最后修改日期：2026-11-10

TimesFM 2.5 seed 0 的 random 与 full 阶段完成了 4,000 次 BF16 LoRA 更新以及全部八个 FEV/TIME 检查点评估。在 step 4000 时，住院人数、M_DENSE、美国消费量与世界 CO2 的 FEV MASE 分别为 random 的 0.6547/1.3943/8.8224/3.9751 与 full 的 0.6646/1.4042/8.8852/3.9794。这是单个 seed；详情与源文件见 docs/results/v1-rage-trev-timesfm-seed0.md。
