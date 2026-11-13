# TimerXL seed 0 Trev 结果

作为正式 v1-rage 答案已于 2026-11-10 废弃：源数据选择不合适。历史结果仅作为可能的参考保留。

最后修改日期：2026-11-10

random-10% 与 full-pool 两个阶段均完成了 4,000 次 BF16 LoRA 优化器更新。`Full` 仅指合格数据池；两个阶段基础模型均冻结，且两个阶段均未使用全参数微调。每个阶段都有全部八个固定检查点评估，每个检查点包含四个 FEV 任务与八个 TIME 任务。源产物位于 `outputs/train/v1-rage-timerXL-trev-84m-{rand,full}/` 与 `outputs/eval/v1-rage-timerXL-trev-84m-{rand,full}/` 下。

| Phase | Step | Hospital MASE | M_DENSE MASE | US consumption MASE | World CO2 MASE |
|---|---:|---:|---:|---:|---:|
| Random 10% | 0 | 0.6688 | 2.0642 | 18.2748 | 7.2453 |
| Random 10% | 4000 | 0.6672 | 2.0354 | 13.4942 | 5.7754 |
| Full pool | 0 | 0.6688 | 2.0642 | 18.2748 | 7.2453 |
| Full pool | 4000 | 0.6675 | 2.0330 | 12.5499 | 5.2563 |

这些数值直接来自对应阶段与 step 的 `fev_results.csv`。在本 Trev 实验中，MASE 是用于检查、比较与报告的唯一指标。MASE 越低越好。在四个 FEV 任务中，full-pool 端点在三个任务上低于 random-10% 端点，住院人数几乎持平。这些观察仅描述 seed 0；没有可用的跨 seed 结论。每个检查点都存在全部八个 TIME 任务输出与完成元数据，但此处未汇总 TIME 指标。

## v1-rage-trev TimerXL seed 0

最后修改日期：2026-11-10

在 Chronos、TimerXL 与 TimesFM 的 Trev 批次中，MASE 是用于检查、比较与报告的唯一指标。全部 Chronos、TimerXL 与 TimesFM 的 Trev rand/full 阶段均使用 LoRA 并冻结基础权重；`full` 指完整的合格数据池，而非全参数训练。TimerXL 84M seed 0 的 random 与 full 阶段完成了 4,000 次更新以及全部八个 FEV/TIME 检查点评估。在 step 4000 时，住院人数、M_DENSE、美国消费量与世界 CO2 的 FEV MASE 分别为 random 的 0.6672/2.0354/13.4942/5.7754 与 full 的 0.6675/2.0330/12.5499/5.2563。源 CSV、任务覆盖与限制见 docs/results/v1-rage-trev-timerxl-seed0.md。
