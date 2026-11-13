# Trev seed 0 最终结果

作为正式 v1-rage 答案已于 2026-11-10 废弃：源数据选择不合适。历史结果仅作为可能的参考保留。

最后修改日期：2026-11-10

全部六个 Chronos、TimerXL 与 TimesFM 阶段均完成了 4,000 次 BF16 LoRA 优化器更新以及八个 FEV/TIME 检查点评估。两个阶段的基础权重均冻结；`full` 指完整的合格数据池。MASE 是此处唯一使用的指标。越低越好。

## step 4000 时的 FEV MASE

| Model | Pool | Hospital | M_DENSE | US consumption | World CO2 |
|---|---|---:|---:|---:|---:|
| Chronos | rand | 0.6776 | 1.4715 | 8.5522 | 3.7926 |
| Chronos | full | 0.6718 | 1.4705 | 8.6321 | 3.7830 |
| TimerXL | rand | 0.6672 | 2.0354 | 13.4942 | 5.7754 |
| TimerXL | full | 0.6675 | 2.0330 | 12.5499 | 5.2563 |
| TimesFM | rand | 0.6547 | 1.3943 | 8.8224 | 3.9751 |
| TimesFM | full | 0.6646 | 1.4042 | 8.8852 | 3.9794 |

## step 4000 时的 TIME MASE

任务级 MASE 是该任务在序列、窗口与变量维度上保存的 MASE 数组的算术平均。不报告跨任务平均值。

| Model | Pool | CPHL/15T | CPHL/30T | CPHL/H | ECDC/D | ECDC/W | epf/H | Import/M | Production/M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Chronos | rand | 1.3327 | 1.0446 | 1.7352 | 1.5273 | 1.1765 | 1.4267 | 1.9883 | 1.0170 |
| Chronos | full | 1.3407 | 1.0513 | 1.6985 | 1.5091 | 1.1744 | 1.4074 | 1.9957 | 1.0297 |
| TimerXL | rand | 0.6788 | 1.1598 | 2.0207 | 5.9848 | 1.8735 | 1.1044 | 2.5787 | 1.2820 |
| TimerXL | full | 0.7964 | 1.0559 | 1.9565 | 5.6738 | 2.0056 | 1.2530 | 2.4763 | 1.2334 |
| TimesFM | rand | 1.1080 | 0.8075 | 1.5769 | 2.6093 | 1.1632 | 1.0085 | 2.0019 | 1.0013 |
| TimesFM | full | 1.1167 | 0.8198 | 1.5936 | 2.4061 | 1.1735 | 1.0536 | 1.9866 | 1.0113 |

所有数值均来自对应的 `outputs/eval/<batch>/step-4000/` 产物。这些是单 seed 的描述性结果。被停止的 seed 1 random Chronos 运行不完整，其临时输出已被删除；不报告 seed 1-3 的结果。

## v1-rage-trev seed 0 最终范围

最后修改日期：2026-11-10

应要求，该研究以一个 seed 收尾。全部六个 Chronos、TimerXL 与 TimesFM seed 0 的 rand/full LoRA 阶段均完成了 4,000 次更新以及八个 FEV/TIME 检查点评估。最终比较仅使用 MASE。不完整的 Chronos seed 1 运行已被停止，其部分输出已被删除；不报告 seed 1-3 的结果。最终 FEV 与 TIME MASE 表格及源路径见 docs/exp/v1-rage-trev/v1-rage-trev-seed0-final.md。


## v1-rage 结果状态更新

最后修改日期：2026-11-10。此前所有 v1-rage 批次均已废弃，因为其源数据选择不合适。仅将结果文件及上述历史数值作为可能的参考保留；其中任何一个都不是正式答案。替代的源数据收集为 src-8192，在 configs/data/src-8192.yaml 中包含 8,192 条固定轨迹。
