# v1-crb-redo：TimerXL 两基线结果

最后修改日期：2026-11-12 23:38 CST。

## 完成情况与数据来源

两个批次各完成 4,000 次优化更新；各自的训练目录、FEV 四任务及 TIME 八任务目录均有完成标记。full 使用 src-96 全部 2,666,312 个合法窗口；rand01 以 seed 0 固定抽取其中 266,631 个合法窗口。两批各累计抽样 128,000 次窗口曝光；随机批次可以重复抽到同一固定窗口。模型均从原始 TimerXL 预训练权重独立开始，均为单一 seed 0。

原始指标文件：`outputs/eval/v1-crb-redo/<批次>/fev/fev_results.csv` 与 `outputs/eval/v1-crb-redo/<批次>/time/<数据集>/<频率>/short/metrics.npz`。下表 FEV MASE 直接取 CSV 的任务行；TIME MASE 对各任务 `metrics.npz` 的 `MASE` 数组在序列、窗口和变量轴求算术均值。差值为 rand01 减 full，正值表示 full 误差更低。

## 第 4,000 步下游评估

| 基准 | 任务 | full MASE | rand01 MASE | rand01 - full |
|---|---|---:|---:|---:|
| FEV | hospital_admissions_1D | 0.667800 | 0.668297 | +0.000497 |
| FEV | M_DENSE_1D | 2.090097 | 2.090301 | +0.000204 |
| FEV | us_consumption_1Y | 16.572460 | 17.244258 | +0.671797 |
| FEV | world_co2_emissions | 6.459984 | 6.811539 | +0.351555 |
| TIME | CPHL/15T short | 1.105670 | 1.120716 | +0.015046 |
| TIME | CPHL/30T short | 0.834222 | 0.817478 | -0.016744 |
| TIME | CPHL/H short | 1.536111 | 1.552243 | +0.016132 |
| TIME | ECDC_COVID/D short | 3.154493 | 3.235430 | +0.080937 |
| TIME | ECDC_COVID/W short | 2.419417 | 2.489539 | +0.070123 |
| TIME | epf_electricity_price/H short | 1.375741 | 1.382876 | +0.007135 |
| TIME | Commodity_Import/M short | 1.997317 | 2.041115 | +0.043798 |
| TIME | Commodity_Production/M short | 1.257721 | 1.256141 | -0.001580 |

FEV 四任务等权平均 MASE：full 6.447586，rand01 6.703599；TIME 八任务等权平均 MASE：full 1.710087，rand01 1.736942。full 在 10 项、rand01 在 2 项的 MASE 更低。两个基准的均值分别计算，没有跨 FEV 与 TIME 混合求总均值；任务尺度不同，FEV 均值受高 MASE 任务影响较大。单一 seed 和这 12 项任务不足以估计统计显著性。

训练清单：`outputs/train/v1-crb-redo/<批次>/training_manifest.json`；随机窗口索引：`outputs/train/v1-crb-redo/timerXL-baseline-rand01/fixed_window_indices.npy`。FEV 本地数据源修订 `d3e7e720f4158eab03ebcb021f1cce779cd92acc`；TIME 数据源修订 `83e3d0b3be28d11c7182bffcc1892d19b36c4da1`。
