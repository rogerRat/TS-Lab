# 2026-11-14 v1-rage-ads 粗匹配余弦直方图

## 操作记录

- 时间：2026-11-14 15:22 CST
- 实验：v1-rage-ads 粗锚点匹配分布检查
- 操作：只读加载已有梯度缓存与各任务 reference 粗梯度；未重算梯度、未修改 ADS 或训练结果。
- Source 锚点：读取 96 份 uniform 清单中的 83,330 个缓存梯度，全部匹配。
- 每个任务从真实 source-reference 配对中有放回均匀抽样 120,000 对，随机种子 42；对向量归一化后计算余弦。图按任务等量抽样，黄色区域标出 |cos|<=0.1。

## 结果摘要

| 任务 | Reference 粗锚点数 | 抽样中 |cos|<=0.1 |
|---|---:|---:|
| FEV 09 | 168 | 88.47% |
| TIME 01 | 113 | 77.64% |
| TIME 02 | 155 | 78.98% |
| TIME 03 | 365 | 80.56% |
| TIME 04 | 122 | 77.24% |
| TIME 06 | 7,720 | 89.19% |
| TIME 07 | 54 | 82.08% |
| TIME 08 | 3 | 93.71% |

八任务等权均值为 83.49%。按 score-matrix.jsonl 中 coarse_round_1 的完整矩阵配对数加权估计为 88.27%；该值由每任务 120,000 对抽样比例加权，仍是抽样估计。差异主要来自 TIME 06 的 source-reference 配对总数远高于其余任务。

## 产物

- 直方图：outputs/selection/v1-rage-ads/rolling/audit/coarse-score-histogram-by-task.png
- 抽样统计：outputs/selection/v1-rage-ads/rolling/audit/coarse-score-histogram-by-task.json
- 图中每个任务等量抽样，不应用该图的等权均值替代按全矩阵配对数加权结果。

