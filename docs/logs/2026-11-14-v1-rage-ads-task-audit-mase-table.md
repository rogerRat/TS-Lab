# v1-rage-ads 任务级审计与 MASE 合并表

- 操作时间：2026-11-14 CST。
- 实验代号 / 批次：`v1-rage-ads` / `rolling-eight-task`。
- 用户要求：以任务为行整理八任务审计，并与 MASE 结果合成一张总表；每行最低 MASE 加粗并标星；解释轨迹级筛选与 ADS 的任务级差异。
- 文档更新：在 `docs/exp/v1-rage-ads/results.md` 加入八行合并总表，含 source q32 对齐及分位数、四臂抽样 best-reference 对齐及分位数、reference 连续性、score-matrix 稀疏度与分位数、source P99 聚集和四臂 MASE。完整九个分位点继续保存在逐任务 JSONL 中。
- 结果核对：Traj 在 FEV 9 与 TIME 3 的 MASE 最低，胜出2/8；rage最低4/8，ancUni和fullRand各1/8。八任务算术平均 MASE：rage 1.4637、Traj 1.4905，rage低约1.8%。因此“ADS总体不如Traj”不成立，但Traj确有任务级胜出。
- 反直觉观察：Traj 的八任务平均 best-reference 对齐最高（0.2148），平均 MASE 也最高（1.4905）；梯度对齐与下游预测误差不等价。任务分布差异、轨迹内均匀采样的覆盖效应及 ADS 对 reference 锚点密度的响应均列为待验证解释，不作因果结论。
- 表格约定：MASE越低越好；每行最小值以粗体及 `*` 标记。对齐和连续性摘要显示均值与 p05/p50/p95，完整分位数、矩阵逐阶段结果及辅助评估指标保留于原始 JSONL/NPZ/CSV。
