# 2026-11-13 v1-crb-redo RR 平均收益与对齐相关

- 批次：sgwdev（single-window-eval）。
- 结果变量：四个完整 RR arm 的任务 MASE 均值相对 full/rand01 较强基线的任务相对收益；n=8。
- P99 定义：每任务 CRB 与 Uniform 安全历史锚点—真实测试样本两两余弦分布 P99 的均值。
- 结果：四 arm 平均 MASE 在 3/8 任务胜过两条基线，任务等权相对收益 -5.57%。P99 与收益 Pearson r=-0.554、Spearman rho=-0.595；历史—测试平均 cos 与收益 Pearson r=-0.043、Spearman rho=-0.214。
- 边界：8 个任务、单 seed；相关仅为描述性探索，不作因果或稳定性判断。
- 文档：更新 docs/exp/v1-crb-redo/exp.md、results.md 和 docs/RESULTS.md。
