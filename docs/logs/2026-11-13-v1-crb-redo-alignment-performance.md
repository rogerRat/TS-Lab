# 2026-11-13 v1-crb-redo 对齐与下游表现关系分析

- 检查时间：2026-11-13 19:40 CST。
- 批次：sgwdev（single-window-eval）。
- 依据：results.md 中八任务 MASE；四个完整 RR arm；全源、top10% 和安全历史—测试梯度对齐缓存结果。
- 观察：四 arm 平均 MASE 在 3/8 任务胜过较强基线；14/32 个 arm-task 同时胜过 full 与 rand01。按任务等权相对收益均值 -5.57%。
- 描述性关联：Spearman rho 对全源对齐 -0.524、top10% 对齐 -0.548、安全历史—测试对齐 -0.214；仅八任务且单 seed，不作因果或显著性结论。
- 文档：逐任务对照和边界说明已加入 docs/exp/v1-crb-redo/results.md；全局索引 docs/RESULTS.md 已更新。
