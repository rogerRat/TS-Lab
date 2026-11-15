# 2026-11-13 v1-crb-redo sgwdev 梯度余弦尾部统计

- 检查时间：2026-11-13 19:32 CST
- 实验与批次：v1-crb-redo / sgwdev（single-window-eval）
- 状态：基于既有安全历史梯度与真实测试梯度缓存完成尾部统计；未重算任何梯度。
- 计算：对每个任务、方法的全部安全历史—测试梯度两两余弦计算 P99 与最大值；另按每个测试样本取安全历史最大余弦，汇总其均值和最大值。
- 结果：两两配对 P99 范围 0.096093–0.388612，配对最大值范围 0.098758–0.619248。详见 docs/exp/v1-crb-redo/results.md 和 outputs/selection/v1-crb-redo/sgwdev/metrics/history_test_alignment_tail.jsonl。
- 注意：每任务真实测试样本数为 2–9；测试样本最佳匹配分布的 P99 不稳定，文档报告其均值和最大值。
- 修改：在 exp.md 中补充尾部统计口径，在 results.md 中增加逐任务逐方法数值，并更新全局 RESULTS.md 摘要。
