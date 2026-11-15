# 2026-11-13 v1-crb-redo sgwdev 相似度分布补充

- 检查时间：2026-11-13 19:34 CST。
- 实验与批次：v1-crb-redo / sgwdev（single-window-eval）。
- 状态：完成 16 组历史—测试梯度配对分布统计；全程复用已保存梯度。
- 统计项：每组全部两两余弦的 P50、P90、P95、P99、最大值、正余弦比例和 cos>=0.1 比例；另按真实测试样本统计最佳安全历史匹配的均值和最大值。
- 结果范围：P50 -0.022–0.028；P90 0.072–0.201；P95 0.085–0.257；P99 0.096–0.389；正余弦比例 33.3%–59.1%。
- 产物：outputs/selection/v1-crb-redo/sgwdev/metrics/history_test_alignment_distribution.jsonl；逐任务表见 docs/exp/v1-crb-redo/results.md。
- 文档：exp.md 已规定新增相似度分析需报告分布，并预先说明大规模配对使用全量分位数或可复现抽样估计的口径、样本量与种子。
