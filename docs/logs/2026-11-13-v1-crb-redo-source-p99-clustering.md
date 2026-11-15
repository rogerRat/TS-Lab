# 2026-11-13 v1-crb-redo src P99 聚集性分析

- 批次：sgwdev（single-window-eval）。
- 数据：缓存 source/reference unit directions；未重算梯度。
- 分数：每个源 trajectory/block unit 对该任务所有 reference units 的最大余弦；组内按分数排序取 top 1%。
- Block 结果：每组 833 个 P99 block，分布于 18–53 条轨迹；top 10 轨迹占 60.7%–98.9%，有效轨迹数 4.6–20.6。
- 时间聚集：相邻高分 block 对 178–571；按每轨迹固定高分 block 数进行 500 次随机重排，期望均值 33.5–244.9；全部 16 个 task-method 组合的经验 p=0.002。最长连续高分 block run 为 5–39。
- Traj 颗粒度每组 96 条轨迹的 top 1% 仅 1 条，报告 best match 轨迹，不推断其轨迹内部连续性。
- 代码：scripts/exp/v1-crb-redo/analyze_source_p99_clustering.py；结果：outputs/selection/v1-crb-redo/sgwdev/source_p99_clustering/。
- 文档：逐任务结果及定义见 docs/exp/v1-crb-redo/results.md 和 exp.md；全局索引已更新。
