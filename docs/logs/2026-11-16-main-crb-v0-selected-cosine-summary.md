# main-crb-v0 入选 top-10% 余弦分数抽样重算

- 操作时间：2026-11-16 18:46 CST。
- 对 8 项任务的 CRB、UNI 入选集合各无放回抽样 4,096 个窗口，按原 source block direction 插值、BF16 cosine 矩阵乘和 task-specific ref blocks 最大值逻辑重算。
- 样本种子基准 20261116，按 task/method 偏移。完成 16 组；报告 JSON 和 .complete 位于 outputs/selection/main-crb-v0/q16-src256-eight-task/cosine-summary/。
- 结果包含均值、标准差、分位数和全量选择时阈值。抽样最小值偶有比原阈值低一个或两个 BF16 量化档，重算矩阵分块与初次全量评分不同；已在结果文档注明为近似估计。
- 实现脚本 scripts/exp/main-crb-v0/summarize_selected_cosines.py；运行日志 logs/main-crb-v0/q16-src256-eight-task_20261116184449.log。
- 结果已追加至 docs/exp/main-crb-v0/results.md。未重跑全 source 池，也未改变训练或筛选产物。
