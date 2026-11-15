# ADS-VIC 已保存 source-anchor 分数阶段图

- 完成时间：2026-11-15 14:25 CST。
- 用户要求先用已有结果绘图。检查发现 `probability.json` 保存了每个最终 source anchor 对最终 reference 集合的最大余弦，但没有逐对 cosine matrix 或 q0-only reference scoring。
- 已绘制 8 任务阶段分布：按 source anchor 属于 q0=32 网格或 q1=8 新增网格分组，统计每个 anchor 的最终最大余弦。此图明确不是逐对矩阵余弦频次。
- 产物：`outputs/selection/v1-rage-ads-v2/ads-vic/anchor-max-cosine/saved-anchor-max-histograms.png`、`.html`、`.json`、`.complete`。
- 完整 q0/q1 逐对矩阵直方图仍由 tmux `ads_vic_hist` 的两个 worker 并行重算；该工作不覆盖已保存的选择、训练和评估数据。
