# v1-rage-ads-v2 概率支持与算法文档更新

- 时间：2026-11-15 02:00 CST
- 选择：8/8 完成；训练和 rolling 评估：40/40 完成；采样支持审计：40/40 行完成。
- 代码确认：ADS-VIC/uni 的 anchor score 使用全 reference 的最大余弦，负值截零，再按 h=4 有限邻域线性衰减并按 max 合并；非全零总权重不加 epsilon，窗口邻域外或权重为零的窗口不可抽中；全零质量才均匀回退。traj 的 source 轨迹分数取全部 reference 轨迹余弦最大值并截零，零分轨迹概率为零；只有所有轨迹权重全零才均匀回退。orig 不训练；reft 在 reference 窗口上均匀采样。
- 审计汇总：ADS-VIC/uni/traj/orig 的零概率源窗口比例分别为 30.3042%、19.7117%、17.5960%、0%；reft 的 source 口径不适用。逐任务计数已写入 docs/exp/v1-rage-ads-v2/results.md。
- 修改文档：更新根目录 ALGO.md、v2 exp.md、results.md 和全局 docs/EXP.md、docs/RESULTS.md；明确“均匀锚点”不等于“均匀窗口概率”，也明确轨迹 ID 列入候选不保证其概率为正。
- 训练/评估完成标记：outputs/train/v1-rage-ads-v2/.complete 与 outputs/eval/v1-rage-ads-v2/.complete。
