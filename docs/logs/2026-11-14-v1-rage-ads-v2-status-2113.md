# v1-rage-ads-v2 阶段进度与预算解读

- 检查时间：2026-11-14 21:13 CST
- 实验代号：v1-rage-ads-v2
- 批次：rolling-baseline-v2
- 状态：选择完成 5/8；TIME04 与 TIME06 正在运行，TIME07 等待下一批；训练/评估尚未开始。

## 已完成任务的选择结果

| 任务 | ADS source 锚点预算 | ADS reference 锚点预算 | 末轮比较配对数 | 选择耗时（分钟） |
|---|---:|---:|---:|---:|
| FEV09 | 783,764 | 744 | 583,120,416 | 56.0 |
| TIME01 | 830,581 | 458 | 380,406,098 | 57.4 |
| TIME02 | 893,872 | 634 | 566,714,848 | 60.8 |
| TIME03 | 967,873 | 1,727 | 1,671,516,671 | 71.1 |
| TIME08 | 205,819 | 4 | 823,276 | 45.0 |

## 解读边界

- 锚点预算表示 ADS-VIC 的选择计算量；配对数表示末轮 source/reference 锚点矩阵的比较规模。它们都不是预测误差或模型效果指标。
- TIME03 的比较矩阵最大，TIME08 最小，说明各任务的 source/reference 规模和阈值命中区域差异明显；这会影响选择计算成本，不能据此推断哪种抽样或训练方法效果最好。
- TIME08 只有 1 条 reference 轨迹，粗层 3 个 reference 锚点，最终 reference 预算为 4。该任务的 reference 证据较少，后续模型指标应按这一数据规模限制解读。
- 当前尚无 v2 训练 checkpoint 或 rolling 评测文件，因此没有可报告的模型质量结论。

## 运行资源

- 21:13 cgroup 内存约 39.0/66.6 GB，GPU 利用率 97%、显存 1,958 MiB；`oom=0`、`oom_kill=0`。
- 状态与过程日志：`logs/v1-rage-ads-v2/progress-poll.jsonl`、各任务选择日志及 `docs/logs/2026-11-14-v1-rage-ads-v2-container-restart.md`。
