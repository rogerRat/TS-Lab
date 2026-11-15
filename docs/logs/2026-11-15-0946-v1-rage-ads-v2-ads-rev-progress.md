# v1-rage-ads-v2 ads-rev 训练进度更新

- 检查时间：2026-11-15 09:46 CST
- 批次：ads-rev-reverse-complement
- 训练：8 个单元全部活跃；FEV09/TIME01/TIME03/TIME07/TIME08 为 1,900/4,000 步，TIME02/TIME04/TIME06 为 2,000/4,000 步。0 个单元报错，训练评估完成标记仍为 0/8。
- 资源：GPU 利用率 99%，显存约 7,756 MiB；容器内存约 19.85/66.57 GB；OOM/OOM-kill 均为 0。
- 结果：选择矩阵匹配统计已写入 `docs/exp/v1-rage-ads-v2/results.md`；模型 MASE 尚未生成，训练后还需模型导出和 rolling 评估。
- 运行会话：`v1_rage_ads_v2_ads_rev`；训练 runner PID 46209，8 个 task worker 保持运行。
- 训练日志：`logs/v1-rage-ads-v2/parallel-train-eval_20261115094232.log` 与 8 个 `train-eval-<task>-ads-rev_20261115094232.log`。
