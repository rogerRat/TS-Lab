# ADS-VIC 仅当前已有任务的阶段图

- 更新时间：2026-11-15 14:52 CST。
- 用户要求立即绘图，只看现有任务；完整矩阵的双 worker 重算已停止，不再继续其他任务。
- 当时 FEV09 与 TIME04 的 q0/q1 矩阵计算已在 worker 内存中完成，但频次直方图尚未写入磁盘；日志仅保留各阶段匹配总数，不能从总数还原 100 桶分布。
- 改用已持久化 `probability.json` 中的 source-anchor maximum cosine，筛选 FEV09、TIME04 两任务，按 q0=32 source anchors 与新增 q1=8 source anchors 分组绘图。分数均是相对于最终 reference anchor 集合的最大余弦；不是逐对矩阵直方图。
- 交互 HTML 和 JSON 已写入 `outputs/selection/v1-rage-ads-v2/ads-vic/anchor-max-cosine/current-fev-09-time-04/`；另生成 SVG 方便静态查看。PNG 导出期间 Kaleido 浏览器启动失败，HTML 与 SVG 已完成。
- 完整矩阵重算脚本已由 tmux Ctrl-C 停止；训练/评估及原有 selection.json、probability.json 未修改。GPU 回到空闲。
