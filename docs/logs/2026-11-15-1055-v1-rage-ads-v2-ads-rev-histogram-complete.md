# ads-rev 余弦频次直方图完成

- 完成时间：2026-11-15 10:54 CST。
- 实验代号与批次：`v1-rage-ads-v2` / `ads-rev-reverse-complement`。
- q0=32 粗筛和 q1=16 精算阶段均覆盖八任务。精算矩阵包括原 q0 锚点和反向加密新锚点。
- 每任务 100 个等宽余弦桶，范围 [-1,1]；图表使用 log 频次轴并标出 cos=0.1。已核对每个阶段直方图计数之和等于该阶段矩阵元素数。
- 复算 `cos>0.1` 配对总数：q0 98,792,453；q1 98,904,744。q1 与原 selection.json 总数相差 1 对，差异集中在 FEV09，原因是阈值附近复算浮点差异；原选择记录、训练与评估产物未修改。
- 产物：`outputs/selection/v1-rage-ads-v2/ads-rev/cosine-histograms.png`、`cosine-histograms.html`、`cosine-histograms.json`；完成标记 `.complete`。图表执行日志：`logs/v1-rage-ads-v2/ads-rev-cosine-histograms_20261115095710.log`。
- 选择结果与 MASE 总表路径：`docs/exp/v1-rage-ads-v2/results.md`。
