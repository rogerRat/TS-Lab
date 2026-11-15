# ads-rev 直方图重算启动记录

- 检查时间：2026-11-15 09:57 CST。
- 实验代号与批次：`v1-rage-ads-v2` / `ads-rev-reverse-complement`。
- 训练评估状态：训练 8/8、评估 8/8 完成；各任务 4,000 步。并行日志记录最后一项于 09:50:47 完成；当前 GPU 利用率 0%，显存占用 0 MiB。
- 用户要求绘制八任务粗筛与精算两阶段的余弦值频次图，并确认阶段为现有 q0=32、q1=16。
- 由于选择执行期间未保存完整余弦矩阵，本次通过相同梯度探针和相同分块相似度实现重算频次，不覆盖 selection.json 或训练评估产物。
- q0 统计全部粗筛锚点对；q1 统计加入未命中粗锚点区间 q1=16 锚点后的完整最终矩阵，包含 q0 锚点。余弦范围 [-1,1] 等宽分箱 100 桶；频次图使用对数纵轴并标出 cos=0.1。
- 执行脚本：`scripts/exp/v1-rage-ads-v2/plot_ads_rev_cosine_histograms.py --bins 100`；远程 tmux 会话：`ads_rev_hist`；环境：`rage-timerxl`。
- 预期输出：`outputs/selection/v1-rage-ads-v2/ads-rev/cosine-histograms.png`、`.html`、`.json`。本日志记录启动情况；完成状态将在后续新增日志。
