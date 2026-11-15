# ADS-VIC 余弦频次直方图重算启动

- 启动时间：2026-11-15 13:54 CST。
- 实验代号与批次：`v1-rage-ads-v2` / `rolling-baseline-v2` 的 `ads-vic` 臂。
- 方法依据：原选择记录配置 q0=32、q1=8、阈值 cosine > 0.1；q1 在命中锚点构成的连续段内精算。q1 结果矩阵包含粗筛锚点与新加密锚点。
- 原运行未保存完整余弦矩阵或分箱频次，故按原选择代码和相同梯度探针重算；不覆盖已有选择、训练、评估结果。
- 余弦区间 [-1,1] 分为 100 个等宽桶；图表使用 log 纵轴并标示 cos=0.1。
- 脚本：`scripts/exp/v1-rage-ads-v2/plot_ads_vic_cosine_histograms.py --bins 100`；环境 `rage-timerxl`；tmux 会话 `ads_vic_hist`。启动时 GPU 0% 利用率、0 MiB 显存占用。
- 输出目录：`outputs/selection/v1-rage-ads-v2/ads-vic/`，计划产物为 PNG、交互 HTML、分箱 JSON 和 `.complete`。
