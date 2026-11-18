# main-ads-v0 时序矩阵与激活次数图重绘

- 检查时间：2026-11-15 23:49 CST。
- 实验代号与批次：main-ads-v0 / ads-histgram。
- 状态：完成；旧版边际分数和图及其 sidecar 已由新版本整目录替换。
- 修改脚本：scripts/exp/main-ads-v0/plot_score_matrix_heatmaps.py。
- 执行命令：/root/autodl-tmp/rage/envs/rage/bin/python -u scripts/exp/main-ads-v0/plot_score_matrix_heatmaps.py；使用 tmux 会话 main_ads_v0_score_heatmaps_redraw，过程日志为 logs/main-ads-v0/score-matrix-heatmaps-redraw_20261115.log。
- 绘图范围：q=1,2,4,8,16,31,64；每个 q 绘制 uni-q、ADS-VIC 最终矩阵、预算匹配 uni，共21张 PNG。
- 边际统计：对每张展示矩阵独立按 uint8 桶ID降序选择 ceil(矩阵配对数/4) 个配对；截止桶内部按 row-major 顺序取足预算。每个选中配对分别给对应 src 行和 ref 列加一次命中；两侧计数和均等于选中配对数。数值与位置保存在 activation_counts_q*.npz。
- 矩阵色阶：统一使用桶ID 115–140，桶中心约对应余弦 -0.098 至 0.098，以余弦0为中心；范围外饱和以提升主体分布对比。移除旧版白色高分叠层。
- 输出：outputs/selection/main-ads-v0/ads-histgram-bf16-u8/score-matrix-heatmaps/；含 heatmap_manifest.json 和 .complete。
- 核验：21个 PNG 均可读取；21份激活次数 sidecar 均通过检查；每张图两侧命中次数总和均等于 ceil(矩阵元素数/4)，锚点位置严格按时间递增。
- 修改了 docs/exp/main-ads-v0/exp.md、docs/exp/main-ads-v0/results.md、docs/EXP.md 和 docs/RESULTS.md。
- 未解决问题：无。
