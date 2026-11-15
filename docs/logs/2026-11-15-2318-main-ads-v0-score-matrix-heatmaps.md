# main-ads-v0 时序评分矩阵与边缘分数和图完成

- 检查时间：2026-11-15 23:18 CST。
- 批次：`main-ads-v0/ads-histgram`；状态：21张图、21份边缘数据均已生成，`.complete` 已核验。
- 图组：q=1,2,4,8,16,31,64；每个 q 分别绘制 uni 粗矩阵、ADS-VIC 最终矩阵、预算匹配 uni。
- 轴顺序：ref 锚点作为矩阵列，按时间由左至右递增；src 锚点作为矩阵行，按时间由下至上递增。刻度标签是轨迹窗口起点。
- 热力图：像素颜色为 uint8 256桶余弦分数；白色叠层标出大于等于粗矩阵 top-25% BF16 截止桶的矩阵格。量化桶边界可能让叠层略偏离精确 BF16 cutoff。
- 边缘分数：对每一行/列的所有配对分数（桶中心近似值）求有符号和。每侧幅值按该侧所有锚点分数和的绝对值总和归一到100%；红/蓝颜色标记和为正/负。精确签名总和、锚点时序位置和相对幅值写在 `marginal_sums_q*.npz`。
- 绘图使用 Conda `rage` 的 matplotlib 3.9.2，执行脚本 `scripts/exp/main-ads-v0/plot_score_matrix_heatmaps.py`，日志 `logs/main-ads-v0/score-matrix-heatmaps_20261115231814.log`。
- 验证：21个 PNG 均可读取；21份边缘文件的src/ref位置均严格递增；各侧绝对相对幅值之和均为100%；热力图目录 `.complete` 存在。
- 目标目录：`outputs/selection/main-ads-v0/ads-histgram-bf16-u8/score-matrix-heatmaps/`。
