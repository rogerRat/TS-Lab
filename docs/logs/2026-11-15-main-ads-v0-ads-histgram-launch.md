# main-ads-v0 ads-histgram 批次启动

- 检查时间：2026-11-15 20:59 CST。
- 实验代号：`main-ads-v0`；批次：`ads-histgram`。
- 输入：已完成梯度缓存；短轨迹 ordinal 1 作 ref（153 窗口），长轨迹 ordinal 0 作 src（230,161 窗口）。
- 方法：q=1,2,4,8,16,31,64；uni 均匀采样两端点；ADS-VIC 阈值 cos>0.1，对所有连续命中区按 max(1,floor(q/4)) 精算；另构造逐侧锚点数完全匹配的 uni 作为同台比较。
- 统计：余弦范围 [-1,1] 等宽 100 桶；保存各桶样本数和样本数/总配对数频率。
- 运行环境：分析使用 Conda rage-timerxl；绘图使用 Conda rage（matplotlib 3.9.2 已存在）。
- 命令：`python -u scripts/exp/main-ads-v0/analyze_ads_histogram.py --bins 100`，tmux 会话 `main_ads_v0_ads_histgram`。
- 初次尝试于 20:40 完成 q=1 矩阵计算后，在转换数据盘软链接路径为相对路径时异常退出；不完整输出保留于 `outputs/selection/main-ads-v0/ads-histgram-partial-20261115-204010/`。已修正输出路径记录并于 20:59 重启。
- 当前状态：运行中；成功标记尚未生成。无结果数值可记录。
