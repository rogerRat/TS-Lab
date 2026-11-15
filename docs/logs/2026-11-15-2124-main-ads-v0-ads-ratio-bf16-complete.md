# main-ads-v0 ADS 比例法 BF16/uint8 批次完成

- 检查时间：2026-11-15 21:24 CST。
- 实验代号：`main-ads-v0`；批次：`ads-histgram`。
- 运行：Conda `rage-timerxl`，RTX 4080 CUDA BF16；2026-11-15 21:20:47 启动，21:23:18 完成，耗时约2.5分钟。
- 方法：q=1,2,4,8,16,31,64；每个粗余弦矩阵取最高 ceil(N/4) 对，BF16精确值并列按row-major取足；两侧入选锚点连续区按 max(1,floor(q/4)) 精算；预算匹配 uni 按 ADS 最终 source/ref 锚点数逐侧均匀布点。
- 输出矩阵以 uint8 保存256桶ID；JSON中每个分布都有100桶计数/频率和256桶计数/频率。100桶由256桶中心值映射得到。
- q对应的 ADS最终总锚点数：1=230,314；2=229,435；4=227,665；8=139,355；16=64,998；31=28,653；64=10,443。详细预算、矩阵元素数、top四分位配对数和BF16 cutoff见 `docs/exp/main-ads-v0/results.md` 及 `cosine_histograms.json`。
- matplotlib：Conda `rage` 中已有3.9.2；使用 `render_ads_histogram.py` 输出 uni、ADS-VIC、预算匹配 uni 单独图及逐 q 对照图。
- 验证：所有矩阵 dtype 为 uint8、shape 与manifest一致；每项100桶及256桶计数总和均等于矩阵元素数；频率和为1；top四分位配对数等于 ceil(N/4)；uni/ADS逐侧预算相等；根目录 `.complete` 存在。
- 旧口径处理：旧目录的 cos>0.1 ADS-VIC 及其预算匹配结果明确标记作废但保留；旧 uni 粗采样统计仍有效。
- 输出：`outputs/selection/main-ads-v0/ads-histgram-bf16-u8/`；渲染日志：`logs/main-ads-v0/ads-histgram-bf16-u8-render_20261115212419.log`。
