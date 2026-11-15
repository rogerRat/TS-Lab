# main-ads-v0 ADS 全矩阵比例规则 BF16 重算启动

- 检查时间：2026-11-15 21:20 CST。
- 实验代号：`main-ads-v0`；批次：`ads-histgram`。
- 用户确认规则：每个粗分数矩阵按 BF16 余弦分数排序，取最高 ceil(N/4) 个 src-ref 配对；每个入选配对贡献两侧锚点。将两侧入选锚点各自分解为连续区间，按 max(1,floor(q/4)) 精算。
- 并列规则：按准确的 BF16 矩阵乘输出分数判定 top 25%；截止 BF16 值相同的配对按 row-major 顺序精确取足 ceil(N/4)。
- 精度与存储：NVIDIA RTX 4080 CUDA BF16 乘法；最终余弦矩阵仅保存 uint8 256 等宽桶 ID。另保存256桶计数和按桶中心聚合的100桶计数及相对频率。
- 预算对照：uni 的 source/ref 锚点数分别与 ADS-VIC 最终数量精确配平；uni 锚点在完整轨迹范围均匀分布并包含两端点。
- 运行命令：`python -u scripts/exp/main-ads-v0/analyze_ads_histogram_bf16.py --bins 100`；Conda `rage-timerxl`；tmux `main_ads_v0_ads_histgram_bf16`。
- 输出：`outputs/selection/main-ads-v0/ads-histgram-bf16-u8/`。
- 旧目录 `outputs/selection/main-ads-v0/ads-histgram/` 的 uni 粗间隔统计有效；当中基于 cos>0.1 的 ADS-VIC 与预算配对结果标记为旧口径，不用于新算法结论。
- 当前状态：BF16 矩阵与比例筛选运行中，完成标记尚未生成。
