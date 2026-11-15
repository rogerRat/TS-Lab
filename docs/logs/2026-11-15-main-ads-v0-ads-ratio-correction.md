# main-ads-v0 ADS 比例规则修正

- 检查时间：2026-11-15 21:08 CST。
- 用户确认：每个粗评分矩阵按全部 src-ref 配对分数排序，选择最高 1/4 配对；入选配对的两端锚点构成精算候选。
- 用户另确认：精算所有命中锚点形成的连续区间，采样间隔为 max(1,floor(q/4))。
- 错误来源：首次实现错误继承旧版 `ALGO.md` 与 `v1-rage-ads-v2` 的固定 cos>0.1 阈值。该规则不适用于此批次。旧输出保留但 ADS-VIC 与其预算匹配 uni 均标记作废口径；uni 粗采样分布仍有效。
- 正确实现：每个粗矩阵取 ceil(N/4) 个最高 bucket ID 配对；截止桶按 row-major 顺序精确处理并列。BF16 CUDA 乘法计算余弦，uint8 保存256桶ID；保留256桶样本数，并按桶中心聚合为100桶统计。预算匹配 uni 按 ADS-VIC 两侧最终锚点数分别等量均匀采样。
- 旧有效 uni 粗矩阵的100桶统计在 `outputs/selection/main-ads-v0/ads-histgram/cosine_histograms.json`。
- matplotlib 已在 Conda `rage` 中可用，版本3.9.2；早前错误启动的 rage-timerxl matplotlib 安装已撤销，rage-timerxl 未保留此次安装改动。
- 修正实现：`scripts/exp/main-ads-v0/analyze_ads_histogram_bf16.py`；目标：`outputs/selection/main-ads-v0/ads-histgram-bf16-u8/`。启动后续状态记入新增日志。
