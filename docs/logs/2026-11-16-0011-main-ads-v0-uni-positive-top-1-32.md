# main-ads-v0 uni 正值 top-1/32 激活图

- 检查时间：2026-11-16 00:11 CST。
- 实验代号与批次：main-ads-v0 / ads-histgram。
- 状态：完成；新增7张仅使用 uni 粗锚点的正值 top-1/32 时序矩阵图。
- 当前 top-1/16 的筛选未显式过滤负分，但 q=1/2/4/8/16 的截止桶ID为138、q=31/64为139，均为正余弦桶，所以入选配对没有负分；旧图的矩阵热力图仍显示负分。
- 执行命令：python -u scripts/exp/main-ads-v0/plot_score_matrix_heatmaps.py --positive-only --activation-denominator 32 --methods uni-q；Conda 环境 rage，tmux 会话 main_ads_v0_uni_positive_top_32。
- 激活定义：先排除负余弦桶（uint8 ID<128），再在非负配对中按桶ID降序选 ceil(Npositive/32) 个；截止桶内按 row-major 顺序截取。ID 128 覆盖零与小正值，uint8 数据无法在桶内区分精确零。
- 颜色：负桶在热力图中置白；非负值使用 Reds 色阶，桶ID 128–148 映射，超过上界饱和。
- 输出：outputs/selection/main-ads-v0/ads-histgram-bf16-u8/score-matrix-heatmaps-positive-top-1-32/；包含 heatmap_manifest.json、.complete、7张 PNG 和7份激活次数 sidecar。
- src 激活锚点比例 q=1/2/4/8/16/31/64：22.12%/19.97%/16.97%/12.18%/10.13%/8.01%/6.56%；ref 七种 q 均100%。
- 核验：7张 PNG 均可读取；激活计数总和等于正值候选数的 ceil(Npositive/32)；截止桶均为正值桶；锚点时序次序严格递增；.complete 存在。
- 原 top-1/16 结果保留为对照。
- 更新文档：docs/exp/main-ads-v0/exp.md、docs/exp/main-ads-v0/results.md、docs/EXP.md、docs/RESULTS.md。
- 未解决问题：无。
