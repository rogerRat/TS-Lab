# main-ads-v0 结果记录

## 批次：traj-replay

- 状态：TimerXL 训练 2,000/2,000 步完成；ordinal 0、1 梯度缓存均完成。下游任务仅启用 TIME1。
- 检查点：`outputs/train/main-ads-v0/checkpoint/adapter/`；LoRA q/v，r=8、alpha=16、dropout=0，训练 seed 0。
- Loss：每 10 步记录，共 200 个点；step 10 为 0.943739474，step 2,000 为 0.730318844。step 1,950 记录 135.59654236。
- Loss 数据：`outputs/train/main-ads-v0/checkpoint/loss_every_10_steps.csv`。
- Loss 图：`outputs/train/main-ads-v0/checkpoint/loss_curve.svg`。
- 梯度缓存：投影 4,096 维、seed 0，逐窗口对 LoRA 参数求梯度；输出至 `outputs/grad_cache/main-ads-v0/traj-replay/`。
  - ordinal 0（T000004）：230,161 个合法窗口，已完成。
  - ordinal 1（T000066）：153 个合法窗口，已完成。
- 并行核验：2026-11-15 16:28 CST，8 个 worker 已启动，GPU 利用率 99%，显存占用 7,356 MiB；容器内存占用约 12.3 GiB / 62.0 GiB。

## 下游任务状态

继承的八个任务为 FEV09（FEV task index 9）、TIME1（CPHL/15T）、TIME2（CPHL/30T）、TIME3（CPHL/H）、TIME4（ECDC_COVID/D）、TIME6（epf_electricity_price/H）、TIME7（Commodity_Import/M）和 TIME8（Commodity_Production/M）。当前仅启用 TIME1，其余七项暂不执行。

## 批次：ads-histgram

- 状态：正确口径的 BF16/uint8 计算、矩阵落盘、100/256 桶统计及绘图全部完成；根目录 `.complete` 已核验。
- 输入：TimerXL LoRA Count-Sketch 梯度；ref 为短轨迹 ordinal 1（153 个窗口），src 为长轨迹 ordinal 0（230,161 个窗口）。
- 采样：q=1,2,4,8,16,31,64，按 `uniform_anchor_positions` 均匀布点并含两端点。矩阵行是 src，列是 ref。
- ADS：每个粗矩阵按 BF16 余弦值全局排序，选取最高 `ceil(N/4)` 个 src-ref 配对；BF16 截止值并列按 row-major 顺序精确取足预算。入选配对两端锚点各自分组连续区间，按 `max(1,floor(q/4))` 精算，保留粗锚点。
- 计算和存储：RTX 4080 CUDA BF16 矩阵乘；每个矩阵文件以 uint8 保存256等宽余弦桶ID（桶宽=2/256）。JSON 同时保存256桶原始样本数与从桶中心聚合出的100桶样本数；相对频率为每桶样本数除以矩阵总配对数。
- Uni：另按 ADS 最终 source/ref 锚点数分别等量均匀布点，得到预算严格匹配的 uni 矩阵；逐侧锚点数量及矩阵元素数与 ADS 完全一致。
- q 级汇总：

| q | 精算间隔 | Uni 粗锚点总数 | ADS 最终锚点总数 | top-25% 配对数 | BF16 截止余弦 | ADS/匹配Uni矩阵元素数 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 230,314 | 230,314 | 8,803,659 | 0.0266113 | 35,214,633 |
| 2 | 1 | 115,160 | 229,435 | 2,244,099 | 0.0275879 | 35,080,146 |
| 4 | 1 | 57,582 | 227,665 | 575,420 | 0.0285645 | 34,809,336 |
| 8 | 2 | 28,793 | 139,355 | 151,053 | 0.0290527 | 12,116,316 |
| 16 | 4 | 14,399 | 64,998 | 43,161 | 0.0312500 | 3,117,600 |
| 31 | 7 | 7,434 | 28,653 | 12,998 | 0.0339355 | 772,902 |
| 64 | 16 | 3,604 | 10,443 | 4,499 | 0.0412598 | 135,590 |

- 输出：`outputs/selection/main-ads-v0/ads-histgram-bf16-u8/`。主要汇总为 `cosine_histograms.json`；uni、ADS-VIC、预算匹配 uni 均保存各 q 的矩阵、锚点位置、100 桶计数/频率和256桶计数/频率。
- 图：`uni-q/cosine_histogram.png`、`ads-vic/cosine_histogram.png`、`uni-budget-matched/cosine_histogram.png`；预算匹配的逐 q 同台图为 `uni-vs-ads-vic/cosine_histograms_by_q.png`。由 Conda `rage` 环境 matplotlib 3.9.2 输出。
- 粗筛高分锚点覆盖：用每个粗矩阵已记录的 BF16 top-25% 截止分数重算入选配对两端的粗锚点命中掩码。ref 粗锚点在所有 q 均 100% 命中；src 命中率从 q=1 到64分别为99.69%、99.56%、99.17%、98.08%、93.66%、79.91%、69.02%。src 命中锚点按10个等长时间区间统计时，各区间相对粗网格基线比例范围为：q1 0.999–1.001、q2 0.999–1.001、q4 0.996–1.002、q8 0.994–1.004、q16 0.983–1.023、q31 0.982–1.036、q64 0.897–1.045。描述性结果未见高分锚点集中于少数时间段的明显效应；q64 的 ref 样本仅5个，分段统计不作强推断。
- 新增锚点位置分布：按轨迹时间切成10等长区间，与均匀粗网格比较。src 在 q=2,4,8,16,31,64 的新增点均覆盖全轨迹；各区间点数相对均匀期望约为0.82–1.12倍。ref 也整体铺开；q=2–31 新增点覆盖约89.5%–97.4%的轨迹跨度，q=64仅新增8点、分别落在中间8个区间。q=1没有新增点，因为粗网格已覆盖全部窗口。位置统计支持“空间分布近似粗采样、整体密度上升”，没有集中在少数时间段。锚点位置文件位于 `ads-vic/source_anchor_positions_q*.npy`、`reference_anchor_positions_q*.npy`，与 `uni-q/` 内粗锚点位置文件对应。
- 时序评分矩阵图：scripts/exp/main-ads-v0/plot_score_matrix_heatmaps.py 输出到 outputs/selection/main-ads-v0/ads-histgram-bf16-u8/score-matrix-heatmaps/，共21张 PNG：7个 q ×（uni 粗矩阵、ADS-VIC 最终矩阵、预算匹配 uni）。x轴 ref 锚点按窗口起点从左至右严格递增；y轴 src 锚点按窗口起点从下至上严格递增。矩阵格颜色表示 uint8 余弦桶；统一使用桶ID 115–140（桶中心约 -0.098 至 0.098）的高对比色阶，以0为中心，范围外饱和；不再叠加白色筛选层。
- 激活次数边际图：对每张矩阵独立按 uint8 桶ID从高到低选取 ceil(矩阵配对数/4) 个配对；截止桶内按 row-major 顺序精确补足预算。每个入选配对分别给对应 ref 列和 src 行的激活次数加1。上侧按时间展示各 ref 锚点命中次数，左侧展示各 src 锚点命中次数；每侧计数总和均等于该矩阵的入选配对数。锚点次序、src/ref 激活次数及选择元数据保存在各子目录的 activation_counts_q*.npz。
- 附加 top-1/16 激活图：仅使用 uni 粗锚点，在每张矩阵独立选取 ceil(N/16) 个最高 uint8 桶配对，截止桶按 row-major 顺序截取；每个配对给 src/ref 端各计一次命中。图与 activation_counts_q*.npz 位于 outputs/selection/main-ads-v0/ads-histgram-bf16-u8/score-matrix-heatmaps-top-1-16/，共7张。src 激活锚点占比 q=1/2/4/8/16/31/64 分别为64.81%/60.99%/56.45%/45.90%/37.83%/28.93%/22.90%；ref 在七种 q 下均为100%。
- 正值 top-1/32 附加图：仅绘 uni 粗锚点；对每张矩阵屏蔽 uint8 桶ID<128 的负余弦值，再在非负桶配对中取 ceil(Npositive/32) 个最高分配对，截止桶按 row-major 顺序截取。ID 128 是零及小正值共用桶，量化数据不能在该桶内区分精确零。输出 outputs/selection/main-ads-v0/ads-histgram-bf16-u8/score-matrix-heatmaps-positive-top-1-32/，共7张 PNG 和7份 activation_counts_q*.npz。src 激活锚点占比 q=1/2/4/8/16/31/64 分别为22.12%/19.97%/16.97%/12.18%/10.13%/8.01%/6.56%；ref 均为100%。
- 旧口径：`outputs/selection/main-ads-v0/ads-histgram/` 中 uni 粗采样 FP32 分布仍有效；其中基于 cos>0.1 的 ADS-VIC 与依赖它的预算配对 uni 不符合本次算法，作为审计历史保留，不纳入以上结论。初次软链接路径错误的部分输出也保留于 `ads-histgram-partial-20261115-204010/`。

## 计划批次状态

| 批次 | 状态 | 实际指标与输出 |
|---|---|---|
| ads-budgeted-vein | ref 归一化平均方向评分定义已确认；预算参数待定；未实现、未启动 | 无运行结果或输出；实现与待定参数见 exp.md 的“后续算法设计草案” |
