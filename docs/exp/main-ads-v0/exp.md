# main-ads-v0 实验设计

## 基础信息

| 字段 | 内容 |
|---|---|
| 实验代号 | `main-ads-v0` |
| 目标 | 基于全新的 TimerXL LoRA 检查点准备两条完整轨迹的梯度缓存；比较多采样间隔下 uni 与 ADS-VIC 锚点梯度方向余弦分布 |
| 假设 | src-256 上训练得到的 TimerXL LoRA 梯度可作为两条完整轨迹的缓存表示 |
| 文档创建时间 | 2026-11-15 CST |
| 最后修改时间 | 2026-11-15 CST |
| 实际实验启动时间 | 2026-11-15 15:42:54 CST |
| 当前状态 | traj-replay 完成；ads-histgram 的 BF16/uint8 比例法矩阵、100/256桶统计、预算匹配 uni 与图表全部完成；下游任务仅启用 TIME1 |
| 继承实验 | `v1-rage-ads-v2`，继承 TimerXL、窗口、LoRA 与优化器主要设置；训练步数改为 2,000 |

## 模型与训练

- 模型：TimerXL 84M，`thuml/timer-base-84m`，本地预训练权重 `/root/autodl-tmp/pretrained_models/timer-base-84m-84m`。
- 本实验从预训练权重和 seed 0 LoRA 初始化，生成独立的新检查点，不加载旧的 8192 训练检查点。
- 精度：BF16 模型前向与权重；TimerXL 原生 V6 损失，480 历史点 + 96 预测点。
- LoRA：`q_proj`、`v_proj`，r=8、alpha=16、dropout=0；训练参数 262,144。
- 优化器：AdamW，lr=1e-4、weight decay=0、betas=(0.9, 0.999)、eps=1e-8、梯度裁剪=1。
- 训练：2,000 次更新；有效 batch=32，micro-batch=8；cosine 学习率，3% warmup，最低学习率比例 0；seed=0。
- 数据采样：src-256 共 256 条轨迹；每步先均匀抽轨迹，再在该轨迹的合法 576 点窗口中均匀抽样；每个窗口起点步长为 1。
- Loss 记录：每 10 步记录该步有效 batch 平均训练 loss，保存 CSV 和曲线图。
- 新检查点：`outputs/train/main-ads-v0/checkpoint/adapter/`。

## 数据处理

- 数据配置：`configs/data/src-256.yaml`。
- 数据文件：`data/src/shard_dataset/src-256/values.npy`、`offsets.npy`；数据集共 256 条轨迹、35 个配置、6,529,220 个点。
- 窗口总长 576；前 480 点作为历史，后 96 点作为预测目标。
- 训练随机数种子为 0。每步从 256 条轨迹均匀有放回抽取 32 个轨迹，再分别在轨迹内均匀有放回抽取合法窗口。

## 梯度缓存

- 批次：`traj-replay`。
- 检查点：本实验生成的 `outputs/train/main-ads-v0/checkpoint/adapter/`。
- 梯度变量：TimerXL 原生 V6 单窗口损失对全部可训练 LoRA 参数求梯度。
- 投影：Count-Sketch 随机投影，4,096 维，seed=0；每个窗口单独计算梯度，不跨窗口求平均。ordinal 0 使用 8 个 GPU worker 并行，每 worker batch=4；ordinal 1 使用单 worker。
- 轨迹：`src-256` ordinal 0 与 ordinal 1，按配置顺序选取。
  - ordinal 0：`chronos / monash_australian_electricity / T000004`，长度 230,736。
  - ordinal 1：`chronos / monash_fred_md / T000066`，长度 728。
- 每条轨迹计算全部合法窗口起点，stride=1。每条轨迹缓存一行对应一个窗口，列数为 4,096；缓存以 float16 保存。
- 输出：`outputs/grad_cache/main-ads-v0/traj-replay/trajectory-000/` 与 `trajectory-001/`，包含 `gradient_sketches.npy`、`manifest.json` 和 `.complete`。
- Replay 内容：待定；确定后更新本实验文档与结果记录。

## 下游任务

任务列表继承 v1-rage-ads-v2，共八项。当前仅启用 TIME1，其余任务保留在实验计划中，暂不执行。

| 任务代号 | 具体任务 | 当前状态 | 配置文件 |
|---|---|---|---|
| FEV09 | FEV task index 9 | 暂不启用 | `configs/eval/fev/v1-rage-ads-fev-09.yaml` |
| TIME1 | CPHL/15T | 当前启用 | `configs/eval/time/v1-rage-ads-time-01.yaml` |
| TIME2 | CPHL/30T | 暂不启用 | `configs/eval/time/v1-rage-ads-time-02.yaml` |
| TIME3 | CPHL/H | 暂不启用 | `configs/eval/time/v1-rage-ads-time-03.yaml` |
| TIME4 | ECDC_COVID/D | 暂不启用 | `configs/eval/time/v1-rage-ads-time-04.yaml` |
| TIME6 | epf_electricity_price/H | 暂不启用 | `configs/eval/time/v1-rage-ads-time-06.yaml` |
| TIME7 | Commodity_Import/M | 暂不启用 | `configs/eval/time/v1-rage-ads-time-07.yaml` |
| TIME8 | Commodity_Production/M | 暂不启用 | `configs/eval/time/v1-rage-ads-time-08.yaml` |

评估只针对当前启用的 TIME1，使用该任务配置中的 rolling 流程。其余七项不进入当前执行队列。

## 执行细节

- 训练入口：`scripts/exp/main-ads-v0/run_main_ads_v0.py --stage train`。
- 并行梯度入口：`scripts/exp/main-ads-v0/compute_grad_cache_parallel.py --workers 8 --batch-size 4`。
- Conda 环境：`rage-timerxl`。
- 训练命令：`python -u scripts/exp/main-ads-v0/run_main_ads_v0.py --stage train`；梯度命令：`python -u scripts/exp/main-ads-v0/compute_grad_cache_parallel.py --workers 8 --batch-size 4`。
- 长任务使用 tmux 会话 `main_ads_v0_traj_replay`。
- 日志：`logs/main-ads-v0/`。
- 训练输出：`outputs/train/main-ads-v0/checkpoint/`。
- 梯度输出：`outputs/grad_cache/main-ads-v0/traj-replay/`。
- 有效重算入口：`scripts/exp/main-ads-v0/analyze_ads_histogram_bf16.py --bins 100`；频次直方图入口：`scripts/exp/main-ads-v0/render_ads_histogram.py`；时序评分矩阵与锚点分数和图入口：`scripts/exp/main-ads-v0/plot_score_matrix_heatmaps.py`。输出到 `outputs/selection/main-ads-v0/ads-histgram-bf16-u8/`。分析使用 Conda `rage-timerxl` 的 CUDA BF16；绘图使用 Conda `rage` 的 matplotlib 3.9.2。

## 批次管理

| 批次 | 内容 | 状态 | 主要参数 | 输出 |
|---|---|---|---|---|
| `traj-replay` | src-256 TimerXL LoRA 训练；记录每 10 步 loss；对 ordinal 0、1 的所有合法窗口计算 4096 维 seed 0 梯度草图 | 训练与两条轨迹梯度缓存已完成；下游仅启用 TIME1 | 2,000 步；LoRA q/v，r=8、alpha=16；训练 seed 0；投影 seed 0；8 worker × batch 4 | `outputs/train/main-ads-v0/` |
| `ads-histgram` | 短轨迹 ordinal 1 作 ref、长轨迹 ordinal 0 作 src；q=1,2,4,8,16,31,64 比较 uni 与 ADS-VIC 的余弦频率；另做逐侧预算匹配 uni | 完成；旧 cos>0.1 口径仅作审计保留；最高25%配对的 BF16/uint8 正确结果已完成 | 每个粗矩阵按 BF16 分数排序取 ceil(N/4) 配对；截止分数并列按 row-major 精确截取；参与配对的两侧锚点连续区按 max(1,floor(q/4)) 加密；矩阵 uint8 存256桶；100桶从桶中心聚合 | `outputs/selection/main-ads-v0/ads-histgram-bf16-u8/` |

## 决策记录

| 项目 | 设置 | 状态 |
|---|---|---|
| 继承来源 | v1-rage-ads-v2 | 已确定 |
| 新检查点 | src-256 上独立训练 2,000 步 LoRA | 已确定 |
| Loss 记录频率 | 每 10 步 | 已确定 |
| 梯度投影 | 4096 维，seed 0 | 已确定 |
| 完整梯度轨迹 | src-256 ordinal 0 和 1 | 已完成 |
| 下游任务范围 | FEV09、TIME1/2/3/4/6/7/8；仅 TIME1 当前启用 | 已确定 |
| Replay 内容 | 待定 | 待确定 |
| ADS 配对筛选 | 每个粗余弦矩阵全局取分数最高的 ceil(N/4) 个 src-ref 配对；BF16 截止分数并列按矩阵行优先序精确截取 | 已由用户确认并完成 |
| ADS 精算区域 | 全局最高 25% 配对中出现的 src/ref 锚点，各自合并连续区间 | 已由用户确认并完成 |
| 精算采样间隔 | max(1, floor(q/4)) | 已确定 |
| 分数精度 | GPU BF16 余弦计算，评分矩阵存 uint8 256桶ID；另保存 256 桶原始计数；100 桶统计按桶中心聚合 | 已完成 |
| 直方图与对照 | [-1,1] 等宽 100 桶；保存每桶样本数及 count/总配对数；uni按ADS-VIC最终两侧锚点数分别匹配 | 已确定 |
