# v1-timerxl-step1000-gradient-shift

## 基础信息
- 实验代号：`v1-timerxl-step1000-gradient-shift`
- 目标：检验 TimerXL 微调 1,000 步后，同一输入窗口的 LoRA 参数损失梯度相对原始预训练模型是否发生显著变化，并量化方向和幅值。
- 假设：相同窗口上的梯度存在方向或幅值变化。
- 创建/修改日期：2026-11-15（CST）。
- 实际启动时间：2026-11-15 15:12:48 CST。
- 状态：完成；逐窗口梯度、bootstrap 统计和结果核验均完成，完成标记已生成。
- 最后记录状态：2026-11-15 15:14:33 CST 完成；结果见同目录 results.md。
- 用户已确认梯度口径：相同窗口的 TimerXL 原生训练损失对可训练 LoRA 参数求梯度。
- 用户已确认抽样与统计：src-96 的 96 条轨迹每条 64 个窗口，共 6,144 个；抽样 seed 42；Count-Sketch 4,096 维、seed 42；轨迹聚类 bootstrap 10,000 次，95% 区间。

## 模型与训练
- 模型：TimerXL 84M，`thuml/timer-base-84m`，revision `70077a71acce1b4c00d98332fcaabc694255d8e5`；权重位于 `/root/autodl-tmp/pretrained_models/timer-base-84m-84m`。
- 原始组：原始基础权重 + seed 0 LoRA 初始状态。LoRA 为 q_proj/v_proj，r=8、alpha=16、dropout=0。
- 微调组：复用 `outputs/train/v1-crb-redo/grad-prepare/warmup/timerxl-84m/adapter-step-1000/` 的现有 step-1000 LoRA；manifest 显示底座 revision 相同、seed 0、训练 1,000 步；本实验不重新微调。
- 已有微调协议：BF16；基础权重冻结；有效 batch 32、micro-batch 8；AdamW，lr=1e-4、weight_decay=0、betas=(0.9,0.999)、eps=1e-8、grad_clip=1；cosine schedule 总计 4,000 步、warmup 0.03；训练数据 src-8192，轨迹均匀后轨迹内窗口均匀有放回抽样，seed 0。运行前复核 manifest。
- 损失：TimerXL 原生 V6，480 历史 + 96 预测；单窗口损失对全部可训练 LoRA 参数求梯度；eval 模式。
- 梯度表示：Count-Sketch 4,096 维，seed 42；FP32 投影累加后 BF16 保存，不保存原始参数维度梯度。

## 数据与抽样
- 数据：`configs/data/src-96.yaml`、`data/src/shard_dataset/src-96/`，固定 96 条轨迹。
- 窗口：长度 576、起点步长 1，不跨轨迹边界。每轨合法窗口等概率无放回抽 64 个，seed 42；共 6,144 个。保存固定窗口索引，两模型严格配对。
- 若任一轨迹不足 64 个合法窗口，停止并更新方案，不静默改为有放回或跳过。
- 两模型使用同一窗口顺序、loss reduction、精度和梯度投影身份。先少量样本核验有限值与 shape，再运行全样本。

## 评估与统计
- 主指标：逐窗口方向差 `1 - cosine(g_original, g_step1000)`，窗口先在轨迹内平均，再对 96 条轨迹等权平均。
- 辅助指标：相对 L2 差 `||g_step1000-g_original||_2/max(||g_original||_2,epsilon)`、范数比 `||g_step1000||_2/max(||g_original||_2,epsilon)`；均按轨迹汇总。
- 统计：以轨迹为单位有放回抽样 10,000 次，seed 42；报告均值和 percentile 95% bootstrap 区间。主指标区间下界大于 0 时，报告相对梯度完全相同的显著差异，同时报告效应量。
- 结论仅适用于固定 src-96 数据池和该 seed-0 step-1000 checkpoint，不外推到不同训练 seed；Count-Sketch 压缩使结论限定于该投影。
- 数值核验：在固定窗口上重复计算原始组梯度以核验重复误差；记录有限值率、零梯度数、范数分布、配对完整率。

## 执行细节
- 计划入口：`scripts/exp/v1-timerxl-step1000-gradient-shift/compare_gradients.py`。
- Conda 环境：`rage-timerxl`。命令：`python -u scripts/exp/v1-timerxl-step1000-gradient-shift/compare_gradients.py`。
- 仅在 ads-v2 GPU 作业完成且确认资源空闲后，于独立 tmux 会话运行；实时日志、阶段日志、心跳、ETA。
- 输出：`outputs/train/v1-timerxl-step1000-gradient-shift/`；日志：`logs/v1-timerxl-step1000-gradient-shift/`；完整核验后生成 `.complete`。
LSDlvZPliY3nirbmgIHvvJrmiafooYzohJrmnKzlt7Llh4blpIflubbpgJrov4for63ms5Xop6PmnpDkuI4gQVNDSUkg5rqQ56CB5qOA5p+l77yb5bCa5pyq6L+b6KGM5qih5Z6L5Yqg6L295oiW5qKv5bqm6K6h566X44CC

## 决策状态
| 项目 | 设置 | 状态 |
|---|---|---|
| 梯度变量 | 可训练 LoRA 参数梯度 | 用户已确认 |
| 数据样本 | src-96 每轨 64 个，共 6,144 | 用户已确认 |
| 抽样/投影 seed | 42 | 用户已确认 |
| 投影维度 | 4,096 | 用户已确认 |
| 统计单位与重采样 | 轨迹聚类 bootstrap 10,000 次，95% 区间 | 用户已确认 |
| loss/精度 | 原生 V6 480+96、BF16 | 沿用项目协议，开跑前核验 |
| 启动时机 | ads-v2 GPU 作业完成后 | 用户已要求等待 |
