# v1-timerxl-step1000-gradient-shift 结果记录

## 批次：paired-src96-step1000

- 状态：完成。实验于 2026-11-15 15:12:48 启动，15:14:33 CST 完成；`.complete` 已核验。
- 对象：TimerXL 84M 原始预训练权重 + seed 0 LoRA 初始状态，与同底座、seed 0 微调 1,000 步的 LoRA checkpoint 对比；原生 V6 loss（480 历史 + 96 预测），计算对 LoRA 参数的逐窗口梯度。
- 样本：src-96 的 96 条轨迹，每轨抽 64 个合法窗口，无放回、seed 42；总计 6,144 个配对窗口。每轨至少有 150 个合法窗口，样本均满足抽样要求。
- 梯度：固定 4,096 维 Count-Sketch，seed 42；原始和 step-1000 梯度均无零范数。原始组重复计算最大 sketch 绝对误差为 `7.45e-9`。
- 统计：先按轨迹汇总，再以 96 条轨迹为单位 bootstrap 10,000 次，seed 42；区间为 percentile 95% CI。

| 指标 | 估计值 | 95% bootstrap 区间 |
|---|---:|---:|
| 方向差 `1 - cosine` | 0.5653 | [0.5256, 0.6054] |
| 平均梯度余弦 | 0.4347 | — |
| 相对 L2 差 | 7.1115 | [4.9393, 10.2253] |
| 梯度范数比 | 7.3737 | [5.2048, 10.5005] |

- 结论：在本实验的固定 src-96 样本与固定 seed-0 checkpoint 上，step-1000 梯度相对原始模型存在统计显著差异。主指标 `1-cosine` 的 95% CI 下界为 0.5256，大于 0；方向差异具有明显量级，平均余弦为 0.4347。投影梯度的平均相对 L2 差约为原始梯度范数的 7.11 倍，平均范数比为 7.37。
- 两模型窗口平均 loss 分别为 3.7110（原始）和 3.5651（step-1000）。该结果只说明本次微调样本上的平均损失较低，不作为梯度显著性的统计检验。
- 边界：结论适用于这一个微调 checkpoint、该 src-96 样本池及固定 Count-Sketch 投影；不代表跨训练 seed 的总体结论。L2 和范数比来自投影梯度，需按投影表示解释。

### 产物

- 汇总指标：`outputs/train/v1-timerxl-step1000-gradient-shift/summary.json`
- 逐轨指标：`outputs/train/v1-timerxl-step1000-gradient-shift/trajectory_metrics.csv`
- 逐窗口指标：`outputs/train/v1-timerxl-step1000-gradient-shift/window_metrics.csv`
- 配对窗口索引：`outputs/train/v1-timerxl-step1000-gradient-shift/window_ids.npy`
- 投影梯度：`outputs/train/v1-timerxl-step1000-gradient-shift/gradient_original.npy`、`gradient_step1000.npy`
- 运行日志：`logs/v1-timerxl-step1000-gradient-shift/paired-src96-step1000_20261115151248.log`
- 完成标记：`outputs/train/v1-timerxl-step1000-gradient-shift/.complete`
