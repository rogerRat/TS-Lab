# v0-timerXL-grad-4-lm-head

最后修改日期：2026-11-11

## v0-timerXL-grad-4-lm-head

首次提出：2026-11-11。最后修改：2026-11-11。批次完成。使用已完成的四条完整 TimerXL LoRA 梯度轨迹，每轨等距抽取128个同窗口，计算 TimerXL 输出头 lm_heads 的4096D投影梯度（seed 42），并匹配已有4096D BF16 LoRA投影。输出头使用冻结骨干与原生 shifted-patch MSE 的解析梯度。两个参数空间不同，只比较各自空间内窗口对的余弦相似度，不做跨空间逐坐标余弦。轨迹内仅纳入起点相隔至少576的窗口对；另报告跨轨迹窗口对。指标：窗口对余弦的 Spearman/Pearson、输出头最接近10%窗口对的 LoRA 余弦均值提升、LoRA最接近10%的召回。轨迹1只有153个窗口，不存在轨迹内非临近窗口对。代码 scripts/temp/v0_timerxl_grad4_analysis.py；正式产物 outputs/train/v0-timerXL-grad-4-lm-head/；结论 docs/exp/v0-timerXL-grad-4-lm-head/results.md。
