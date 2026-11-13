# src-96-q4-output-head-estimate

最后修改日期：2026-11-11

## src-96、q=4 输出头梯度资源测算

首次提出：2026-11-11。最后修改：2026-11-11。仅估算，未运行。沿用固定 96 条轨迹，按逐轨迹 q=4 预算合计每模型 666,575 个锚点。模型为 TimerXL 84M、TimesFM 2.5、Chronos-T5-Base；输出头梯度投影 4096 维并以 BF16 保存的容量约每模型 5.1–5.4 GiB、三模型 15.3–16.2 GiB。用时低置信度区间：TimerXL 0.5–2 小时、TimesFM 9–30 小时、Chronos 8–30 小时。详细依据和未投影容量风险见 `docs/results/src-96-q4-output-head-estimate.md`。本估算不改变当前运行中的 Uniform 批次。
