# v2-foundation-expansion

最后修改日期：2026-11-13

## v2-foundation-expansion 工程验证

六个固定修订权重均在 CPU 上完成 512+64 窗口的预测、训练损失、LoRA 反传和 AdamW 单步、双窗口批量 VJP 与逐窗口梯度核对；以下参数数目来自各模型实际加载的 `smoke.json`。这是接入验证，没有正式训练集或预测质量指标。Moirai 权重只可用于非商业研究。

| 模型 | 实际参数 | LoRA 可训练参数 | 冒烟验证 |
|---|---:|---:|---|
| chronos-2-small | 27,934,624 | 196,608 | 通过 |
| chronos-2 | 119,477,664 | 589,824 | 通过 |
| chronos-bolt-tiny | 8,652,672 | 98,304 | 通过 |
| chronos-bolt-mini | 21,236,096 | 172,032 | 通过 |
| chronos-bolt-small | 47,718,016 | 294,912 | 通过 |
| moirai-2.0-small | 11,387,208 | 73,728 | 通过 |

验证产物位于 `outputs/train/v2-foundation-expansion/`，每个模型均有 `smoke.json` 和 `.complete`。
