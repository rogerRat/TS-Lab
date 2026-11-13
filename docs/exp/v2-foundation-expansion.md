# v2-foundation-expansion

最后修改日期：2026-11-12

## v2-foundation-expansion：轻量基础模型接入

首次提出和最后修改：2026-11-12。目标：按用户批准的范围，将 Chronos-2-Small（27,934,624 参数）、Chronos-2（119,477,664）、Chronos-Bolt-Tiny（8,652,672）、Mini（21,236,096）、Small（47,718,016）和 Moirai-2.0-R-Small（11,387,208）接入现有预测、训练、LoRA 与基于梯度的数据选择流程。参数数目与模型修订来自官方 Hugging Face 模型记录；六模型的实际参数、推理、损失、LoRA 反传及逐窗口梯度已逐一核验。Moirai 权重为 CC-BY-NC-4.0，仅用于非商业研究。

| 批次 | 权重 ID 与固定修订 | 接入方法 | 状态 |
|---|---|---|---|
| Chronos-2-Small | autogluon/chronos-2-small @ ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a | 官方模型前向、原生分位数损失、注意力 LoRA、独立序列梯度 | CPU 真权重冒烟通过：推理、损失、LoRA、单窗梯度、双窗 VJP 等价 |
| Chronos-2 | amazon/chronos-2 @ 29ec3766d36d6f73f0696f85560a422f50e8498c | 同上 | CPU 真权重冒烟通过：推理、损失、LoRA、单窗梯度、双窗 VJP 等价 |
| Bolt-Tiny | amazon/chronos-bolt-tiny @ a0e552de83495b5c28c14c71c374f3e33280b340 | 官方模型前向、原生分位数损失、注意力 LoRA、逐窗口梯度 | CPU 真权重冒烟通过：推理、损失、LoRA、单窗梯度、双窗 VJP 等价 |
| Bolt-Mini | amazon/chronos-bolt-mini @ 251268337516a88e253628c43e1d26ec577b376b | 同上 | CPU 真权重冒烟通过：推理、损失、LoRA、单窗梯度、双窗 VJP 等价 |
| Bolt-Small | amazon/chronos-bolt-small @ 772f3d25d38aec6d914c8949dab4462e2d46f5d8 | 同上 | CPU 真权重冒烟通过：推理、损失、LoRA、单窗梯度、双窗 VJP 等价 |
| Moirai-2.0-R-Small | Salesforce/moirai-2.0-R-small @ 30f43ff08c8494f4943ae1521e9d4e94a0fbb389 | 官方模块与窗口转换、直接预测 token 的分位数损失、注意力 LoRA、逐窗口梯度 | CPU 真权重冒烟通过：推理、损失、LoRA、单窗梯度、双窗 VJP 等价 |

现阶段仅开发接口，不启动正式实验。窗口、优化器参数、数据池、训练步数、评估配置仍由后续具体实验配置决定，不沿用其他模型的数值。代码路径：src/adapters/foundation/chronos_family.py、moirai2.py、spec.py、factory.py；src/gradient/new_foundation.py；src/rage/experiments/students.py、full_pool_models.py、lora.py；验证入口 scripts/exp/v2_foundation_expansion/smoke_foundation_models.py（以 `python -m scripts.exp.v2_foundation_expansion.smoke_foundation_models` 运行）。已核对 Chronos 官方损失与逐窗口损失、六模型预测输出、LoRA 可训练参数、梯度投影和标准输出路径；真实权重烟雾测试均完成。用户确认所有新增模型使用专用 Conda 环境；Chronos-2/Bolt 统一使用新建 rage-chronos2，Moirai 使用新建 rage-moirai2。后者已安装并通过依赖检查，Torch 2.4.1、uni2ts 2.0.0、transformers 4.45.2、PEFT 0.13.2。真实权重冒烟使用 512 点历史、64 点预测、LoRA r=8、alpha=16、dropout=0、128D 梯度投影；双窗口 VJP 与逐个反传核对。Chronos-2 与 Bolt 的训练使用各自官方模型前向返回的分位数损失；逐窗口梯度以同一归一化分位数公式一次批量前向配合 VJP 计算，避免逐窗重复加载。Moirai 使用官方预测窗口转换和模块前向，以最后一个历史 token 的直接预测 token 计算有观测掩码的缩放空间 pinball 损失；该固定窗口训练损失是项目实现，并非官方发布的微调配方，当前限制预测长度不超过 `num_predict_token * patch_size`。所有六模型使用 512 历史点、64 预测点做工程冒烟，128D 双窗口 VJP 与逐窗梯度等价，且 LoRA AdamW 更新及更新后预测通过；Bolt-Tiny 和 Moirai 另验证正式模型构建器的 4096D 双窗口探针。模型注册见 configs/pretrained_models/chronos2.yaml、moirai2.yaml 和 requ/third_parties/model/integrations.lock。该验证不产生正式预测质量指标。当前 GPU 上的 v1-crb-redo 批次不受此次接入影响。
