# LoRA 训练支持

最后修改日期：2026-11-10

## 适用范围

可选 LoRA 路径支持以下已登记的可训练模型批次：

| Student | 模型家族 | 默认 target modules | 备注 |
|---|---|---|---|
| chronos-tiny | Chronos-T5 Tiny | q, v | 仅注意力适配器；绑定的输入/输出 embedding 保持冻结 |
| chronos-small | Chronos-T5 Small | q, v | 历史 student 别名；共用 Chronos-T5 训练路径 |
| chronos-base | Chronos-T5 Base | q, v | 仅注意力适配器 |
| timerxl-84m | TimerXL Base 84M | q_proj, v_proj | 使用原生 TimerXL 损失与预测适配器 |
| timesfm-2.5 | TimesFM 2.5 200M | q_proj, v_proj | 使用原生 TimesFM 训练损失 |

TimesFM 3.0 与 Sundial 在 student 训练登记中仍仅支持推理或不可部署，不属于本次 LoRA 训练范围。

## 配置

LoRA 通过 model worker manifest 的 training 对象显式启用。不含 lora 的现有 manifest 继续使用原有全参数训练路径。

manifest 片段示例：

    {
      "training": {
        "lora": {
          "r": 8,
          "alpha": 16,
          "dropout": 0.0
        }
      }
    }

默认值为 rank 8、alpha 16、dropout 0.0。target_modules 可通过模块后缀列表提供，用于受控变体；所选后缀必须实际存在于该 student 的模型中。

实现基于 PEFT LoRA，bias="none"，task_type=None。骨干参数由 PEFT 冻结，仅适配器参数参与优化与梯度裁剪。模型原有损失函数保持不变。对于带适配器的模型，适配器模型引用更新为 PEFT 包装器，使原生损失路径使用可训练模块。

每个 student worker 在 LoRA 注入后创建其基线状态。PEFT 默认零初始化的 B 矩阵保证初始前向函数不变。检查点状态以现有格式保存包装后的模型状态。恢复契约记录归一化后的 LoRA 设置，因此 rank、target 或 dropout 的变更不会静默复用不兼容的优化器检查点。

## 输出头梯度缓存

LoRA 训练与输出头（output-head）梯度缓存是相互独立的功能。梯度缓存运行器仍使用预训练检查点路径，不会自动加载 student LoRA 检查点。除非 worker manifest 显式包含 training.lora，否则现有实验不启用任何 LoRA 适配器。

## 依赖与验证状态

Chronos、TimerXL 与 TimesFM 的模型需求配置中固定 PEFT 0.18.1。配置安装器会在对应运行环境中检查 PEFT 导入。

本次变更仅经过静态检查与语法校验。尚未加载任何模型检查点，未运行梯度或优化器步骤，也尚未测量任何 LoRA 行为或质量对比。
