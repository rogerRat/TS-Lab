# 2026-11-17 main-crb-supp-chronos2-small probe 方案记录

- 时间：2026-11-17 01:24（Asia/Shanghai）。
- 决策：默认复用 TimerXL q16 source gradients 与 TimerXL q8 reference gradients，Chronos-2-small 作为训练和评估模型。
- 依据：用户明确选择 Chronos-2-small 并要求覆盖全部 24 个任务；probe 模型未指定。沿用现有 TimerXL probe 保持选择信号一致并复用已完成的 399,044 个 source anchor 梯度。
- 结果解释：这是 TimerXL CRB 选择信号迁移到 Chronos-2-small 的实验，不是 Chronos-2-small 原生 probe 的 CRB 实验。
- 队列：依赖 tmux 会话 main-crb-supp-chronos2-small 已排队。其启动脚本等待 /root/autodl-tmp/rage/outputs/eval/main-crb-supp/run/.complete；前置 TimerXL 队列失败或无完成标记时，后续自动停止。
- 状态：前置任务完成前不启动 Chronos-2 GPU 训练。若用户要求 native Chronos-2 probe，可在队列开始前改为重建全部 q16/q8 梯度。
