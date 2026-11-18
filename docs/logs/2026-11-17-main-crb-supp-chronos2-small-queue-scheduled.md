# 2026-11-17 main-crb-supp-chronos2-small 队列排队

- 排队时间：2026-11-17 01:23（Asia/Shanghai）。
- 服务器：ssh tsl；项目 /root/TS-Lab；tmux 会话 main-crb-supp-chronos2-small。
- 当前前置任务：main-crb-supp TimerXL 队列仍有 tmux 会话；检查时 outputs/eval/main-crb-supp/run/.complete 不存在。
- 调度行为：新会话等待前置队列的完成标记；前置队列退出但没有完成标记时停止。标记出现后顺序执行 Chronos-2-small 队列 prepare、validate、run。没有启动与 TimerXL 并行的 GPU 训练。
- 新队列：Chronos-2-small，24 个任务，6 个方法；每任务 16 个评估点，共 384 个点。新加 12 个任务的安全 reference 已预检。
- 实现状态：TIME/FEV Foundation evaluator 支持本地 Chronos-2-small checkpoint 及原生分位数；本地模型 CPU 推理 smoke 通过，输出 2×3×96；源文件 py_compile 与 git diff --check 通过。
- 模型配置：autogluon/chronos-2-small，revision ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a，27,934,624 参数；环境 rage-chronos2。
- 待决策：probe 暂按现有脚本使用 TimerXL q16/q8 梯度。用户可改为 Chronos-2-small probe；在队列实际开始前可以调整脚本和文档。当前没有进行 Chronos-2 GPU 训练。
- 后续：收到 probe 选择后定稿。prepare 阶段会向 docs/EXP.md 和 docs/RESULTS.md 写入 384 个批次初始状态。
