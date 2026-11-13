# v0-grad-prepare-src-96tr-q16-4096d

最后修改日期：2026-11-11

## v0-grad-prepare-src-96tr-q16-4096d

首次提出：2026-11-11。最后修改：2026-11-11。状态：仅 Uniform 梯度准备；TimerXL 和 TimesFM 已完成，Chronos 四进程运行中。用户于 2026-11-11 21:03 CST 决定停止 CRB 并删除本批次已产生的 CRB 清单及仅由 CRB 使用的梯度缓存。原三模型 CRB+Uniform 队列已停止，后续仅计算 Uniform；旧的 CRB+Uniform 批次完成标记已按新协议重新核验。

输入为固定 `src-96`：96 条完整轨迹，覆盖 src-4096 的 35 个配置，每配置至少 1 条，其余按配置规模分配，配置内 seed 0 无放回抽样；不按轨迹长度替换。索引 `configs/data/src-96.yaml`，独立数据快照 `data/src/shard_dataset/src-96/`；共有 2,721,512 个值和 2,666,312 个合法窗口，逐条源值与偏移已验证。

三模型为 TimerXL 84M、TimesFM 2.5 和 Chronos-T5-Base。窗口上下文 512、预测 64、步长 1。基础参数冻结；LoRA r=8、alpha=16、dropout=0，Chronos 目标 q/v，另两模型目标 q_proj/v_proj，初始化 seed 0。固定 4096 维 Count-Sketch 投影，seed 42，FP32 累加后以 BF16 位编码保存。Uniform 在每条轨迹的合法起点等距取锚点，q=16；逐轨迹取整后每模型共 166,659 个锚点。三模型共用相同 96 条轨迹和相同的 Uniform 锚点身份，各自保存独立梯度缓存。

入口 `scripts/prepare_src_96_gradients.py`；当前队列 `scripts/temp/run_src_96_uniform_queue.sh`，tmux `v0-grad-src-96-uniform`。TimerXL 已完成八进程、TimesFM 已完成两进程；两模型的旧共享缓存已过滤为仅 Uniform 行，CRB 清单和 CRB 专属梯度已删除。Chronos 已完成的 Uniform 行同样保留并过滤，剩余轨迹由四进程继续准备。各模型与整批仅在 Uniform 缓存和清单全部验证后写 `.complete`。旧 CRB+Uniform 计划和约 333,318 次/模型的总计算上界已由用户决策终止，不再作为现行执行方案。
