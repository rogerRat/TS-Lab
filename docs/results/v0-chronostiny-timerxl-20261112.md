# v0 ChronosTiny 与 TimerXL 梯度关系补充

最后修改日期：2026-11-12

## ordinal 0 完整轨迹

ChronosTiny 使用原始预训练权重，基础模型冻结，LoRA r=8、alpha=16、dropout=0、q/v 目标模块、初始化 seed 0。窗口为 context=512、prediction=64、stride=1。每个窗口的 LoRA 梯度投影至 4096 维 Count-Sketch（seed 42），以 BF16 位编码保存。该批次完整覆盖 src-96 ordinal 0 的 230,161 个合法窗口。

每轨 128 个等距窗口构成 8,128 个非临近窗口对（窗口起点至少相隔 576）。TimerXL 与 ChronosTiny 各自在其模型内部计算窗口对余弦相似度；参数空间不同，投影坐标不具跨模型逐坐标对齐含义。两模型窗口对余弦的 Spearman/Pearson 为 0.0758/0.0768，Top-10% 窗口对重合率为 14.4%。全量相邻窗口对余弦的 Spearman 为 -0.1379。轨迹级排序因只有一条轨迹不可估计。

同轨 k=1 指标的 TimerXL/ChronosTiny 均值为：方向余弦 0.9314/0.9348，相对 L2 差 0.4085/0.3574，相对模长差 0.1799/0.0478。它们是在各自梯度空间内计算的窗口时间变化统计，不是跨模型梯度向量相似度。该结果是单轨诊断，不能推出跨轨普遍关系。

## ChronosTiny 前 64 轨扩展

扩展目标为 src-96 前 64 轨全部 1,980,371 个合法窗口，协议沿用原始 ChronosTiny 权重及 4096 维 BF16 LoRA 梯度投影，并计划与同窗口 TimerXL 缓存比较。用户随后调整范围并停止该批次。8 个 worker 曾启动，但整批没有完成标记和有效全轨分析；当前保留部分梯度缓存供后续可能使用，不纳入结果汇总。缓存状态不代表完成。

## 相关批次状态

- TimerXL src-96 前四轨完整 LoRA 梯度：完成；缓存继续供四轨时间变化分析使用。
- TimerXL src-96 前 64 轨 LoRA 与 lm_heads 梯度：完成；详细分析见 [64 轨变化报告](v0-timer-XL-grad-64-change-velocity.md)。
- src-96 三模型 q=16 Uniform 梯度准备：TimerXL、TimesFM 已完成；Chronos 未完成，当前无运行进程，保留已有缓存。该准备批次没有可报告的 Chronos 全批结果。

配置、日志、缓存和输出的具体路径及完成标记记录在 [EXP.md](../exp/EXP.md)。所有数值只取自已完成输出及报告；未完成的 64 轨扩展不作插补或外推。

## ChronosTiny 与 TimerXL LoRA 梯度关系：src-96 ordinal 0 单轨试验

使用原始预训练 ChronosTiny（非预热检查点），在src-96第0条轨迹的230,161个合法窗口逐窗计算LoRA梯度，投影为4096D Count-Sketch（seed42）并以BF16保存；基础模型冻结，LoRA r=8、alpha=16、dropout=0、q/v目标模块、初始化seed0。与同轨全量TimerXL 4096D投影梯度比较。两模型参数空间不同，投影坐标不对齐，因此主相关性定义为同一轨迹内窗口对余弦相似度的跨模型相关，而非两个模型梯度向量的直接余弦。

每模型沿全轨等距取128个相同窗口，保留起点相隔至少576的8,128个窗口对。TimerXL与ChronosTiny窗口对余弦的Spearman为0.0758、Pearson为0.0768；Top 10%相似窗口对重合率为14.4%。全量相邻窗口对余弦序列的Spearman为−0.1379。k=1平均指标：TimerXL与ChronosTiny方向余弦0.9314与0.9348，相对L2差0.4085与0.3574，相对模长差0.1799与0.0478。现有单轨数据不支持较强的窗口关系相关性；轨迹级排序因仅一条轨迹无法评估，也不能推断跨轨总体关系。

逐滞后指标、配对窗口对统计和完成标记位于 outputs/train/v0-chronostiny-timerxl-lora-track0/analysis/；全部投影梯度缓存位于同目录formal/。完整性标记已核验。
