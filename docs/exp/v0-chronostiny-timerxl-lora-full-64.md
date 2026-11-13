# v0-chronostiny-timerxl-lora-full-64

最后修改日期：2026-11-12

## v0-chronostiny-timerxl-lora-full-64

首次提出与最后修改：2026-11-12。状态：用户调整后停止，部分梯度缓存保留；无全轨分析结果。目标是在此前固定 src-96 前64轨的所有1,980,371个合法窗口上，计算原始预训练 ChronosTiny 的 LoRA 梯度，并与已完成的同窗口 TimerXL LoRA 梯度检验变化结构是否相关。用户明确要求全窗口计算，先前讨论的每轨128点方案不用于梯度准备。

ChronosTiny 使用原始预训练权重 cache/pretrained_models/amazon/chronos-t5-tiny/29d808298f1a62493e7b9a5e08529d0d930fa189，不使用 v0-grad-mech-test 的100步预热检查点。上下文512、预测64、步长1；基础模型冻结，LoRA r=8、alpha=16、dropout=0、目标q/v、初始化seed 0。全量梯度用Count-Sketch投影到4096D，seed 42，以BF16位编码保存；不保存98304维原始LoRA梯度。TimerXL对照是 v0-timer-XL-grad-64-change-velocity 已完成的同轨同窗口4096D BF16 LoRA缓存。两模型原始参数空间不同，虽然投影维度和seed相同，投影坐标不具有逐维对齐含义；不得将两个4096D向量直接作跨模型余弦来声称模型梯度相关。分析应比较各模型内部的同窗口对余弦关系，以及全轨逐k变化指标与轨迹级排序；轨迹内与轨迹间分别报告。全轨批次完成前不得填写相关性结果。

配置 configs/rage/v0-chronostiny-timerxl-lora-full-64.json；准备入口 scripts/temp/prepare_chronostiny_lora_full64.py；输出 outputs/train/v0-chronostiny-timerxl-lora-full-64/，日志 logs/v0-chronostiny-timerxl-lora-full-64/。运行环境 rage。16窗口预演成功：原始权重可加载、LoRA可构建，投影与BF16缓存写入正常。正式全量计算已于2026-11-12 02:56 CST 在 tmux v0-chronostiny-timerxl-full64 启动，8个worker各处理不重叠的约247546个窗口。512窗口单worker预演用时23.4秒；8 worker同时运行首512窗口的早期ETA约9小时，仅作初步估计。队列 scripts/temp/run_chronostiny_timerxl_full64.sh 将在全部worker及正式完成标记核实后自动运行 scripts/temp/analyze_chronostiny_timerxl_full64.py。分析计算各轨全量k=1..15方向余弦、相对L2差和相对模长差的轨迹级排序，逐窗口相邻方向余弦的同轨时间相关，以及每轨最多128个等距窗口构成的非临近窗口对余弦相关。128点只用于分析窗口对，不改变全窗口梯度准备范围。
