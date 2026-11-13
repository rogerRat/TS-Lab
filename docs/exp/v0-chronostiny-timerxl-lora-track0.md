# v0-chronostiny-timerxl-lora-track0

最后修改日期：2026-11-12

## v0-chronostiny-timerxl-lora-track0

首次提出与最后修改：2026-11-12。状态：完成（2026-11-12）。按用户最新要求，仅计算固定 src-96 的第0条轨迹，完整覆盖其230,161个合法窗口。使用原始预训练 ChronosTiny 权重 cache/pretrained_models/amazon/chronos-t5-tiny/29d808298f1a62493e7b9a5e08529d0d930fa189；基础模型冻结，LoRA r=8、alpha=16、dropout=0、目标q/v、初始化seed 0。上下文512、预测64、步长1。LoRA梯度投影到4096D Count-Sketch（seed42），BF16位编码保存，不保存原始参数维度。完成后与TimerXL同轨全部窗口梯度比较；两模型参数空间不同，采用同窗口配对关系和轨迹内时间变化指标，不把投影坐标视作跨模型对齐。配置 configs/rage/v0-chronostiny-timerxl-lora-track0.json；准备入口 scripts/temp/prepare_chronostiny_lora_full64.py；分析入口 scripts/temp/analyze_chronostiny_timerxl_full64.py；tmux v0-chronostiny-timerxl-track0。用户要求GPU资源尽量跑满后，已改为8个worker并行，每worker负责约28,770个不重叠窗口；worker0复用之前单worker已完成的窗口缓存。启动后各worker首批512窗口均已进入进度阶段。单worker512窗口预演约23秒；8 worker早期并行阶段测得首批处理约1.2秒，仍需以持续吞吐更新ETA。结果：230,161/230,161个窗口梯度及分析目录.complete均已核验。每轨128个等距窗口构成8,128个非临近窗口对（窗口起点间隔至少576），两模型内部窗口对余弦的Spearman=0.0758、Pearson=0.0768；Top 10%窗口对重合率14.4%。全量相邻窗口对余弦Spearman为−0.1379。轨迹级排序因只有一条轨迹而不可估计。k=1时TimerXL/ChronosTiny平均方向余弦0.9314/0.9348，相对L2差0.4085/0.3574，相对模长差0.1799/0.0478。两模型投影坐标没有跨模型对齐含义；数值仅用于各自内部窗口关系与时间曲线比较。单轨结果不足以判断跨轨普遍相关性。正式输出 outputs/train/v0-chronostiny-timerxl-lora-track0/；分析表为 analysis/pair_geometry.csv、lag_metrics.csv、summary.json。
