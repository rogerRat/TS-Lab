# v0-timer-XL-grad-4-change-velocity

最后修改日期：2026-11-11

## v0-timer-XL-grad-4-change-velocity

首次提出：2026-11-11。最后修改：2026-11-11。使用已完成的四条 TimerXL LoRA 4096D BF16投影梯度轨迹，长度分别230161、153、24384、230209。对每个滞后 k=1..512 和每条轨迹的每个有效窗口对 (t,t+k) 逐一计算三个指标：(1) 方向余弦 cos(g[t],g[t+k])，先以各向量模长归一化；(2) 相对L2差 ||g[t]-g[t+k]|| / ((||g[t]||+||g[t+k]||)/2)；(3) 相对模长差 abs(||g[t]||-||g[t+k]||) / ((||g[t]||+||g[t+k]||)/2)。每个k报告所有有效对的均值、总体标准差和对数 N-k。轨迹1仅153个窗口，k>=153无有效对，记录为零对且指标未定义。直接使用全部有效窗口对，不抽样；不跨轨迹边界。禁止将投影空间结论视为未投影LoRA参数空间结论。代码 scripts/temp/v0_timerxl_grad4_change_velocity.py；输入 outputs/train/v0-timerxl-src96-first4-full-grad-4096d-bf16/formal/；输出 outputs/train/v0-timer-XL-grad-4-change-velocity/。远端 Conda rage-timerxl，CUDA矩阵计算，tmux v0-timerxl-grad4-change-velocity。完成后核验全滞后对数并分析曲线形态。

v0-timer-XL-grad-4-change-velocity 已完成（2026-11-11）：四轨全部有效窗口对、k=1..512，每轨各一张三面板PNG/SVG图。轨迹1仅k<=152有效。原始结果与分析见 docs/exp/v0-timer-XL-grad-4-change-velocity/results.md；输出目录和图目录完成标记均已核验。

#### v0-timerXL-grad-4-lm-head：change-velocity-comparison 补充批次

首次提出：2026-11-11。最后修改：2026-11-11。检验输出头梯度计算的方向余弦、相对L2差、相对模长差曲线是否接近LoRA。原每轨128个等距样本不能构成k=1..512的连续窗口对，故在长轨迹0、2、3各选两个相距很远、长度768的连续窗口段（起点分别约为可用范围的25%、75%），短轨迹1使用完整153窗口，总共4761个窗口。在相同窗口上计算输出头lm_heads的4096D投影梯度，使用相同模型、窗口、投影维度和seed；匹配既有LoRA BF16投影梯度。各段内所有k=1..512有效窗口对按对数加权合并；短轨迹仅k<=152。三个指标定义沿用 v0-timer-XL-grad-4-change-velocity，逐k报告输出头曲线、同段LoRA曲线及全轨LoRA参考曲线。比较同段两曲线的数值误差与形态，并检查抽样段LoRA曲线相对全轨是否代表性。代码 scripts/temp/v0_timerxl_grad4_head_velocity.py；输出 outputs/train/v0-timerXL-grad-4-lm-head/change-velocity/；远端rage-timerxl环境、tmux v0-timerxl-head-velocity。此补充批次不代表全轨所有窗口的输出头曲线。

v0-timerXL-grad-4-lm-head 的 change-velocity-comparison 补充批次于2026-11-11完成。4761个同窗口输出头梯度、逐k对照表和四张分轨叠加图均已生成并核验；输出头不能整体数值近似LoRA三指标曲线。详见 docs/exp/v0-timerXL-grad-4-lm-head/v0-timerXL-grad-4-lm-head-change-velocity.md。

#### v0-timer-XL-grad-4-change-velocity：stride-16-rank 补充批次

首次提出：2026-11-11。最后修改：2026-11-11。四轨各从第1个合法窗口开始，每隔16窗口取一个已有4096D BF16投影LoRA梯度。仅对k=16,32,...,512计算同一采样相位上所有有效窗口对的方向余弦、相对L2差、相对模长差，和全量曲线的相同k逐点对照；k<16不可直接识别。以四轨共同k={16,32,48,64}的等k平均值比较三指标排序：方向余弦低者变化快，相对L2差及相对模长差高者变化快。另比较三条长轨k=16..512的排序。为了评估起点敏感性，重复采样相位0..15；正式默认相位0即第1、17、33...个窗口。短轨153点在k较大时有效采样对很少，仅共同前四个k用于四轨排序。读取现有完整缓存，不重新计算模型梯度。代码 scripts/temp/v0_timerxl_grad4_stride16_rank.py；输出 outputs/train/v0-timer-XL-grad-4-change-velocity/stride-16-rank/；远端rage-timerxl环境，tmux v0-timerxl-stride16-rank。

#### v0-timer-XL-grad-4-change-velocity：two-random-16-blocks 补充批次

首次提出：2026-11-11。最后修改：2026-11-11。用户撤回每16窗口采一个梯度方案，改为每条轨迹随机选择两个互不重叠、各含16个连续合法窗口的区间。使用固定随机种子0以便复现，仅从四轨现有完整LoRA 4096D BF16投影梯度缓存读取所选128个窗口的梯度，不新增模型计算。每个区间内计算k=1..15的方向余弦、相对L2差及相对模长差，跨区间不配对；每轨每k按两区间合计2*(16-k)对取均值。和全轨全量k=1..15曲线比较逐k绝对差、曲线形态；按15个k等权平均构造各轨变化速度分数，方向余弦越低表示方向变化越快，相对L2/模长差越高表示相应变化越快。检验四轨全量排序能否由两个区间恢复。k=15每轨仅2对，必须注明不确定性。该一次抽样只能判断此次随机区间的表现，不能估计随机抽样成功率。代码 scripts/temp/v0_timerxl_grad4_two_blocks.py；输出 outputs/train/v0-timer-XL-grad-4-change-velocity/two-random-16-blocks/。

two-random-16-blocks 补充批次已完成（2026-11-11）。seed 0每轨两个互不重叠的16窗口区间，共128梯度；三项k=1..15排序都未完整恢复全轨，均将轨迹0和3对调。结果与限制见 docs/exp/v0-timer-XL-grad-4-change-velocity/v0-timer-XL-grad-4-two-random-16-blocks.md；旧stride-16-rank方案已撤回，不作为现行结论。
