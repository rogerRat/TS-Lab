# v0-crb-mech-test

最后修改日期：2026-11-11

## v0-crb-mech-test

首次提出与开始：2026-11-11。最后修改：2026-11-11。状态：完成原始对比、补充分析、CES 与 CES-Oracle 对比。

目标：在 64 条计算机生成的平滑向量轨迹上，按 CRB 固定三阶段预算逻辑选锚点，与等预算均匀锚点比较逐点向量重建误差，并分析轨迹粗糙度和 CRB 相对 Uniform 的误差差值。

| 批次 | 轨迹和预算 | 方法、指标与完成标准 | 状态 |
|---|---|---|---|
| 64-smooth-uniform-reconstructors | 同一 64 条轨迹、原 Uniform 锚点和 q={4,8,16,32} | 比较线性、自然三次样条、PCHIP、q−1 次 Chebyshev、RBF-GP（两个固定长度尺度）与平滑样条（固定 lambda 敏感性网格）；全 4096 维评估并与 CRB/CES 对照 | 完成 | 2880 条方法-轨迹记录；q≤16 线性 Uniform 平均 L2 最低，q=32 自然三次和低 lambda 平滑样条略低；PCHIP q=32 方向误差最低；见结果报告 |
| 64-smooth-ces-oracle | 同一 64 条轨迹；复用 CES 均匀 pilot 和 q 预算 | 用 mock 生成器解析 g'' 得到真实曲率；剩余锚点按真实曲率密度积分分配并按局部曲率 CDF 放置；与估计曲率 CES、Uniform、CRB 对比 | 完成 | 1024 条四方法指标记录；CES-Oracle 平均误差未优于 Uniform 或估计版 CES，但四个 q 的 L2 均低于 CRB；详见结果报告 |
| 64-smooth-ces-pilot | 64 条既有轨迹；q={4,8,16,32}；CES、Uniform、CRB 等预算 | 均匀 pilot 占预算 50%，至少 3 个 pilot（端点包含）；按非均匀三点二阶差分估曲率；剩余预算按区间长度乘曲率的 2/5 次方分配，区间内均匀布点；报告全 4096 维逐点误差及块内向量/方向方差 | 完成 | 768 条方法级重建记录；CES 平均误差接近 Uniform 且低于 CRB；q=32 的 L2 低于 CRB 64/64 轨迹；见结果报告 |
| 64-smooth-block-internal-variance | 原 64 条及极平缓单轨迹，复用各方法已有锚点块 | 块内向量 L2 方差、相对向量方差、方向方差与模长方差；每点只计入一个块，均值作块内代表 | 完成 | 原 64 条上 q=32 时 CRB 的相对向量方差、方向方差均在 64/64 轨迹较低；绝对模长方差不一致；详情见结果报告 |
| ultra-smooth-single-reconstruction | 1 条、长度 4000、维数 4096；frequency=0.005–0.05 周期/归一化时间；整条轨迹归一化到平均向量范数 1 | 复用 q={4,8,16,32} 的 CRB/Uniform 锚点重建和指标；单条结果只作极平缓曲线补充，不做相关性统计 | 完成 | 粗糙度分数 1.8373e-7；L2 四个 q 均 Uniform 较低；方向余弦误差约 1e-8–1e-7 |
| 64-smooth-anchor-reconstruction | 64 条，长度 4000，维数 4096；随机种子 20261111；平滑频率参数几何分布于 0.05–2.0 | q={4,8,16,32}；每个 q 严格使用 q 个锚点；CRB 覆盖阶段计入两个端点，剩余预算按覆盖/误差/分裂 2:1:1 分配；CRB 与 Uniform 均在全长上逐维线性插值；报告 mean vector L2、逐坐标 RMSE、mean relative L2、mean direction cosine error；并计算粗糙度与各方法误差及 CRB-Uniform 差值的 Spearman 相关 | 完成 | 64 条；512 条 method-trajectory 指标记录；q=4/8/16/32 的平均 L2 及余弦误差均为 Uniform 较低；见结果报告 |

方法细节：每条轨迹由独立随机的 12 维正弦系数通过 4096 维随机基底映射得到；频率从 0.25 到 18 周期/归一化时间，粗糙度参数单调调制相位。粗糙度排序指标为二阶差分 RMS 除以平均向量范数。轨迹每个合法点视为一个 CRB 位置，通过 TimeSeriesFragment 和现有 CRBEngine 执行真实的覆盖、误差探测与分裂调度。Uniform 端点包含在内。q=4 时端点占用两项覆盖预算，故剩余两项按离散化后的覆盖/误差/分裂配额处理；脚本必须核实最终配额与总预算。方向余弦误差定义为逐点平均 1-cos(x_t, xhat_t)，零范数项按 0 cosine 处理。

Uniform 重建器 sweep 入口与配置：scripts/v0_crb_mech_uniform_reconstructors.py、configs/rage/v0-crb-mech-uniform-reconstructors.json。平滑样条 q=4 不满足 SciPy 最少 5 个锚点条件；lambda 网格为敏感性分析，不从测试误差挑选。GP 使用固定 RBF 长度尺度和 alpha。

64-smooth-ces-oracle 入口与配置：scripts/v0_crb_mech_ces_oracle.py、configs/rage/v0-crb-mech-test-ces-oracle.json。解析真实曲率仅因本 mock 生成器已知；真实梯度场通常不可用。CES 入口与配置：scripts/v0_crb_mech_ces.py、configs/rage/v0-crb-mech-test-ces.json；块内分析入口：scripts/v0_crb_mech_block_variance.py。CES 第一阶段固定为预算一半的等距 pilot，若少于 3 个则取 3 个；q=4 时因此 pilot 用 3 个，剩 1 个自适应点。第二阶段区间权重为区间长度乘 (曲率²+epsilon²)^(1/5)，即曲率的 2/5 次方密度的区间积分近似；采用 largest-remainder 整数分配，区间内等距布置。所有方法用完整 4096 维重建误差评价。

入口与配置：scripts/v0_crb_mech_test.py、configs/rage/v0-crb-mech-test.json。输出写入 outputs/train/v0-crb-mech-test/，运行日志写入 logs/v0-crb-mech-test/。运行环境：AutoDL Conda rage；CPU 两线程，不占用 GPU；在 tmux 执行；成功后写 .complete。



## v0-crb-mech-test Uniform + Cancellation 分割（real-real oracle 诊断）

最后修改日期：2026-11-11。状态：完成。

目标：检验只依赖 Uniform 真实锚点端点的 cancellation 判据，能否通过增加虚拟分块边界改善 RAGE shape representativeness。64 条轨迹、4096 维、4000 点、种子 20261111 和频率/粗糙度生成参数沿用主批次；q={4,8,16,32}。

每个相邻 Uniform 锚点区间计算 kappa=||g_a+g_b||/(||g_a||+||g_b||)，零分母时定义为 1。若 kappa<tau，则仅增加一个虚拟切分点；其梯度值由端点线性插值得到，不增加真实梯度计算。阈值 tau={0,0.2,0.4,0.6,0.8,0.9,1.0}，tau=0 为无分割基线。对比 midpoint 与 minimum_norm 两种位置。Minimum-norm 使用 clamp(-g_a dot (g_b-g_a)/||g_b-g_a||^2,0,1)，并把分割索引限制为父区间两侧至少各有 max(2,ceil(0.1*width)) 点；宽度不足时取不超过 floor(width/2) 的可行长度。

每个子块用项目 RAGE v7 shape_rows 特征 [midpoint,slope/sqrt(12)]/sqrt(energy)，shape score 使用已有 linearSum 加权余弦。为避免待评轨迹泄漏到参考库，按轨迹序号分两折，参考集使用另一半轨迹的 Uniform q=32 区间；交换两折后评估。参考权重按区间覆盖跨度归一化，source 子块分数按其覆盖跨度加权。此指标表示相对固定 q=32 Uniform 参考形状分布的相似度，不等同于逐点重建 L2。

应用户补充要求，主要终点评估改为块内代表性。对原始 mock 梯度点按每个 Uniform 区间或虚拟子区间分块，以块内均值为描述性代表，沿用 scripts/v0_crb_mech_block_variance.py 的 population vector L2 variance、relative vector variance、direction variance、norm variance。每条轨迹逐 q/tau/rule 与未分割 Uniform 块配对；分别报告等块权均值和按块覆盖点数加权均值。块内方差与方向方差均按原分析定义计算，块均值使用区间内所有真实 mock 点，只用于代表性分析，不表示运行时免费获得块内真实梯度。新增入口 scripts/v0_crb_mech_cancellation_variance.py，输出 3,584 条轨迹级与 58,288 条子块级记录；前缀和实现与原 block_metrics 数值差在 5.5e-13 以内。

入口 scripts/v0_crb_mech_cancellation_split.py；配置 configs/rage/v0-crb-mech-cancellation-split.json；结果 outputs/train/v0-crb-mech-test/64-smooth-cancellation-split-diagnostic/；日志 logs/v0-crb-mech-test/。输出含逐轨迹汇总、逐区间 cancellation/切分/子块分数以及 .complete。计算采用 AutoDL Conda rage、CPU 两线程，在 tmux 会话 v0-crb-mech-cancellation-split 执行。



## v0-crb-mech-test Uniform Linear 六方法对照（real-real oracle 诊断）

最后修改日期：2026-11-11。状态：完成。

目标：严格固定原 Uniform 的真实锚点与分段线性重建，只改变块分析用的虚拟分割边界。各方法的逐点线性重建及其误差因此与原 Uniform+Linear 相同；新终点只比较块内 vector L2 variance、relative vector variance、direction variance、norm variance。

同一 64 条轨迹、4096 维、长度 4000、seed=20261111；q={4,8,16,32}；阈值 tau={0,0.2,0.4,0.6,0.8,0.9,1.0}。Cancellation 使用 kappa=||g_a+g_b||/(||g_a||+||g_b||)，零分母取 1；当 kappa<=tau 时触发。Minimum-norm 反转点按 clamp(-g_a dot (g_b-g_a)/||g_b-g_a||^2,0,1) 计算，切分点按原预注册规则限制父区间两侧长度至少 10%、至少 2 点，且不超过区间可行长度。

| 方法 | 选区间 | 分割位置 |
|---|---|---|
| tau-mid | kappa<=tau | midpoint |
| tau-rev | kappa<=tau | minimum-norm reversal |
| tau-mid-base | 每轨迹/q/tau 随机抽取与 tau-mid 相同 K 个区间 | midpoint |
| tau-rev-base | 与 tau-mid-base 共用同一随机 K 区间 | 每个随机入选区间自己的 minimum-norm reversal |
| all-mid | 所有 Uniform 区间 | midpoint |
| all-rev | 所有 Uniform 区间 | 每个区间自己的 minimum-norm reversal |

随机基线不放回抽样；seed 为 20261111+918273，并按轨迹/q/tau 派生可复现随机数。这样对比 tau 方案与 random-base 时，切分数和位置规则匹配，唯一变化是选择哪些父区间。All 方法不依赖 tau，所有相邻 Uniform 锚点区间均切一次。

分块后以块内全部原始 mock 点的向量均值作为描述性代表，按项目原 block variance 定义计算指标。逐轨迹与原始未切分 Uniform 块配对；分别提供等块权与按覆盖点数加权汇总。块均值属于离线分析量，不是可额外获得的真实梯度。入口 scripts/v0_crb_mech_cancellation_six_methods.py；配置 configs/rage/v0-crb-mech-cancellation-six-methods.json；输出 outputs/train/v0-crb-mech-test/64-smooth-cancellation-six-methods/；日志 logs/v0-crb-mech-test/。远端 Conda rage，CPU 两线程，tmux v0-crb-mech-cancellation-six-methods。结果：outputs/train/v0-crb-mech-test/64-smooth-cancellation-six-methods/ 中 .complete 已核实；7,680 条轨迹级记录与 107,520 条父区间记录完整。



## v0-crb-mech-test Uniform Linear 六方法 est-real 复评

最后修改日期：2026-11-11。状态：完成。

纠正前一批次的指标口径：上一批以块内真实均值代表真实值，命名为 real-real 方差估计，属于 oracle 诊断。当前 est-real 批次重用同一份仅由 Uniform 锚点生成的六方法区间分割；每个子块的代表值是原 Uniform+Linear 在该块每个采样点的估计值的算术均值。块内真实值不参与代表值构造或分割选择，只在最终评估中用于计算真实 L2 误差、相对 L2 误差和方向余弦误差。此批不增加梯度、不改变 Uniform anchor set、不改变逐点 Linear 重建。

入口 scripts/v0_crb_mech_cancellation_estimated_rep.py；配置 configs/rage/v0-crb-mech-cancellation-six-methods.json；区间分割来自 outputs/train/v0-crb-mech-test/64-smooth-cancellation-six-methods/interval_selection.csv；输出 outputs/train/v0-crb-mech-test/64-smooth-cancellation-six-methods/est-real/；日志 logs/v0-crb-mech-test/；tmux v0-crb-mech-cancellation-est-real。完成 64 条轨迹、q={4,8,16,32}、六方法；7,680 条轨迹级与 130,912 条块级记录，.complete 已核验。按点数加权 tau=0.9 时 cancellation 选区的 L2/方向误差均低于匹配随机基线；等块权结果方向不同，详见报告。



## v0-crb-mech-test Uniform 梯度重建的最优虚拟分块策略

最后修改日期：2026-11-11。状态：完成。

目标：固定 64 条既有 4096D 轨迹、Uniform 锚点及原 Linear 重建，比较不同 anchor-only 区间排名在相同 K 下对真实 est-real 块代表误差的预测/降低能力。q={4,8,16,32}，每个 q 的原始区间数 m=q-1；K=0,...,m。真实锚点位置、重建器和真实梯度计算预算不变；虚拟切分一律取离散 Midpoint，不计算新梯度。

代表定义：每个块的代表向量是原始 Uniform piecewise-linear 重建在该块所有离散时刻的向量均值。真实轨迹只用于计算代表相对真实点的 absolute vector L2 error、relative L2 error、direction error；并用于独立 Oracle 与 interval-gain 诊断。主指标按覆盖点数加权，等块权为辅助。原区间采样点采用不重叠归属：非末区间为 [a,b)，末区间为 [a,b]。

| 方法 | 每轨迹/q 决策 |
|---|---|
| Random-K | 从 m 个区间无放回随机取 K；100 个固定随机重复 |
| Cancellation-K | 计算端点 kappa，按 kappa 升序取前 K；零分母按 1 |
| Endpoint-Difference-K | 按 ||g_b-g_a||^2 降序取前 K |
| Predicted-Gain-K | 按离散精确的线性代表方差下降系数乘 ||g_b-g_a||^2 降序；该系数趋近 n_I/16 |
| L2-Oracle-K | 点加权曲线按真实 est-real L2 单区间 midpoint split 总收益降序；等块权曲线按父块与两个子块的块均值误差下降排序，均固定取 K |
| Direction-Oracle-K | 点加权曲线按真实方向误差总收益降序；等块权曲线按父块与两个子块的块均值方向误差下降排序，均固定取 K |

Oracle 对每个指标和汇总权重分别排序，作为对应曲线的上限。真实单区间收益即使为负，也保持 K 并继续按排序选择。预测分数与实际单区间收益按 q 及跨 q pooled 计算 Spearman。Selection Efficiency 为相对 K=0 的总体误差下降之比，分母接近零时记为未定义。64 条轨迹为统计单位；非随机方法曲线、方法间差值使用 paired bootstrap 95% CI；Random 每轨迹/q/K 使用 100 次随机抽样，报告重复均值、标准差、区间及对轨迹 paired bootstrap。所有 CI 和随机种子/抽样口径进入结果产物。

产物包括 8 张主指标 K–Error 曲线图（4 q × L2/Direction）、逐 q 表格、效率表、Spearman 表、逐轨迹曲线和随机重复数据。 已补充分解子批次：E_total=E_repr+E_estim，按点加权与轨迹等权两种方式汇总（轨迹等长，数值相同），覆盖所有方法/q/K；逐轨迹最大身份残差 8.99e-15，12 张 MSE 曲线见 docs/results/v0-crb-mech-test.md。入口 scripts/v0_crb_mech_virtual_split_selection.py；配置 configs/rage/v0-crb-mech-virtual-split-selection.json；输出 outputs/train/v0-crb-mech-test/64-smooth-virtual-split-selection-k-error/；日志 logs/v0-crb-mech-test/。远端 Conda rage、CPU 两线程、tmux v0-crb-mech-virtual-split-selection。


#### 64-large-var-k-error-composition

批次完成（2026-11-11）。64 条 4096D×4000 mock；相邻点完整向量差与两点平均模长之比目标约 0.5，实测轨迹均值 0.50005；lag-16 方向余弦目标为 0，实测均值 0.0000375。模长乘宽幅平滑包络（CV=0.3225）。固定 Uniform anchors + Linear，q={4,8,16,32,64,128}，midpoint 虚拟分块 K=0..q−1；Random-K 100 次、paired bootstrap 5000 次。所有方法、point-weighted 与 equal-block、6 个 q 的 K 曲线和 Representation/Estimation/Total MSE 已完成。结论：Direction 随切分数下降；L2 在 q≤64 的全切分下上升，q=128 才略降。Cancellation K=1 对 L2 没有稳定胜过 Random；端点差异与 Predicted-Gain 在本批也未给出一致优势。完整参数、CI、区间相关、选择效率及图表索引见 `docs/results/64-large-var-k-error-composition.md`。输出 `.complete` 已核验。
