## 2026-09-17 最新方法决策：参考意见冲突与概率构建（reference-opinion-v1）

**优先级与历史边界。** 本节记录用户本轮讨论与变更，覆盖下文旧协议中的“逐参考质量固定”“Global仅为消融”“LESS全部参考线性求和”“T=0.25默认”“RAGE-Traj重建窗口长度加权均值”等现行性表述；旧正文与伪代码保留作历史，不删除。当前冻结资产来自V3：C=512/H=64/D=1、投影4096/seed42、CRB密度2%/1%/1%、closest_zero_v2批量分裂、预算用尽且全部合法窗口保留。所有使用采样温度的默认方法为T005=0.05；T025历史保留原身份。文档变更不表示启动新训练。

### A. 为什么增加本项消融

细粒度梯度估计、意见聚合、采样集中程度是不同问题。Main对每个源块分别获得全部参考块的余弦意见，但有符号线性汇总会让不同模式相互抵消；高匹配区域可能在进入Softmax前已经失去优势。增加评委数量本身不造成稀释：将同向块拆分并相应拆分其权重，评分不变。影响来自方向异质性及其权重，而非块数。

当前线性Main可写为s_j=unit(g(s_j))·Q，Q=Σ_i alpha_i unit(g(r_i))。Q只是代数汇总，不是方法中的源轨迹评分。参考方向分散可缩小Q范数，即使各局部原始梯度模长相同也成立。Traj则先对实际CRB锚点原始梯度取FP32算术平均，再以余弦单位化轨迹代表；它可能强调很弱的净方向，也不解决不同参考轨迹之间的抵消。当前Traj不是按重建窗口贡献加权的平均，旧定义降为历史。

已有T005 NEM seed0：Chronos rand/Main/Traj MASE=1.413860/1.441458/1.245335，TimesFM=.726650/.713524/.689784。Main/Traj参考汇总范数分别.225068/.882898和.158771/.710910；两者汇总方向余弦.994828/.859855。支持“当前聚合可能削弱可用选择信号”的假设，不证明参考权重不合理，也不证明保守或线性方法普遍无益。表示、平均测度与源竞争单位尚未隔离。TimeMoE共同训练退化单列健康审计，不用其排序支持主机制。

统一平移不改变最终Softmax；统一正缩放a等价于温度T/a。不同专家分别缩放会改变话语权，负分截断/max/指数汇总还会改变源候选排序，不仅是温度变化。分布集中度相同也可能选择不同数据。Main的近块均匀不等于rand的合法窗口均匀。

### B. 统一符号和四种Global聚合

i索引全部参考单位，j索引全部源单位。轨迹版本两侧单位都是轨迹；块版本两侧单位都是块。c_ij=cos(g(r_i),g(s_j))。令N_ref为参考轨迹数，避免与源池窗口当量N混淆。

- 轨迹参考权重：alpha_i=1/N_ref。
- 块参考权重：alpha_i=(1/N_ref)×(参考块拥有合法起点数/所属参考轨迹全部合法起点数)。所有参考权重之和为1，分母不因CRB改变。
- 轨迹等权是对下游Benchmark按轨迹评估的适配，不主张真实部署应忽略时间频次；真实场景的长度加权属于另一目标先验。

四种源单位评分明确为：

$$s_j^{\mathrm{linearSum}}=\sum_i\alpha_i c_{ij},$$
$$s_j^{\mathrm{zeroClipSum}}=\sum_i\alpha_i\max(c_{ij},0),$$
$$s_j^{\mathrm{maxSum}}=\max_i c_{ij},$$
$$s_j^{\mathrm{expSum}}=\sum_i\alpha_i\exp(c_{ij}).$$

全部Global版本随后只对源单位做一次Softmax：p_j=exp(s_j/T)/Σ_k exp(s_k/T)，默认T=.05。没有源长度乘子、没有Softmax后参考权重、没有总分正值门控；抽中源单位后在其拥有合法窗口中均匀采样。

**maxSum不乘alpha再取max**，也不额外乘1/N_ref：专家平等竞争取最高意见，块占比不再作为分数乘子。这是主动放弃比例先验的变体；max(alpha_i c_ij)会混淆比例和匹配，不是本轮定义。max不是零门控，全负时最高分仍可负。

**zeroClipSum是子分数截断，不是源支持截断**。负意见不贡献也不抵扣；源块全部非正时s_j=0，经Softmax仍有非零质量；全部源块零分则单位均匀。它允许对部分参考有害的源块因其他参考正支持被选择，不保证净收益。

**expSum按用户本轮字面定义为exp(raw cosine)→加权求和→外层全局Softmax**，内层尺度固定1。不得实现成Σalpha_i exp(c_ij/T)直接归一化，也不得替换成log-sum-exp评分；这两者属于另一种kernelSum/LSE设计，尚未纳入四组件。本轮expSum内层exp使负值仍有较小正贡献，没有保留负贡献抵扣，且不保证比linearSum更集中。

### C. 正式命名与主方法选择

| 方法名称 | 表示/锚点 | 默认聚合 | 决策理由 |
|---|---|---|---|
| LESS-Main（slug less-main） | 均匀rounded-linspace锚点原始梯度均值；轨迹级 | maxSum / global | 尊重原LESS跨子任务取最大影响的意见合并方式；仍披露时间序列采样与Softmax适配 |
| LESS-linearSum（slug less-linearSum） | 同LESS-Main | linearSum / global | 保留原项目V3轨迹等权线性聚合；名称中的Sum是加权和，权重总和1，不恢复未经1/N_ref缩放的历史求和 |
| RAGE-Main（slug rage-main） | CRB锚点；线性插值中点块方向 | zeroClipSum / global | 本轮暂定设计主项：保留两级权重，最小改动隔离负意见抵扣；不是根据测试集成绩优选 |
| RAGE-Traj（slug rage-traj） | CRB实际锚点原始梯度均值；轨迹级 | zeroClipSum / global | 与新Main使用相同意见规则，形成粒度对照 |

显式RAGE变体：rage-main-{linearSum,zeroClipSum,maxSum,expSum}与rage-traj-{linearSum,zeroClipSum,maxSum,expSum}。可增加-global/-local后缀；不写后缀默认global。LESS同样支持四组件，主比较保留less-main和less-linearSum。rage-uniform使用同规则、只替换误差阶段锚点；实际K必须配对。

LESS原论文§4.2对子任务形成代表，跨warmup检查点累积影响，跨子任务取max，再全局Top-k；官方matching.py为子任务内样例影响mean、子任务间max。两者内部平均写法不同。新LESS-Main只恢复跨参考max原则，不宣称完整复现原文LoRA/Adam多检查点及Top-k。参考轨迹作为子任务专家是本项目适配，轨迹未必等于语义模式。原论文：[§4.2](https://arxiv.org/html/2402.04333v3#S4.SS2)，[官方实现](https://github.com/princeton-nlp/LESS/blob/main/less/data_selection/matching.py)。

### D. Global/Local：归一化约束，不是另一套匹配算法

同一配对核K_ij>0，Z_i=Σ_j K_ij：

$$p_j^{local}=\sum_i\alpha_i K_{ij}/Z_i,$$
$$p_j^{globalKernel}=\frac{\sum_i\alpha_i K_{ij}}{\sum_i\alpha_i Z_i}.$$

Local等价于GlobalKernel使用有效参考先验alpha_i/Z_i；所以预算可理解为一个参考偏置，但两者原始alpha相同并不保证分布相同。Local保证每个参考行贡献alpha_i，GlobalKernel允许匹配强弱决定其实际质量。注意GlobalKernel不是本节linearSum-global；后者是exp(Σalpha_i c_ij/T)，指数位置也不同。

本轮可执行local精确定义：每位专家使用psi(c_ij)生成一行Softmax，再按alpha_i混合；psi分别为c、max(c,0)、c、exp(c)。无源长度乘子，不增加旧门控。maxSum-local退化为linearSum-local，因为每行每个源只有一个子分数，没有跨专家max；不得作为独立八格之一重复训练。若先跨专家取max再local，专家入口已经丢失，也不是原Local。

### E. 尚未被两轴覆盖的已有消融

核心另外两轴：表示粒度block/trajectory（RAGE-Traj），锚点采集adaptive/uniform（RAGE-Uniform，仅第二阶段改均匀、保留分裂、实际K配对）。还有源基础采样测度（块等权、轨迹等权、合法窗口均匀/长度先验）、范数与聚合顺序、温度/数值vs排名映射、正对齐硬门控；与zeroClip子分数截断不能混同。

历史代码保留random_cover、no_error、no_split_detect、drop_only、fixed_blocks、window_return、uniform_segments和旧grid。前三类分别检验覆盖、风险分配、分裂；后续涉及表示/支持。drop_only及旧检测网格与全部窗口保留冲突，window_return/fixed_blocks等旧Top-k部署不能直接当现V3消融，必须先适配。密度/投影/head-only/探针检查点/长上下文属于既有后续或敏感性计划，不自动启动扫描。
