## 2026-09-18 V4正式启动：固定三seed、V3归档与巡检

最新决策覆盖前一版不限seed数的计划：V4固定仅使用seed0、seed1、seed2，总共三轮。轮间串行，当前轮两个模型×九任务×12训练方法全部完成训练与评估后才开始下一seed；轮内Chronos Base和TimesFM 2.5并行，TimeMoE暂停。13方法及聚合公式不变，外层Softmax T=.05；Exp内层T=1.00、先exp再双层加权，Max保留加权后择优。新seed不是新源池，pool_seed0和projection_seed42仍固定。

总规模666个独立结果项=18个orig基线+216训练评估项×3seed。orig首轮在训练前测量，后续引用；静态选择概率和已有字段校验梯度缓存复用，各seed独立训练/抽样与checkpoint，不复用训练成绩。父协议seeds=[0,1,2]，子轮seeds=[K]，seed_schedule.stop=3，默认第三轮结束后停止，禁止自动进入seed3。

V3已完成189/189。已整理全部既有输出目录，归档746个cell JSON及配置、文档、日志、run元数据，生成完整all_cells.jsonl与按各原run/model/task相对rand改进降序的all_results_ranked.csv。归档位于/root/autodl-tmp/TS3/outputs/archives/results-through-v3-20260918，含results-docs-configs.tar.gz及明确字段inventory.json，无hash。原结果、选集、梯度缓存及保留checkpoint留在原路径；大梯度不重复复制。最终V3报告另存docs/archive/MAIN-v3-final-20260918.md。

归档后停止已结束的V3 scheduler/report loop，清理其过期PID文件及可丢弃报告partial，不删除正式成果或V4依赖材料。V3 execution_control切换archived，防止旧守护误续跑。V4已按用户授权真实启动：configs/rags/main_v4.json；run=/root/autodl-tmp/TS3/outputs/main-v4-three-seeds-20260918；父入口rags.experiments.v4_seed_rounds，先推进seed0。恢复仍须用父入口，不能绕过跨模型seed屏障。

MAIN-v4.md直接复用MAIN-v3的报告生成模板，保持任务分节、模型表、状态/巡检、末尾汇总与固定方法说明的版式，只更换内容及版本标识。所有结果表（任务模型表、末尾聚合汇总表）均按相对rand改进降序排序，正值更好，缺数据/缺rand基准列末；不是按绝对MASE或方法名排序。末尾汇总按方法聚合seed，orig仅计一次，不把基线引用算成新实验。三seed汇总包含均值、样本标准差及逐seed胜rand配对信息。

79项相关回归通过，覆盖固定三seed止步、全矩阵屏障、失败/续跑、基线去重、表格降序和同模板报告、旧V3兼容。已建立本任务每30分钟巡检，并维护远程30分钟报告更新；正常进度保持安静，整轮/全批完成或明显异常时通知。本机不关机，已启用保持空闲唤醒，远程独立运行；不执行关机操作。当前实验进行中，尚不对V4效果作结论；未提交/Push。

---
历史记录：

## 2026-09-18 V4范围与编排：两模型、自然数seed逐轮串行

本节为最新实验范围与编排决策，覆盖此前“三模型、只做seed0”的新意见聚合矩阵规模。聚合公式不变：RAGS Linear/ZeroClip/Max先按双层alpha加权，Exp先exp(c/1.00)再按alpha加权加和，外层Softmax .05；LESS-Main保留原始余弦Max。13方法、九任务不变，不指定任何RAGS主方法。

V4暂时只做chronos-base和timesfm-2.5。TimeMoE-200M因当前微调困难退出V4，不运行它的准备、训练或评估；历史结果、材料和实现保留。替代模型尚未确定，本轮不自行选择或加入新模型。此前正在运行的V3批次沿其原冻结配置处理，本次不停止或修改历史批次。

编排制度：seed固定自然数0、1、2、3、4……，从0开始逐轮串行；必须先完成同一seed下两个模型×九任务×全部训练方法的训练和评估，再放行seed+1。并行放在当前seed的一轮内部，两模型并行，不能让较快模型提前进入下个seed。失败、缺项、身份不符或非有限MASE都会阻止跨轮；续跑从最早未完成的轮次继续，已完成项按原有续跑逻辑跳过。父套件使用单例锁，避免两个编排进程分别推进不同seed。

没有指定总seed数，不擅自把01234限定为五轮。默认持续自然数序列；--max-seeds只提供可选的运行次数上限，不是当前科学协议的一部分。各轮位于独立rounds/seedK目录，配置seeds只包含[K]，训练checkpoint、结果和seed身份独立。

第一轮为234格=18 orig基线+216训练/评估格（两个模型×九任务×12训练方法）。orig与训练seed无关，在第一轮训练前测量，后续只引用首次测量结果。每增加一个seed，新增216训练/评估格；S个seed的唯一测量总数为18+216×S。每轮的完整展示仍包含18个基线引用。原351格规模是旧三模型seed0设计，不能再作为V4规模。

训练seed控制PyTorch随机数、CUDA随机数与训练时源窗口抽样。源池pool_seed=0、4096维投影seed42、固定参考、梯度材料及静态选择概率保持不变；后续seed复用已完成seed0的静态概率选集，并采用自己的训练/抽样seed，不是重复引用训练结果。已有字段校验梯度缓存按轮链接复用，不重新为每个seed计算梯度，不新增TimeMoE梯度任务；现有缓存的单写者锁仍有效。

远程配置：configs/rags/main_v4.json。入口：python -m rags.experiments.v4_seed_rounds --config configs/rags/main_v4.json --run <数据盘V4父目录>。--dry-run仅打印计划，不加载模型、不启动任务。正式入口须遵守父目录execution_control.json运行开关；本次修改与验证不启动V4实验。父state.json记录当前seed、完成seed及下一seed；各轮结果沿用原cell JSON/selection NPZ格式，无hash，协议与字段明确校验、JSON原子发布。

验证状态：77项相关回归测试通过（3条既有Swig弃用警告）。验证了跨模型轮次屏障、自然数seed配置、失败/缺项阻止下一轮、续跑、静态材料与orig复用、实际seed汇总及旧V3兼容。编排dry-run为两模型、首轮234格、后续每seed新增216格、jobs_launched=0。本次未启动V4实验，未提交/Push。

---

历史设计（V4范围与seed编排以上述09-18决策为准）：

## 2026-09-18 当前聚合协议：Linear/ZeroClip/Max先加权，Exp后加权

本节是现行定义，覆盖2026-09-17的“所有变体先加权”方案；历史推导、配置和结果保留，不覆盖或冒充新结果。仍为13方法、九任务、三模型、seed0，共351 cells；八个RAGS变体均中立，得到结果之前不指定RAGS主方法。本轮只更新准备材料，未启动新批次。

参考专家i与源对象j的原始余弦为c_ij。Block双层权重alpha_i=(1/N_ref)×该块拥有的合法起点数/所属参考轨迹全部合法起点数；Traj内部占比为1，即alpha_i=1/N_ref。权重总和为1，不乘源对象长度，不对最终概率再次加权。

| RAGS变体 | 外层Softmax前得分 |
|---|---|
| linearSum | S_j=Σ_i alpha_i c_ij |
| zeroClipSum | S_j=Σ_i ReLU(alpha_i c_ij)=Σ_i alpha_i ReLU(c_ij) |
| maxSum | S_j=max_i(alpha_i c_ij)，不相加 |
| expSum | S_j=Σ_i alpha_i exp(c_ij/1.00)，先exp，再双层加权求和 |

全部最后统一p_j=Softmax_j(S_j/0.05)。Exp内层温度明确固定T_inner=1.00；外层温度T_outer=0.05。不得将内层偷偷改成0.05，不得改成log-sum-exp或指数混合后直接归一化。配置和审计显式记录exp_inner_temperature=1.00、weight_placement=after_transform；其余RAGS三变体为before_transform。除Exp外，其他规则不变。

选择理由：Linear与ZeroClip保留参考内容占比；Max保留加权后择优，既尊重比例也允许长块影响胜者。Traj等权Max的1/N不改变胜者/排序，但会改变固定温度下的概率集中度，不能说概率完全不受影响。Block专家很多时原始Max可能接近饱和，各源对象最大值趋同；加权Max也可能因块权重小而缩小分差，本轮把它视作待验证变体。

Exp先加权后指数的旧式Σexp(alpha*c)，在alpha小时近似M_ref+Σalpha*c，外层Softmax消掉共同M_ref后接近linearSum。现改为Σalpha*exp(c)，alpha控制意见测度，指数负责非线性评价。同方向块细分且总权重保持不变时，新Exp得分不变。其展开为1+Σalpha*c+(1/2)Σalpha*c²+…：外层Softmax消掉共同常数1，但高阶项不再被alpha平方压小；内层T=1仍可能较温和，是否提高性能必须等待新实验，不能声称已证明优于Linear。

LESS版本边界：正在运行的旧V3 LESS是(1/N_ref)Σ_i c_ij，内核求和后在调用层除以len(reference)，不是未归一化总和。已准备的新LESS-Main仍保留此前指定的原始余弦Max，即max_i c_ij，作为原论文多子任务Max原则的适配基线。本次不更改该基线。若表示相同，RAGS-Traj-maxSum分数为LESS-Main的1/N_ref，固定T=.05下选择强度不同。新旧LESS主要变化为Mean→Max，不是新增1/N；旧线性结果展示为LESS-linearSum。

执行配置仍为configs/rags/matrix_reference_opinion_20260917.json，聚合协议更新为reference-opinion-v4-mixed-weight-order（聚合版本号，不表示已启动V4实验）。上一版保存为configs/rags/matrix_reference_opinion_v3_weighted_first_20260917.json，状态superseded_not_launched，继续通过校验。已有V3运行协议与缓存身份保留；只为Exp切换权重位置，Max/Linear/ZeroClip语义不变。没有新梯度准备、训练或评估启动，未提交/Push。

验证：99项相关回归测试通过（3条既有Swig弃用警告）。覆盖精确公式、内层T=1/外层T=.05、同向参考块细分不变性、Block/Traj生产入口、历史V3与上一版配置保护；未启动新批次。

---

历史记录（以下旧定义由09-18当前协议覆盖）：

### 2026-09-17 精确定义补充：RAGS先双重加权，再处理专家意见

本轮用户明确修正Max/Exp的权重顺序；本段覆盖此前“max不乘alpha”“exp后乘alpha”的RAGS定义。参考专家i对源对象j的余弦为c_ij，先得到q_ij=alpha_i c_ij。Block参考alpha=(1/N_ref)×参考块合法起点占所属参考轨迹比例；Traj参考alpha=1/N_ref（内部长度占比为1）。不乘源对象长度。

| RAGS聚合 | Softmax前源对象评分 |
|---|---|
| linearSum | s_j=Σ_i q_ij |
| zeroClipSum | s_j=Σ_i ReLU(q_ij) |
| maxSum | s_j=max_i q_ij，不相加 |
| expSum | s_j=Σ_i exp(q_ij)，exp之后不再乘alpha |

四者最后统一p_j=Softmax_j(s_j/.05)。ZeroClip由于alpha>0，与Σalpha_i ReLU(c_ij)等价；Max与Exp不等价于上一版原始余弦max和Σalpha_i exp(c_ij)。RAGS所有八臂仍为中立变体，不选主方法；13方法/九任务/三模型/seed0、351cells及其余配方不变。

**LESS-Main基线例外。** LESS-Main保留上一轮用户要求的原论文跨参考max原则：max_i c_ij，不在max前乘1/N_ref。本段是对RAGS四聚合变体的定义，不将原生LESS的max悄悄改为加权max。若锚点及代表相同，RAGS-Traj-maxSum得分恰为LESS-Main得分/N_ref；同配置T=.05对应相对LESS的有效温度.05×N_ref，不能宣称两者仅差锚点、选择强度相同。

**尺度与粒度的直接推论。** 加权Max可能选择权重更大但余弦较低的专家；参考块alpha小会压小源候选间分差。同向块细分虽然不改变linear/zeroClip评分，但会改变加权max及exp评分。因此“所有组件都不受块细分影响”不成立。

expSum可展开为M_ref+Σq_ij+(1/2)Σq_ij²+…，M_ref是参考专家总数，为所有源对象共同常数，最终Softmax消掉。参考块alpha很小时，expSum近似linearSum，非线性奖励可能很弱；负意见仍通过一阶项抵消正意见。不能沿用“指数汇总一定有效避免抵消”的论断。实现用Σexpm1(q_ij)进行Softmax，严格等价于Σexp(q_ij)减共同常数M_ref，避免大量接近1的项淹没数值精度；审计同时记录原始s_j、Softmax用中心分数与共同常数。

当前配置协议更新为reference-opinion-v3-weighted-first，RAGS metadata记录weight_placement=before_transform；LESS记录after_transform且Max输入raw_cosine。上一版中立配置保留为configs/rags/matrix_reference_opinion_v2_neutral_20260917.json（superseded_not_launched），不同顺序概率版本隔离。既有正在运行的V3线性protocol显式保持旧顺序，线性语义不变。本轮不启动新批次。

验证：94项相关回归及1项LESS/加权Traj-Max尺度验证通过；旧V3冻结线性批次不变，新批次尚未启动，未提交/Push。

---

## 上一版定义（权重顺序以本页最新段为准）

## 2026-09-17 当前实验矩阵：Block/Traj × 四种意见聚合（中立比较）

**本节覆盖前一轮“暂定RAGS-Main为zeroClipSum”的方案，该暂定主项撤回。** 在得到实验结果之前，不将任何RAGS变体宣称为主方法；八个变体地位相同。LESS-Main仍采用参考轨迹间max合并。历史线性run、已有结果与之前未启动的方案均保留身份，不改名冒充新规则。

| 展示方法 | 执行名称 | 聚合 |
|---|---|---|
| rand | rand | 合法窗口均匀 |
| orig | orig | 不训练 |
| LESS-Main | less-main | maxSum |
| RAGS-Block-linearSum | rags-block-linearSum | 加权有符号和 |
| RAGS-Block-zeroClipSum | rags-block-zeroClipSum | 子分数截零后加权和 |
| RAGS-Block-maxSum | rags-block-maxSum | 参考块余弦最大值 |
| RAGS-Block-expSum | rags-block-expSum | 子分数exp后加权和 |
| RAGS-Traj-linearSum | rags-traj-linearSum | 加权有符号和 |
| RAGS-Traj-zeroClipSum | rags-traj-zeroClipSum | 子分数截零后加权和 |
| RAGS-Traj-maxSum | rags-traj-maxSum | 参考轨迹余弦最大值 |
| RAGS-Traj-expSum | rags-traj-expSum | 子分数exp后加权和 |
| TSDS | tsds | 原生分布 |
| LP-Mirror | lp-mirror | 原生proxy评分→Softmax |

共13方法，沿用九任务×三模型×seed0，完整矩阵351 cells=27 orig+324训练格。所有涉及采样Softmax的臂统一T005=.05；orig/rand/tsds没有该温度。统一使用Global竞争，不追加Local、LESS-linearSum、Uniform、覆盖/分裂或块构造臂；这些只留历史/候选，不是已完成的研究成果。

本轮消融研究主轴只有表示粒度（Block/Traj）和参考意见聚合（linearSum/zeroClipSum/maxSum/expSum）。其余固定：C512/H64/D1、模型/池/参考/训练配方、4096维投影seed42、CRB 2%/1%/1%、批量closest_zero_v2预算用尽且全窗保留。Block用线性插值中点代表；Traj用实际不同CRB锚点原始梯度FP32算术均值后余弦；源单位内部均匀抽合法窗，无额外源长度乘子。

Sum组件参考轨迹权重1/N_ref，参考块权重为(1/N_ref)×所属轨迹合法起点占比；max直接比较余弦、不乘alpha。zeroClip不把源零评分概率强制为零。expSum明确为exp(raw cosine)→加权和→外层Softmax(.05)，内层尺度1；不替换为LSE或exp(c/.05)直接归一化。用户书写maxsum统一记maxSum，仅名称大小写规范。

新配置 `configs/rags/matrix_reference_opinion_20260917.json`，协议 `reference-opinion-v2-neutral`，显式rags_main_method=null、rags_main_method_selected=false；校验拒绝任何主方法声明或矩阵增删。前一轮main_reference_opinion_20260917/ablation_reference_opinion_20260917标为superseded_not_launched，不删除。新矩阵定义与执行入口已接入，当前尚未启动此351格批次；在跑V3线性批次继续按其冻结协议，不静默切换。得到结果也应区分经验排序和独立验证，不能将旧结果作为新规则证据。

本轮验证：87项相关回归通过，包含GPU配对一致性、中立13方法校验及旧V3冻结协议保护；未提交/Push。

---

## 前一轮方案（被上述中立矩阵替代）

## 2026-09-17 最新方法决策：参考意见冲突与概率构建（reference-opinion-v1）

**优先级与历史边界。** 本节记录用户本轮讨论与变更，覆盖下文旧协议中的“逐参考质量固定”“Global仅为消融”“LESS全部参考线性求和”“T=0.25默认”“RAGS-Traj重建窗口长度加权均值”等现行性表述；旧正文与伪代码保留作历史，不删除。当前冻结资产来自V3：C=512/H=64/D=1、投影4096/seed42、CRB密度2%/1%/1%、closest_zero_v2批量分裂、预算用尽且全部合法窗口保留。所有使用采样温度的默认方法为T005=0.05；T025历史保留原身份。文档变更不表示启动新训练。

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
| RAGS-Main（slug rags-main） | CRB锚点；线性插值中点块方向 | zeroClipSum / global | 本轮暂定设计主项：保留两级权重，最小改动隔离负意见抵扣；不是根据测试集成绩优选 |
| RAGS-Traj（slug rags-traj） | CRB实际锚点原始梯度均值；轨迹级 | zeroClipSum / global | 与新Main使用相同意见规则，形成粒度对照 |

显式RAGS变体：rags-main-{linearSum,zeroClipSum,maxSum,expSum}与rags-traj-{linearSum,zeroClipSum,maxSum,expSum}。可增加-global/-local后缀；不写后缀默认global。LESS同样支持四组件，主比较保留less-main和less-linearSum。rags-uniform使用同规则、只替换误差阶段锚点；实际K必须配对。

LESS原论文§4.2对子任务形成代表，跨warmup检查点累积影响，跨子任务取max，再全局Top-k；官方matching.py为子任务内样例影响mean、子任务间max。两者内部平均写法不同。新LESS-Main只恢复跨参考max原则，不宣称完整复现原文LoRA/Adam多检查点及Top-k。参考轨迹作为子任务专家是本项目适配，轨迹未必等于语义模式。原论文：[§4.2](https://arxiv.org/html/2402.04333v3#S4.SS2)，[官方实现](https://github.com/princeton-nlp/LESS/blob/main/less/data_selection/matching.py)。

### D. Global/Local：归一化约束，不是另一套匹配算法

同一配对核K_ij>0，Z_i=Σ_j K_ij：

$$p_j^{local}=\sum_i\alpha_i K_{ij}/Z_i,$$
$$p_j^{globalKernel}=\frac{\sum_i\alpha_i K_{ij}}{\sum_i\alpha_i Z_i}.$$

Local等价于GlobalKernel使用有效参考先验alpha_i/Z_i；所以预算可理解为一个参考偏置，但两者原始alpha相同并不保证分布相同。Local保证每个参考行贡献alpha_i，GlobalKernel允许匹配强弱决定其实际质量。注意GlobalKernel不是本节linearSum-global；后者是exp(Σalpha_i c_ij/T)，指数位置也不同。

本轮可执行local精确定义：每位专家使用psi(c_ij)生成一行Softmax，再按alpha_i混合；psi分别为c、max(c,0)、c、exp(c)。无源长度乘子，不增加旧门控。maxSum-local退化为linearSum-local，因为每行每个源只有一个子分数，没有跨专家max；不得作为独立八格之一重复训练。若先跨专家取max再local，专家入口已经丢失，也不是原Local。

### E. 尚未被两轴覆盖的已有消融

核心另外两轴：表示粒度block/trajectory（RAGS-Traj），锚点采集adaptive/uniform（RAGS-Uniform，仅第二阶段改均匀、保留分裂、实际K配对）。还有源基础采样测度（块等权、轨迹等权、合法窗口均匀/长度先验）、范数与聚合顺序、温度/数值vs排名映射、正对齐硬门控；与zeroClip子分数截断不能混同。

历史代码保留random_cover、no_error、no_split_detect、drop_only、fixed_blocks、window_return、uniform_segments和旧grid。前三类分别检验覆盖、风险分配、分裂；后续涉及表示/支持。drop_only及旧检测网格与全部窗口保留冲突，window_return/fixed_blocks等旧Top-k部署不能直接当现V3消融，必须先适配。密度/投影/head-only/探针检查点/长上下文属于既有后续或敏感性计划，不自动启动扫描。
