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

## 2026-09-17 最新实验决策：专家冲突聚合消融（reference-opinion-v1）

**本节为本轮现行设计入口。** 覆盖下文v1/v2的默认温度、方法名、逐参考固定质量、三种子及RAGS-Traj重建贡献均值口径；历史正文、附录结果与旧决策全部保留。当前V3原始资产为C512/H64/D1、2%/1%/1% CRB、全部窗口保留、4096维投影seed42、三学生自身探针、仅seed0。四温度方法lp-mirror/LESS/RAGS默认T005=.05，原T025批次按历史身份留存。详情见[方法最新决策](reference_opinion_method_20260917.md)。

### 1. 问题、观察和结论边界

本项检验：细粒度参考方向已被识别，为什么不能改善选择？线性有符号意见混合是否在进入Softmax前抹掉局部高匹配？它与采样强度、参考先验以及源单位竞争如何区分？

已有T005 NEM seed0（MASE越低越好）：

| 学生 | rand | 旧RAGS-Main-linearSum | 旧RAGS-Traj-linearSum |
|---|---:|---:|---:|
| Chronos Base | 1.413860 | 1.441458 | 1.245335 |
| TimesFM 2.5 | .726650 | .713524 | .689784 |

参考汇总范数Main/Traj分别.225068/.882898和.158771/.710910；方向余弦.994828/.859855。固定Main源块，仅匹配参考信号尺度，源轨迹最高质量10%获得概率从31.94%/30.92%升至54.31%/55.12%，未做该反事实训练。这些数据支持聚合尺度假设，不证明线性聚合或比例权重不适合；Main仍在TimesFM优于rand。TimeMoE各方法共同退化单列训练健康/负迁移，不拿排序反转判定细粒度机制。

全局平移在聚合后为共同常数时无效；统一正缩放等价T变化且不改变排序。max/zeroClip/exp会改变排序，不仅控制集中度。评委数量增加不必压平，同向参考块按比例细分时加权和不变。当前Main概率的近块均匀与rand合法窗口均匀不是同一基线。

### 2. 名称和主方法迁移

| 现名称 | 本轮精确协议 | 旧产物如何解释 |
|---|---|---|
| less-main / LESS-Main | 轨迹余弦，参考间max，Global Softmax T005 | 已跑同名V3线性产物展示LESS-linearSum，保留原文件/slug/协议 |
| less-linearSum / LESS-linearSum | 轨迹参考1/N_ref加权线性和，Global Softmax T005 | 原项目LESS-Main评分的显式保留对照 |
| rags-main / RAGS-Main | 块级，zeroClipSum，Global，两级alpha，T005 | 已跑同名V3线性产物展示RAGS-Main-linearSum |
| rags-traj / RAGS-Traj | 轨迹级，zeroClipSum，Global，alpha=1/N_ref，T005 | 已跑同名V3线性产物展示RAGS-Traj-linearSum |

RAGS主项本轮暂定zeroClipSum：保留目标比例先验，只改变负意见抵扣，不由最终测试集成绩挑选。它不是已被验证更优的方法。LESS恢复原论文跨子任务max意图，但保留本项目轨迹代表和概率重采样适配，不称完全原生复现；原文用全局Top-k。不同方法比较锚点时必须匹配聚合规则，新less-main(max)与rags-traj(zeroClip)不再仅差锚点位置。

**历史不重命名实体、不覆盖结果。** 已运行V3仍按冻结reference-mean协议线性计算；新reference-opinion-v1配置及选集独立，不在同一run混入新旧协议。旧T025全部方法的历史展示后缀仍保留；无温度臂的后缀仅表示历史批次。未经新规则实际计算的旧cells不能改名冒充新主方法结果。

### 3. 聚合组件：先锁定公式再比较

c_ij为参考单位i与源单位j的余弦；两级alpha总和1。Global四评分：

| 后缀 | Softmax前源单位评分 | 在处理什么 |
|---|---|---|
| linearSum | Σalpha_i c_ij | 加权净意见，允许抵消 |
| zeroClipSum | Σalpha_i max(c_ij,0) | 累计加权正支持，不扣负意见 |
| maxSum | max_i c_ij | 唯最强匹配，不使用alpha分数乘子 |
| expSum | Σalpha_i exp(c_ij) | 连续强调强匹配，内层尺度固定1 |

四者外层均p=Softmax(s/.05)。max不是零门控；zeroClip也不排除零分源块。expSum按用户本轮定义包含外层Softmax，不等同于Σalpha exp(c/.05)直接归一化或LSE；不得在实现中偷换。max(alpha*c)不属本轮max。

先固定锚点、单位支持、参考/源代表、源基础测度和训练配方，只改变聚合；block与trajectory分别做四组件。基础候选矩阵为原七方法加less-linearSum、rags-main-{linearSum,maxSum,expSum}、rags-traj-{linearSum,maxSum,expSum}，共14名称；zeroClip显式别名与main不重复计算。本文件列出可执行候选，不自动授权扩成14×9×3训练矩阵。

主要端到端解释范围先聚焦正常训练的Chronos/TimesFM。任务和运行子矩阵必须在启动前登记，不根据最终测试集结果挑最有利子集。本轮保留seed0成本约束，不沿用下文旧三个种子。主项由设计理由预登记；开发任务可检查协议与集中度，不能将最终测试结果用于选择主聚合规则。

### 4. Global/Local是参考质量与归一化约束

同一配对核K，Z_i=Σ_j K_ij：local=Σalpha_i K_ij/Z_i；globalKernel=Σalpha_i K_ij/Σalpha_i Z_i。local可写成globalKernel使用alpha_i/Z_i的有效参考先验，说明固定预算是一种偏置；不说明同alpha产生同概率。当前linearSum-global还涉及先加分再exp，与同核全局归一化不同。

本轮local组件按每个专家的psi(c)→行Softmax→alpha混合定义，无源长度乘子。psi为c、max(c,0)、c、exp(c)。maxSum-local与linearSum-local同分布，不作为独立新训练。Global权重仅在评分汇总；Local权重在行概率混合，二者语义必须随metadata记录。无门控Local中无好匹配的专家仍分配固定质量，zeroClip后全零行仍单位均匀。

比较Global/Local须冻结同一配对核、基础测度；用参考行实际贡献、Z_i和有效先验报告预算偏置。不把“四聚合×二归一化”机械当八种独立算法。

### 5. 其他已有消融的分类和当前状态

| 轴 | 既有变体 | 当前处理 |
|---|---|---|
| 表示粒度 | RAGS-Traj / block；旧window_return | Traj已在生产入口；window_return旧部署待适配 |
| 锚点采集 | RAGS-Uniform / adaptive；no_error | Uniform生产入口已有，实际K配对；不能把它误标成全程均匀LESS |
| 初始覆盖 | random_cover | 历史代码存在，当前不自动新增训练 |
| 分裂与预算 | no_split_detect、drop_only、旧grid | 去分裂需重新定义等预算；drop_only/旧过滤违反全部窗口保留，只作历史 |
| 块边界 | fixed_blocks、uniform_segments | 历史部署存在，需适配当前窗口支持/概率协议 |
| 源基础测度 | 块等权、轨迹等权、合法窗口均匀/长度先验 | 评分相同也会改变分布；同支持/同测度随机对照属于必要诊断，不与rand混同 |
| 方向与模长 | 原始梯度平均后余弦 vs 局部余弦平均；净方向重新单位化 | 已有CPU诊断，尚无独立端到端因果结果 |
| 概率映射 | T005/T025；value/rank；正对齐硬门控 | 历史变体已有；硬门控与zeroClip子分数变换分别记录 |
| 成本/模型敏感性 | 密度、投影/head-only、探针检查点、长上下文 | 既有后续计划，本轮不自动扩展 |

### 6. 验收、输出与因果隔离

1. 小矩阵逐项重建四Global公式与Local行混合，验证归一化、max不乘alpha、zeroClip全非正仍单位均匀、expSum含外层Softmax。
2. 固定同向块比例细分不改变加权组件；验证Local与有效先验alpha/Z的等价性及与原Global不同。
3. 旧V3协议恢复/守护重启仍线性；新协议校验聚合、Global/Local、温度、指数尺度、模型/池/预算字段，概率缓存不可跨规则复用。
4. 先报告同温度的原始分布：源块/轨迹质量、熵、ESS、最高10%质量、正负配对与参考贡献；再在允许的开发范围做相近集中度对照，隔离“选谁”和“选多强”。与rand距离在共同合法窗口域计算TV/KL，不能比较不同单位的原始熵冒充同测度。
5. 报告MASE、实际训练抽取量、重复率、梯度K及新增缓存成本。概率/熵诊断不能代替端到端成绩；共享缓存命中也不能把方法独立准备成本记零。

代码接入不增加真实梯度需求，非线性配对评分使用有界ref×src批矩阵及可用GPU，不构建全部参考块×源块稠密矩阵。新聚合不能只从旧线性最终source_scores恢复，需读取已缓存双方代表；复用已备梯度即可。
