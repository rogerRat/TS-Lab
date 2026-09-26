## 2026-09-17 最新实验决策：专家冲突聚合消融（reference-opinion-v1）

**本节为本轮现行设计入口。** 覆盖下文v1/v2的默认温度、方法名、逐参考固定质量、三种子及RAGS-Traj重建贡献均值口径；历史正文、附录结果与旧决策全部保留。当前V3原始资产为C512/H64/D1、2%/1%/1% CRB、全部窗口保留、4096维投影seed42、三学生自身探针、仅seed0。四温度方法lp-mirror/LESS/RAGS默认T005=.05，原T025批次按历史身份留存。详情见[方法最新决策](_method-0903.md)。

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
