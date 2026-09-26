# 从 Embedding 替换风险到机制可辨识的选择性迁移

> 历史归档：保留当时的设计、操作或实验证据，不作为当前执行指令。当前入口见 `docs/README.md`。

> **文档类型：** 开发/论文研究路线决策
> **初始检索日期：** 2026-08-14
> **路线重整日期：** 2026-08-15
> **适用范围：** 当前 TS3/FEV 的 source-reference 训练数据迁移框架
> **文献结论性质：** 基于公开 arXiv、Crossref 和论文 HTML 页面的初步风险评估，不构成穷尽式专利或全部数据库查重。
> **研究纪律：** target/reference 的评估 future 不得参与训练数据选择、gate 构造或超参数选择。

## 0. 给背景知识较弱读者的结论

### 0.1 现在不应转向 Embedding

当前最值得投入的方向，不是把 16 维人工 descriptor 换成更复杂的 learned embedding，也不是继续寻找一种“更强的相似度”。更值得研究的问题是：

> **为什么 OT/KDE 有时能找到有用数据，有时却过于自信地把训练曝光集中到并不真正有用的数据；以及如何在训练前判断应该更偏向 Rand，还是更偏向 OT。**

已有结果已经给出一条很有价值、但还没有闭合的机制线索：

~~~text
reference heterogeneity
    -> source distribution concentration / regime conflict
    -> delivered training-window diversity or utility
    -> negative transfer
~~~

中文解释如下：

1. 一个任务的 reference 历史如果包含很多不同状态，即 **reference 异质性高**；
2. 全局 OT/KDE 可能为了匹配这些状态，把 source 质量强烈集中到少数 AnalysisSegment、轨迹或互相冲突的模式；
3. 模型训练时真正看到的是从这些轨迹中抽出的局部窗口。它们可能重复、覆盖不足，或者只是表面相似而对预测无用；
4. 最终 OT/KDE 反而不如覆盖更广的 Rand，这就是 **负迁移**。

这条链目前是“最值得验证的机制假设”，不是已经证明的因果结论。它的价值在于每一环都能测量，也能用控制实验干预。相比之下，直接切换 Embedding 会同时改变表示、距离、候选集和权重集中度，既增加工程量，也让失败原因更难解释。

### 0.2 五个关键概念

| 概念 | 本文中的含义 | 直观解释 |
| --- | --- | --- |
| Rand | 按原始 source 分布进行随机采样 | 不针对当前任务挑数据，但覆盖较广 |
| OT/KDE | 根据 reference 的 16D descriptor 对 source 重加权 | 多看“长得像目标历史”的训练数据 |
| Reference heterogeneity | 同一 task 的不同 reference 局部窗口在 16D 空间中的离散程度 | 目标历史内部是否包含很多不同状态 |
| Distribution concentration | 少数 AnalysisSegment、轨迹或 legal window start 获得大部分采样概率 | 训练是否反复看到少数数据 |
| Negative transfer | 使用目标条件迁移后，预测表现反而低于 Rand | “聪明挑数据”不如广覆盖随机采样 |

### 0.3 本文最终推荐的研究路线

保留当前 16D descriptor、OT/KDE 和 Rand，把它们当作机制研究所需的固定实验系统。先补齐 ESS、entropy、top-mass、unique exposure 和 delivered-window fidelity，再用 matched-concentration Random、shuffled-reference 和 lambda sweep 闭合机制链。只有机制成立后，才实现一个低自由度、可解释的选择性迁移方法：

~~~text
可靠且不过度集中     -> 更偏向 OT
方向可能有用但过于集中 -> 将 OT 向 Rand 收缩
reference 不可靠或交付失真 -> 回退 Rand
~~~

这条路线的特色不是“又一种相似度”，而是：

> **让训练数据迁移知道自己何时可信、何时应降低强度、何时应拒绝迁移。**

## 1. 当前方法、已有证据与真正的缺口

### 1.1 当前 TS3 实际做了什么

当前生产路径可以抽象为：

~~~text
source/reference history window X
        -> 16D descriptor Z = h(X)
        -> source/reference similarity
        -> OT/KDE AnalysisSegment mass
        -> trajectory -> AnalysisSegment -> legal start
        -> delivered local training window
        -> Chronos training
        -> FEV evaluation
~~~

16 维 descriptor 描述趋势、局部动态、频谱、分布形态和事件特征。OT/KDE 的直接目标是：

~~~text
p_weighted-source(Z) ~= p_reference(Z)
~~~

这里的 Z 只来自允许使用的历史。预测真正关心的却是未来 Y，因此还存在两个没有被静态匹配直接约束的问题：

~~~text
p(Y | Z)
p(Z_next | Z_current)
~~~

换句话说，descriptor distribution 更接近，只能证明“匹配发生了”，不能证明这些数据对模型预测更有用。

### 1.2 当前结果已经支持什么

现有 20-task 离线分析最重要的结果是：

- OT/KDE 在 20/20 task 上都降低了 post-transfer descriptor feature-mean gap；
- 但 forecast metric 没有在 20/20 task 上同步改善；
- reference 内部 descriptor 离散度越高，OT/KDE 相对 Rand 的收益通常越弱；
- 10K 时，临时 heterogeneity index 与 OT 的 MASE improvement 为 rho=-0.616，与 WQL improvement 为 rho=-0.662；
- KDE 的对应关系分别为 rho=-0.526 和 rho=-0.508；
- 跨 1K--10K checkpoint，四个“method × metric”组合中有 8--9/10 个 checkpoint 保持负向；
- 当前尚未记录 source weight ESS、entropy、top-mass、实际 unique exposure 和 delivered-window fidelity。

因此，当前最稳妥的表述仍然是：

> 在当前 20 个 FEV task、16 维静态 descriptor、全局 OT/KDE 和当前训练预算下，reference 内部局部特征离散度越高，OT/KDE 相比 Rand 的收益越弱，负迁移风险越高。

这是一条稳定的 task-level 风险关联，还不是“异质性导致负迁移”的因果证明。

### 1.3 为什么最可能的问题是“过度自信地集中错了”

Rand 与 OT/KDE 的核心区别不是训练模型、训练预算或 source 内容，而是 source exposure probability。既然 post-transfer gap 已经下降，继续证明“OT 找到了 descriptor 上更相似的数据”边际价值很低。真正缺少的是：

~~~text
计划给了谁更高概率
    -> 训练实际交付了哪些局部窗口
    -> 这些窗口是否重复、冲突或缺少覆盖
    -> 模型为何因此受益或受损
~~~

最可能的故障并不是完全找不到相似 source，而是 OT/KDE 以很高权重反复选择少数“descriptor 上相似”的 source，但这些 source：

- 在局部窗口层面并没有交付相似内容；
- 覆盖了互相冲突的 reference regime；
- 牺牲了 Rand 的广覆盖和有效样本数；
- 或者在相同历史 descriptor 后具有不同的 future 规律。

这正是 ESS、entropy、top-mass、unique exposure 和 delivered-window fidelity 能区分的机制。

### 1.4 16D 可解释性是资产，不是落后

当前 16D descriptor 的局限确实存在：它不是完整序列，也不保证与 Chronos 的内部表示或梯度一致。但是在现阶段，可解释性恰好是机制研究的优势：

- 可以指出究竟是 volatility、frequency、asymmetry 还是其他结构在 task 内冲突；
- 可以分别测量 reference dispersion、source coverage 和 delivered-window gap；
- 可以把 planner 的目标与模型实际输入放在同一结构空间审计；
- 可以构造明确的 matched-concentration、regime coverage 和 feature-group 控制；
- 失败时能够定位是表示不足、权重集中还是 trajectory-window 交付失真。

Embedding 会把这些可读维度压进一个高维表示。它可能更强，也可能更集中，但在缺少当前机制指标时，很难知道改善或退化来自哪里。因此 16D 目前应被保护为 **测量仪器**，而不只是等待替换的旧表示。

## 2. 其他工作、方法重合与 Embedding 路线风险

本节保留现有文献检索与重合度分析。它回答的不是“Embedding 永远不能做”，而是“为什么它不应成为当前两个月的主线”。

### 2.1 重合度总览

当前没有检索到一篇与本项目完全一一对应的论文，即同时实现以下全部要素：

- 当前 TS3 的 16 维趋势、动态、频谱、分布和事件 descriptor；
- FEV 的 source-reference 跨任务设置；
- 训练阶段的 source window reweighting；
- OT/KDE 形式的 reference-conditioned 分布迁移；
- Chronos 或同类 time-series foundation model；
- 用 embedding lookup 替换当前 16 维 descriptor。

但是，Embedding 方案的主要组件已经分别被多个方向覆盖：

| 风险类型 | 当前判断 | 说明 |
| --- | --- | --- |
| 完全重复风险 | 中低，约 3--5/10 | 暂未发现完全相同的端到端方法。这个结论受检索范围和公开程度限制。 |
| 核心组件重合风险 | 高，约 8/10 | embedding、nearest-neighbor lookup、时间序列 retrieval、source-target similarity、训练数据 mixture selection 均已有近邻工作。 |
| 只替换 16D descriptor 的论文创新风险 | 很高，约 8.5--9/10 | 容易被评为将现成 embedding 插入现有 OT/KDE pipeline 的增量替换。 |
| embedding + ANN + 原有 OT/KDE | 高，约 7.5--8/10 | 比简单替换多了一层检索，但研究问题仍然接近已有 retrieval/data selection 工作。 |
| target-conditioned、window-level、future-blind、diversity-constrained 训练迁移 | 中等，约 5--6/10 | 仍处在拥挤领域，但可以形成区别于 inference-time retrieval 的明确研究问题。 |
| 加入负迁移机制、ESS/coverage 约束和完整跨任务证据 | 中等偏低，约 4--5/10 | 具有较清晰的独立贡献潜力，但不能保证中稿。 |

最重要的文献风险判断是：

> 不应把论文主线写成“将 16 维人工特征升级为 embedding”。更有辨识度的问题是 future-blind、target-conditioned 的训练窗口迁移为何失败，以及如何用覆盖、多样性和可靠性约束避免负迁移。

### 2.2 近邻论文逐项对照

| 工作 | 已有方法的核心内容 | 与当前 TS3 的相似点 | 改成 embedding 后的新增重合点 | 关键差异 | 综合重合风险 |
| --- | --- | --- | --- | --- | --- |
| [RAF](https://arxiv.org/abs/2411.08249) | 在 Chronos 等 TSFM 上建立时间序列数据库，用模型 embedding 相似度检索历史序列及其 future，增强预测输入 | 都使用历史窗口、source/reference 相似性和 Chronos | embedding、top-k lookup、TSFM、跨域历史序列检索几乎直接重合 | RAF 主要是 inference-time retrieval；TS3 是 training-time source window reweighting | 很高 |
| [RATD](https://arxiv.org/abs/2410.18712) | 冻结预训练 time-series encoder，将窗口编码后做 top-k retrieval，用参考序列指导 diffusion denoising | 都使用数据库中的历史窗口作为可迁移信息 | 冻结 encoder、embedding distance、nearest-neighbor reference lookup 直接重合 | RATD 服务于 diffusion forecasting；TS3 服务于 Chronos training data sampling | 高 |
| [TimeRAG](https://arxiv.org/abs/2412.16643) | 用 K-Means 构造历史序列知识库，再用 DTW 检索相似序列并放入 forecasting LLM prompt | 都进行历史窗口相似性搜索并利用 source/reference 样本 | embedding/cluster knowledge base 和检索式 forecasting 更接近 | TimeRAG 主要在预测时拼接参考序列，不做 TS3 式 source reweighting | 中高 |
| [Optimizing the Training Diet](https://arxiv.org/abs/2512.11546) | 用 MOMENT embedding、K-Means 和 Optuna 搜索训练数据 cluster 的最优 mixture weights | 都把表示空间用于训练数据组成或采样分布调整 | learned embedding -> cluster -> training mixture 与拟议方案高度相似 | 其 target 是同一 PMSM 数据集的下游性能，不是 FEV source-reference 跨任务 transfer；不做 ANN reference lookup | 很高 |
| [Wasserstein Dataset Similarity](https://arxiv.org/abs/2507.22189) | 把时间序列数据集建模成分布，用 Wasserstein distance 估计 source-target 相似度并辅助 source 选择、fine-tuning 和性能估计 | 都研究 source/reference 分布接近是否带来迁移收益 | embedding-space distribution matching 会进一步落入同一 source-target similarity 问题 | 其方法是 dataset-level MVN/Wasserstein distance，不是 learned window embedding，也不负责窗口级训练采样 | 高 |
| [Shapelet-based Source Selection](https://arxiv.org/abs/2409.20005) | 用 shapelet similarity 选择适合 target 的 source datasets，再进行多源预训练和 transfer learning | 都是 target-conditioned source selection，且关注 negative transfer | 如果 embedding 用于选择 source windows，问题层面更加相似 | 其对象主要是 time-series classification 的 dataset-level source selection | 中高 |
| [SARAF](https://arxiv.org/abs/2606.04135) | 指出相似度检索在非平稳序列中可能失效，加入 stationarity-aware relevance/diversity selection | 都面临历史相似不等于未来相似的问题 | embedding lookup 如果只按相似度取 top-k，会直接遇到其批评的问题 | SARAF 主要针对 inference-time retrieval；TS3 可研究 training-time negative transfer | 高 |
| [TSDS-Toolbox](https://arxiv.org/abs/2608.08119) | 提供 time-series dataset similarity 的统一评测工具，覆盖 dataset-level 和 series-level similarity | 都把相似性用于 source selection 和迁移评估 | embedding-based series/dataset reducers 可直接成为拟议方法的基线 | 主要是评测工具，不是同一训练采样方法 | 中高 |
| [TS-RAG](https://arxiv.org/abs/2608.06223) | 检索相似时间序列并通过 reference tokens 融合输入和检索序列 | 都使用相似历史序列增强时序模型 | embedding retrieval 与 neural fusion 的组合更接近 | 仍是 inference-time retrieval，而非 target-conditioned training distribution transfer | 中高 |
| [Align-RAG](https://arxiv.org/abs/2608.05571) | 对检索到的 past-future window 做 amplitude/phase 对齐，再输入冻结 TSFM | 都关注 retrieved window 是否真正有预测效用 | embedding retrieval + future-aware alignment 会与拟议的 utility-aware retrieval 形成近邻 | 不改变训练数据采样，强调无训练的推理增强 | 中高 |
| [CrossRAG](https://arxiv.org/abs/2607.29459) | 使用 shape-aware memory、future-consistent contrastive learning 和 cross-attention 融合检索结果 | 都需要处理“历史相似但 future 不一致”的 hard negative | 如果 embedding 以 source future consistency 训练，概念上会部分重合 | CrossRAG 主要是 inference-time RAG，并非 TS3 的 source sampling | 中高 |
| [TS2Vec](https://arxiv.org/abs/2106.10466) | 学习可用于时间序列分类、预测和异常检测的层次化表示 | 当前人工 descriptor 与 learned representation 都是窗口结构压缩 | 直接采用 TS2Vec 作为 lookup encoder 会把表示部分变成现成组件 | TS2Vec 本身不是 source-reference training sampler | 表示层高、方法层低 |
| [TNC](https://arxiv.org/abs/2106.00750) | 使用 temporal neighborhood coding 学习非平稳时间序列 representation | 与局部窗口、regime 和邻域相似性有关 | 可作为 embedding lookup 的现成 encoder baseline | 不涉及 Chronos training data transfer | 表示层中等 |

### 2.3 最直接的三个重合来源

#### A. RAF：与 Chronos 和 embedding retrieval 的直接重合

RAF 是最需要正面引用和区分的工作之一。其方法包括：

- 为不同数据域建立时间序列数据库；
- 使用模型 encoder 得到 query 和 database sequence 的 embedding；
- 用 embedding distance 选 top-k 或 top-1 相似序列；
- 取被检索序列的历史和 future；
- 将其作为额外上下文输入 Chronos 等 TSFM。

因此，如果论文只声称“使用 embedding 查找相似时间序列来帮助 Chronos”，会与 RAF 产生非常直接的重合。

TS3 可以区分的地方是：

- RAF 是 inference-time retrieval；
- TS3 是 training-time source window selection/reweighting；
- RAF 直接把 retrieved future 作为预测上下文；
- TS3 的 reference future 在选择阶段不能被使用，source future 只作为训练监督；
- TS3 目标是改变 Chronos 的训练 exposure distribution，而不是改变单次预测输入。

这个差异必须在论文问题定义、算法图和实验协议中明确出现，否则审稿人可能把 TS3 看成一种没有直接利用 future 的 RAG 变体。

#### B. Optimizing the Training Diet：与训练数据选择的直接重合

该工作已经公开采用：

~~~text
MOMENT encoder
    -> dense time-series embeddings
    -> K-Means behavioral clusters
    -> Optuna search over cluster sampling ratios
    -> train target forecaster on selected mixture
~~~

这与“embedding 后选择训练样本”在方法结构上非常接近。它与 TS3 的主要差异是：

- 它在同一 PMSM 数据集内部寻找最优 training diet；
- 它以目标模型验证性能直接驱动 mixture search；
- 它没有当前 TS3 的 source/reference 两角色和 FEV 跨任务 reference population；
- 它没有当前 TS3 的 OT/KDE static transfer distribution；
- 它不是逐个 reference window 进行 ANN lookup。

因此，TS3 不能把“embedding 用来改进训练数据组成”作为唯一贡献。需要强调 target-conditioned cross-task transfer、future-blind selection、窗口级分布迁移和 negative-transfer control。

#### C. SARAF：说明 naive similarity lookup 本身并不稳妥

SARAF 的核心批评与当前 TS3 的离线诊断高度一致：

~~~text
历史窗口相似
    !=
未来演化相似
~~~

SARAF 认为非平稳和 regime shift 会使单纯相似度检索变得脆弱，因此同时考虑 relevance 和 diversity，并根据 stationarity 调整检索策略。

这对 TS3 的含义是：

- 直接把 16D descriptor 换成 embedding，不能自动解决历史相似与未来效用不一致；
- embedding 可能只让局部相似样本更集中，反而降低 source exposure diversity；
- 需要记录 ESS、weight entropy、top-k mass、unique source windows 和 regime coverage；
- 需要证明相似度能转化为 training utility，而不是只降低静态 descriptor gap。

### 2.4 当前 TS3 与这些工作的边界

当前 TS3 的组合位置是：

~~~text
target/reference history population
    -> source window distribution transfer
    -> OT/KDE reweighting
    -> training-time sampling
    -> Chronos/FEV evaluation
~~~

| 研究层面 | 当前 TS3 | 现有近邻工作的常见做法 |
| --- | --- | --- |
| 检索/匹配对象 | source 和 reference 的局部历史窗口 | query 与 database 的相似序列，或 source/target 整个数据集 |
| 使用时机 | 训练前构建 transfer distribution，训练时采样 | 多数是 inference-time retrieval；少数是训练数据 mixture search |
| reference future | 选择时不使用 target future | retrieval forecasting 常常拿 retrieved future 作为上下文 |
| 分布目标 | 让 weighted source descriptor distribution 接近 reference | top-k reference、cluster mixture 或 dataset distance |
| 模型 | Chronos training adapter | Chronos/TSFM inference、diffusion、LLM 或其他 forecaster |
| 主要风险 | global static matching 的 negative transfer | similar-history/different-future、非平稳、retrieval redundancy |

改成 Embedding 后，方案会更接近已有的共同范式：

~~~text
encode -> retrieve/cluster -> select/reweight -> forecast or train
~~~

当前真正能够形成区分的，不是 encode 这一步，而是训练阶段的 exposure distribution、future-blind 的可靠性判断，以及负迁移控制。

### 2.5 DSIR 与 LESS 对数据选择路线的启示

DSIR 在语言模型 continued pretraining 中使用 hashed n-gram importance resampling。它说明：当表示与训练目标较接近、target 样本能代表后续分布、并且重采样保留了足够覆盖时，静态数据选择可以有效。

LESS 则说明这种成功不能直接泛化。在 targeted instruction tuning 中，简单 n-gram matching 并不总是稳定优于 Random，因此 LESS 转向与目标损失更相关的梯度相似度。即使如此，warmup、模型状态、代理 loss 与最终 metric 的错位仍然会限制稳定性。

对 TS3 的直接启示是：

> 当前结果不否定数据选择；它否定的是“静态相似度下降足以证明 forecast utility 提升”。研究重点应从相似度本身转向相关性收益与多样性代价之间的可辨识权衡。

### 2.6 为什么现在改 Embedding 风险高

#### 论文风险

如果只做：

~~~text
16D descriptor -> pretrained embedding
~~~

而 global matching、OT/KDE、fixed lambda、sampler、Chronos 和 FEV protocol 都不变，那么贡献很容易被评价为 representation replacement。即使再加 ANN、K-Means 或 top-k，也会显著靠近 RAF、RATD、TimeRAG 和 Training Diet。

| 方案版本 | 与已有工作的方法重合 | 论文重复风险 | 预计审稿评价 |
| --- | --- | --- | --- |
| 现有 16D descriptor 直接换成现成 embedding | 高 | 高 | 工程替换/ablation，创新不足 |
| embedding + ANN top-k + 原有 OT/KDE | 很高 | 高 | embedding retrieval 与 training reweight 的简单组合 |
| embedding + K-Means + source mixture | 很高 | 高 | 与 Training Diet 的 encode-cluster-optimize 范式接近 |
| embedding + target/reference window matching | 高 | 中高 | 有应用差异，但需要明确区别于 RAF/RATD/TimeRAG |
| embedding candidate recall + local matching + diversity/ESS | 中高 | 中等 | 有方法贡献潜力，但需要完整机制证据 |
| 上述方法 + future-blind protocol + negative-transfer analysis + 20-task evidence | 中等 | 中等偏低 | 可形成 TS3-specific contribution，但价值主要来自机制与协议，而不是 Embedding |

#### 科学风险

Embedding 仍然只根据可见历史产生表示。它不会自动保证：

~~~text
历史表示相似
    -> future 机制相似
    -> 训练梯度一致
    -> 最终预测更好
~~~

更强的表示甚至可能产生更尖锐的近邻和更集中的权重，从而加重当前最可疑的 failure mode。

#### 工程与时间风险

切换 Embedding 需要重新决定 encoder、层、归一化、窗口协议、距离、索引、冻结策略、缓存和计算预算，还需要重新跑表示基线和公平消融。两个月内这些工作会挤占最重要的机制控制、paired seeds 和 20-task 确认。

#### 证据资产风险

当前 heterogeneity 结果、16D feature gap、OT/KDE 权重和 Rand 对照已经形成连续证据。切换表示会让已有证据无法直接连接到新方法。现阶段更合理的做法是固定表示，利用它追踪 failure chain。

## 3. 新的研究问题与论文主张

### 3.1 研究问题

新的核心问题应写成：

> **在看不到 target evaluation future 的情况下，能否根据 reference 可靠性、source exposure 集中度和实际训练窗口交付质量，判断 target-conditioned transfer 何时可信，并选择更偏 Rand、收缩后的 OT，或完整 OT？**

这里的“future-blind”不是说永远不能使用任何 future。source 的训练 future 可以作为 source supervision；reference 安全历史内部也可以构造 pseudo-cutoff。禁止使用的是 benchmark target 的 held-out evaluation future，因为它会造成数据泄漏。

### 3.2 主张一：相似性对齐不是充分条件

可以由当前结果直接支持的主张是：

> OT/KDE 可以成功降低 source-reference descriptor gap，但这种静态对齐不是 forecast utility 提升的充分条件。

20/20 task 的 gap reduction 与不稳定的预测收益已经提供了直接反例。论文不需要再把大量篇幅用于证明“匹配成功”，而应解释匹配为什么没有稳定转化为效用。

### 3.3 主张二：负迁移是可以沿 exposure chain 诊断的

需要通过下一阶段实验建立的主张是：

> Reference 异质性并不是一个孤立的相关变量。它会通过 source 权重集中、regime 冲突或 AnalysisSegment-proxy 到 local-window 的交付失真，改变模型真正看到的训练窗口，从而影响负迁移风险。

该主张只有在以下四层同时获得证据时才成立：

1. reference 属性能够预测 exposure 的变化；
2. exposure 的变化能够在实际 delivered windows 中观察到；
3. delivered diversity 或 proxy utility 的变化与性能方向一致；
4. 干预 concentration、lambda 或 delivery 后，性能按机制预期变化。

### 3.4 主张三：迁移强度应当是可拒绝、可收缩的

最终方法主张应是：

> Target-conditioned training data transfer 不应作为所有 task 的固定动作。一个 future-blind、低自由度、可审计的可靠性与多样性约束，可以在适合的 task 上保留 OT 收益，并在不适合的 task 上向 Rand 收缩或回退。

这个主张比“OT 总是优于 Rand”更现实，也更有研究辨识度。Rand 不是需要被淘汰的弱基线，而是 transfer 不可信时的安全先验。

### 3.5 目前不能声称什么

在控制实验完成前，不应声称：

- reference heterogeneity 已经被证明会导致 concentration；
- Rand 获胜一定是因为看到更多 unique windows；
- 16D descriptor 是负迁移的根因；
- ESS 越大就一定越好；
- 一个用 20 个 task 拟合出来的 gate 可以泛化；
- selective transfer 已经消除了 negative transfer；
- Embedding 一定无效或永远不应使用。

严谨的叙述应始终区分“当前观察”“机制假设”“干预后因果证据”和“最终方法结果”。

## 4. 如何测量完整机制链

### 4.1 Reference heterogeneity：目标历史内部有多不一致

当前定义是同一 task 的 reference 局部 segment 在 16D descriptor 上的离散度。对第 j 个 descriptor：

~~~text
heterogeneity_j = std(Z_1,j, Z_2,j, ..., Z_M,j)
~~~

当前 volatility clustering、spectral centroid 和 asymmetry 的 task 内 std 最稳定。下一步应保留逐维量，不要只留下一个总分；总 heterogeneity index 仅用于简化 gate 或排序，而且其组合方式必须在确认集之前固定。

还应补充：

- reference segment 间的平均和尾部 pairwise distance；
- 每个 descriptor 的 robust dispersion，例如 IQR/MAD；
- 简单 regime coverage 或 cluster balance，仅作为辅助审计；
- 安全历史内部 pseudo-cutoff 的 regime 稳定性。

不应一开始就引入复杂深度聚类。20 个 task 很容易过拟合，逐维结构指标更可解释。

### 4.2 Planned concentration：采样计划有多集中

当前 planner 的直接输出不是每个完整 window 的显式概率，而是 source AnalysisSegment 质量 q_(i,k)。运行时先把同一 trajectory 内的 segment 质量求和，再依次采样 trajectory、segment 和 legal start。一个跨 segment 的 window 可能经由多条路径被选中，这些路径概率会相加。

因此 concentration 必须在三个固定粒度分别报告，不能混用：

1. **AnalysisSegment 粒度：** planner 直接输出的 q_(i,k)；
2. **Trajectory 粒度：** P(t_i) = sum_k q_(i,k)；
3. **Legal-window-start 粒度：** 由 segment 质量、重叠几何和 start representation 共同诱导的窗口概率。

设某一固定粒度上归一化概率为 p_i，且所有 p_i 之和为 1。下面的 ESS、entropy 和 top-mass 都必须标注粒度；不同粒度或不同 support size 的原始 ESS 不能直接比较。

**Effective Sample Size（ESS）**

~~~text
ESS = 1 / sum_i p_i^2
normalized_ESS = ESS / N
~~~

如果所有 N 个 source 等概率，ESS=N；如果概率集中在少数 source，ESS 会明显下降。normalized ESS 便于比较 source 数不同的任务。

**Entropy 与 perplexity**

~~~text
H = -sum_i p_i log(p_i)
perplexity = exp(H)
normalized_entropy = H / log(N)
~~~

Entropy 衡量概率分布有多均匀；perplexity 可以理解为“相当于均匀使用了多少个 source”。

**Top-mass**

分别记录 top-1%、top-5%、top-10、top-100 segment/trajectory/window-start 累积拿走了多少概率。ESS 相同的两个分布仍可能有不同头部形状，因此 top-mass 不能省略。

**组成与覆盖**

同时记录加权后的 domain、frequency、trajectory length 和 descriptor regime 组成，避免把“source 数多”误当成“结构覆盖广”。

### 4.3 Realized exposure：训练真正看到了什么

计划概率不等于实际曝光。有限训练步数、DataLoader 预取、层级采样和窗口起点都会改变实际结果。

当前代码已经提供 training_audit 事件，不需要先修改生产训练语义：

- slot_scheduled 已记录 trajectory probability 和 exposure ratio；
- window_selected 已记录 trajectory ID、legal rank 和 window start；
- slot_delivered 能区分“被选中”与“真正被 adapter 接受”的窗口；
- batch_finalized 能排除失败 batch。

因此第一步应新增离线 analyzer，复用现有 audit 还原 realized exposure；只有 segment identity 或局部 descriptor 确实无法从 transfer artifact 与 window locator 恢复时，才增量扩展 audit schema。

每个真正 delivered 的训练样本至少应记录或可重建：

- trajectory identifier；
- local window 的 start/end；
- sampling method、task、seed、step；
- local-window 16D descriptor；
- 是否与此前窗口完全重复；
- trajectory 与 local-window 的 domain/frequency/length/regime。

由此计算：

- unique trajectory exposure；
- unique local-window exposure；
- 重复率和最大重复次数；
- realized ESS/entropy/top-mass；
- 结构 regime coverage；
- planned probability 与 realized frequency 的偏差。

对任一离散 support、训练交付次数 B 和计划概率 p_i，还可先计算期望 unique exposure：

~~~text
E[unique] = sum_i (1 - (1 - p_i)^B)
~~~

再将它与 training_audit 中的 realized unique exposure 比较。若局部 descriptor 回放代价过高，可以固定间隔抽样，但抽样规则必须对 Rand 和 OT 完全相同。

### 4.4 Delivered-window fidelity：计划的相似性有没有送到模型

OT/KDE 在 AnalysisSegment proxy 上分配质量，但模型最终消费的是局部训练窗口。当前 overlap-aware window geometry 是正式数据契约：AnalysisSegment 长度/stride、window 长度/stride 和跨 segment overlap 都必须使用 run 对应的 metadata，不能用简化后的均匀窗口假设代替。AnalysisSegment 本身是否重叠也以对应 run 的 metadata 为准。

对 trajectory i 的 legal window j，其实际计划概率由所有可到达该窗口的 segment 路径共同诱导：

~~~text
P(w_(i,j)) = sum_k q_(i,k) * overlap_contribution_(i,j,k) / segment_normalizer_(i,k)
~~~

因此应优先用当前 window geometry helper 或等价解析计算得到 induced window-start distribution，而不是只用 trajectory mass 近似。

定义三个层次：

~~~text
D_plan       = reference 与计划加权 source 表示的距离
D_delivered  = reference 与实际 delivered windows 分布的距离
D_utility    = delivered windows 与安全 proxy 预测效用的关系
~~~

“delivery fidelity”至少回答：

1. D_plan 相比 Rand 下降时，D_delivered 是否也下降；
2. reference 中的主要 regime 是否在 delivered windows 中得到覆盖；
3. 改善是否只来自少数重复窗口；
4. AnalysisSegment 相似但 induced/delivered local window 不相似的比例有多大。

除了均值差，还应使用分位数、nearest-distance distribution、regime coverage 和完整分布距离。当前 post-transfer feature-mean gap 太弱，不能独立代表 fidelity。

### 4.5 Utility 与 regime conflict：相似窗口是否真的有用

target evaluation future 不能参与选择，因此 utility 必须分为两类。

**训练前可用的 future-blind proxy**

- 在 reference 安全历史内部构造 pseudo-cutoff，用前段预测后段；
- 用冻结 Chronos 的 pseudo-future loss 或 likelihood 评估局部可预测性；
- 在 source 内部比较“相似 history 后的 future dynamics 是否一致”；
- 比较不同 source regime 对同一 reference pseudo-window 的 loss 排名；
- 必要时在很小的代表集上测量梯度 cosine/conflict，不能一开始全量计算。

**仅用于事后科学审计的 held-out future**

真实 target future 可以用于解释最终失败、per-window 分解和论文评估，但不能回流到 gate、阈值或候选选择。

如果两个 source window 在 16D 上相似，但其 source future dynamics、probe loss 或梯度方向长期不一致，就能更直接地支持“descriptor similarity 不等于 training utility similarity”。

### 4.6 机制闭合标准

不能因为某个 task-level correlation 显著就宣布根因。完整证据应当形成：

~~~text
reference heterogeneity
    -> planned concentration / regime conflict
    -> realized diversity and delivery fidelity
    -> proxy utility or model behavior
    -> negative transfer
~~~

至少还要有一个干预把中间环节改掉，并观察结果按预期变化。例如，在不改变相似度定义的情况下提高 ESS 后，负迁移减轻；或者在保持 concentration 相同的情况下打乱 reference 后，收益消失。

## 5. 最高性价比的方法：Selective Rand-OT

### 5.1 从二选一改为连续收缩

当前实现已经在 AnalysisSegment mass 上用 lambda 混合 Rand/raw mass 与 task-conditioned OT target mass。因此不需要发明新 sampler，只需要把固定 lambda 变成由 task 可靠性决定的 lambda_task：

~~~text
q_lambda = (1 - lambda_task) * p0 + lambda_task * q_OT
lambda_task in [0, 1]
~~~

- lambda_task=0：完全使用 Rand/raw segment mass；
- lambda_task=1：完全使用当前 OT target segment mass；
- 0<lambda_task<1：保留一部分 task relevance，同时恢复一部分覆盖和多样性。

这不是根据 target test 结果挑一个最好的 lambda。研究目标是用训练前、future-blind 的可靠性指标决定 lambda_task 的上限，并用 ESS/entropy/top-mass 约束防止过度集中。KDE 可以复用同一机制，但第一版应先固定 OT 主线，避免同时增加 method-selection 自由度。

### 5.2 三种动作

| 训练前状态 | 动作 | 原因 |
| --- | --- | --- |
| reference 较一致、pseudo-cutoff 稳定、delivery fidelity 高、OT 后 ESS 仍充足 | 更偏向 OT | task-specific relevance 有证据且多样性代价可控 |
| relevance 方向可能有效，但 OT 权重过度集中 | shrink toward Rand | 保留方向，降低重复曝光和覆盖损失 |
| reference 高异质且不稳定，或 segment-to-window fidelity 低，或 proxy utility 不支持 | 回退 Rand | 当前 task-conditioned signal 不值得信任 |

如果实验最终表明 KDE 与 OT 的 concentration 形态不同，可以为二者分别校准；第一版不应同时做复杂 method selection、Embedding 和 dynamic schedule。

### 5.3 一个低自由度、可审计的 gate

建议采用两阶段规则，而不是在 20 个 task 上训练黑盒分类器。

**阶段一：可靠性上限**

使用少数预先定义的 future-blind 指标，把 lambda_max 限制在 {0, 0.5, 1}：

- reference heterogeneity；
- reference 内部 pseudo-cutoff stability；
- 可在训练前解析计算或 dry sampling 得到的 segment-to-window delivery fidelity。

**阶段二：多样性投影**

从 lambda_max 开始向下收缩，直到同时满足：

- normalized ESS 不低于预设下限；
- normalized entropy 不低于预设下限；
- top-1%/5% mass 不超过预设上限；
- 关键 regime coverage 不低于预设下限。

这种设计容易解释：可靠性决定“最多相信 OT 多少”，多样性约束决定“OT 能否以这个强度安全执行”。

阈值必须通过 leave-one-task-family-out 或明确的 discovery/confirmation task 划分确定。不能先查看某个 task 的 held-out 结果，再为它选择 lambda。

### 5.4 为什么这比直接做 local matching 或 Embedding 更优先

Selective Rand-OT 复用全部已有资产，只改变混合强度，工程面最小，而且能直接测试“相关性收益与多样性损失”的主假设。

Local/regime matching 只有在 delivery audit 证明 AnalysisSegment-level 计划没有交付到 local windows 时才值得实现。Embedding 只有在现有表示被证明确实无法区分 future-consistent source，且 concentration/fidelity 已经得到控制后才值得重启。两者都应由机制结果触发，而不是并行堆叠。

### 5.5 方法边界

Selective Rand-OT 不能保证每个 task 都超过 Rand。合理目标是：

- 保留低风险 task 上 OT 的主要收益；
- 降低高风险 task 的 negative-transfer rate；
- 改善 worst-task regret；
- 让每次偏向 OT 或回退 Rand 都有可审计理由。

如果 gate 只提升均值，却通过牺牲少数 task 造成更差尾部风险，就没有解决本文提出的问题。

## 6. 实验路线：先辨识机制，再实现方法

### 6.1 Phase 0：冻结协议与防止事后选择

在新增实验前固定：

- source catalog、reference bundle、16D descriptor 和窗口几何；
- Chronos 规模、训练预算、batch、optimizer 和 checkpoint；
- Rand/OT/KDE 的 matched task、matched step 和 paired seed；
- discovery task、confirmation task 和 task-family 划分；
- 主指标、negative transfer 定义和异常处理；
- target evaluation future 只用于最终评估，不参与 gate。

任何用 test-selected checkpoint 得到的结果都不能简单称为“best”。主比较应使用预先固定或 matched 的 checkpoint。

### 6.2 Phase A：20-task 零训练或低成本机制审计

第一阶段不改变生产 sampler，也不需要重跑完整训练。对 20 个 task 计算：

- reference 逐维 heterogeneity、robust dispersion 和简单 regime coverage；
- Rand、OT、KDE 的 planned ESS、entropy、perplexity、top-mass；
- expected unique exposure 和已有 run 能恢复的 realized exposure；
- AnalysisSegment-level plan distance 与 induced/delivered window distance；
- reference 到 delivered window 的 fidelity；
- pseudo-cutoff stability 和 source-only future consistency proxy。

这一阶段首先验证三个问题：

1. heterogeneity 是否真的对应更强 concentration 或 regime conflict；
2. planned concentration 是否真的转化为更低 realized diversity；
3. AnalysisSegment-level matching 是否真的传递到 local training windows。

如果第一条就不成立，论文不能继续使用“heterogeneity 导致 concentration”作为主机制；应转而检查 regime conflict 或 delivery mismatch。

### 6.3 Phase B：代表 task 的控制实验

代表 task 不应只按最终表现挑选。建议预先选择四个象限：

| Reference heterogeneity | 历史 OT 相对 Rand | 用途 |
| --- | --- | --- |
| 低 | 正向 | 正常成功案例 |
| 低 | 负向 | 检查表示或 utility mismatch |
| 高 | 正向 | 检查 heterogeneity 是否并非充分风险条件 |
| 高 | 负向 | 最可能的 concentration/regime-conflict 案例 |

如果预算只能做三个 task，至少保留一个正向、一个负向、一个高异质大 reference population task，并在文档中说明删减规则。

先用单 seed 做明显无效条件的筛除；所有支持论文主张的关键比较至少使用三个 paired seeds。

### 6.4 必须做的控制

| 控制 | 保持什么 | 改变什么 | 能回答的问题 |
| --- | --- | --- | --- |
| Matched-concentration Random | 与 OT 相近的 ESS/entropy/top-mass | 不使用 reference 指定具体 source | 退化来自“集中本身”还是“集中错对象” |
| Shuffled-reference OT | OT 算法；另按 reference 数量和 heterogeneity 配对 | 将 task 与其他 reference 配对 | 收益是否真的 task-specific |
| Lambda sweep | 表示、reference、sampler 不变 | transfer 强度 | 是否存在 relevance-diversity trade-off |
| Candidate-uniform | 保留 OT 选出的 support | support 内改为均匀 | 价值来自候选集合还是尖锐权重 |
| Planned vs delivered audit | 同一采样计划 | 比较轨迹和实际局部窗口 | 数学目标是否到达模型输入 |
| Regime-balanced OT | 保留 OT 总体方向和预算 | 限制各 regime 头部集中 | 高异质失败是否来自 regime coverage/conflict |

Matched-concentration Random 必须在论文声称起作用的同一粒度匹配整个概率形状，而不只是一个 ESS 数字。若主张 delivered-window diversity 是原因，就至少应匹配 induced window-start 的 ESS/top-mass/expected unique exposure，并同时报告 trajectory 和 segment 粒度的残余差异。Shuffled reference 也不能被默认视为 concentration 相同；必须配对 reference 规模/heterogeneity，并在结果中复核 concentration。

### 6.5 如何根据观察选择下一步

| 观察 | 更支持的解释 | 下一步 |
| --- | --- | --- |
| Matched-concentration Random 与 OT 同样退化 | 多样性损失是主因 | 实现 ESS/entropy-constrained shrinkage |
| Matched-concentration Random 正常，OT 退化 | OT 集中了错误 source | 检查 utility proxy、regime conflict 和表示不足 |
| D_plan 下降但 D_delivered 不降 | segment-proxy/window 交付错配 | 实现 local-window matching，而不是 Embedding |
| D_delivered 下降但 source future consistency 差 | 历史相似不等于 future utility | 加 source-only/pseudo-future reliability gate |
| 小 lambda 好、大 lambda 坏 | relevance-diversity trade-off | 使用连续收缩或 diversity projection |
| Shuffled-reference 与真实 reference 一样好 | task-specific 匹配证据不足 | 不主张 reference-conditioned benefit |
| 控制 concentration 后高异质 task 仍失败 | 不只是集中问题 | 检查 regime conflict、conditional variables 和表示边界 |

### 6.6 Phase C：只实现被证据支持的方法

机制结果决定开发分支：

- concentration 是主因：实现 ESS/entropy/top-mass constrained shrinkage；
- reference reliability 是主因：实现 heterogeneity/pseudo-cutoff lambda cap；
- delivery mismatch 是主因：实现 local-window 或 regime-conditioned matching；
- utility mismatch 是主因：加入 source-only/pseudo-future reliability，不使用 target future；
- 没有机制稳定复现：不要包装复杂 gate，保留为负迁移分析论文或重新评估方向。

一次只实现一个主要干预。不要同时加入 Embedding、dynamic OT、gradient selector 和复杂 gate，否则无法归因。

### 6.7 Phase D：20-task 确认

最终至少比较：

| 方法 | 目的 |
| --- | --- |
| Rand | 安全基线 |
| 当前固定强度 OT | 原方法 |
| 当前固定强度 KDE | 原方法的另一种重加权 |
| 固定 lambda shrinkage | 检查单纯降低迁移强度是否已经足够 |
| concentration constraint only | 隔离多样性约束的贡献 |
| reliability gate only | 隔离 task 可靠性判断的贡献 |
| 完整 Selective Rand-OT | 验证二者组合 |

对 20 个 task 报告 matched checkpoint、paired seed 的逐任务和聚合结果，并做 leave-one-task-family-out。只有 confirmation task/family 的结果才能支持 gate 泛化。

### 6.8 跨规模与 Embedding 基线的优先级

先在 9M 完成机制和方法筛选。46M 只用于少量未参与机制发现的 task 做方向确认；200M 只有在核心结论稳定且预算允许时才考虑。

Embedding 不进入主 sampler 开发。论文若需要回应 reviewer，可以在方法锁定后增加一个或两个 frozen embedding diagnostic baseline，只回答：

- 它是否改善 source future consistency；
- 它是否改善 delivered-window utility；
- 它是否加重 concentration；
- 在相同 ESS/coverage 下是否仍优于 16D。

如果它只降低 embedding distance，却不改善上述指标，就停止扩展。

## 7. 评价、报告与验收标准

### 7.1 最终性能指标

除 MASE/WQL 等主预测指标外，还要报告：

- mean 和 median improvement relative to Rand；
- negative-transfer rate；
- worst-task regret；
- 正向 OT task 的收益保留率；
- gate 选择 Rand/shrink/OT 的 coverage；
- 不同 task family 的结果；
- paired-seed difference、置信区间或 bootstrap interval；
- checkpoint curve 或预先定义的 curve AUC，而不是只挑单点。

“Selective”方法的价值主要体现在减少错误迁移，因此只报告平均值是不够的。

### 7.2 机制指标

每个 task/method/seed 至少报告：

- reference heterogeneity；
- planned 与 realized ESS/entropy/top-mass；
- planned 与 realized unique trajectory/window exposure；
- regime/domain/frequency coverage；
- D_plan 和 D_delivered；
- pseudo-cutoff stability；
- source-only future consistency 或 frozen-model proxy loss；
- 这些量与最终 improvement 的关系。

机制图应展示完整链，而不是只给 heterogeneity 与最终 metric 的一条相关线。

### 7.3 成功标准

完整方法要被认为成功，至少应满足：

1. 相比固定 OT，negative-transfer rate 和 worst-task regret 明显改善；
2. 低风险 task 上保留大部分 OT 正收益，而不是全部退化成 Rand；
3. gate 的判断只使用 future-blind 信息；
4. 关键结果跨 paired seeds、matched checkpoints 和主要指标稳定；
5. 至少在留出 task family 或 confirmation task 上复现；
6. 中间机制指标按预期变化，而不只是最终分数偶然变好。

### 7.4 失败与止损条件

- heterogeneity 不预测 concentration、regime conflict 或 delivery mismatch：停止主张该机制链；
- matched-concentration Random 不复现 OT 退化：不能把问题简单归因于集中度；
- shrinkage 只让所有方法趋近 Rand、不能保留正向 task 收益：gate 没有提供选择价值；
- gate 在 discovery task 有效、在留出 family 失效：不要声称泛化；
- 关键结论只存在于单 seed、单 checkpoint 或单 metric：不作为论文主张；
- Embedding baseline 只改善表示距离：不进入生产路线；
- 两种机制都需要复杂高自由度模型才能勉强拟合 20 个 task：优先保留机制分析，不制造黑盒方法。

## 8. 两个月执行顺序

### 第 1 周：补齐测量

先实现离线 analyzer，复用现有 training_audit 与 transfer artifact，输出 planned/realized ESS、entropy、top-mass、unique exposure、重复率和 delivery fidelity。只有现有记录无法恢复关键量时才增量扩展 audit schema；不改变训练采样行为。

### 第 2 周：全 task 离线链路分析

对 20 个 task 检查 heterogeneity -> concentration/regime conflict -> delivery 的关系，预先固定代表 task 象限和后续控制。

### 第 3--4 周：关键控制实验

完成 matched-concentration Random、shuffled-reference、candidate-uniform 和 lambda sweep。先单 seed 筛选，再对决定性对照使用至少三个 paired seeds。

### 第 5 周：实现唯一被支持的干预

优先实现 Selective Rand-OT 的 reliability cap 与 diversity projection。只有 delivery mismatch 有清晰证据时才改 local-window matching。

### 第 6--7 周：全量确认

完成 20-task matched comparisons、leave-one-task-family-out、negative-transfer rate、worst-task regret 和机制链复核。少量 46M task 只做方向确认。

### 第 8 周：锁定实验与写作

停止增加表示或方法组件，集中整理 protocol、seeds、checkpoints、失败案例、机制图、限制和可复现实验记录。

## 9. 建议的论文叙事

### 9.1 一句话定位

> **我们研究 future-blind、target-conditioned 的 TSFM 训练数据迁移何时会因过度自信的 source exposure 而产生负迁移，并提出一种在 Rand 与 OT 之间可解释地选择、收缩或拒绝迁移的方法。**

### 9.2 三项候选贡献

1. **现象与边界：** 证明静态 descriptor alignment 在 20 个 FEV task 上并不是 forecast utility 的充分条件，并识别 reference heterogeneity 这一稳定风险信号。
2. **机制辨识：** 通过 concentration、realized exposure、delivery fidelity 和 matched controls，追踪 target-conditioned weighting 如何改变模型实际看到的数据并产生负迁移。
3. **选择性迁移：** 提出 future-blind、低自由度的 Selective Rand-OT，在可信时保留 OT，在不可信或过度集中时向 Rand 收缩。

其中第一项已有较强观察证据；第二、三项必须等待控制实验和留出验证，不能提前写成既定事实。

### 9.3 与其他工作的区别

论文不主张首次使用 embedding、nearest-neighbor、OT 或数据选择。区别应明确落在：

- training-time source exposure，而不是 inference-time retrieval；
- reference future 不可见的 target-conditioned transfer；
- 从 planned distribution 到 delivered local windows 的机制审计；
- Rand 作为安全先验的 selective transfer；
- negative-transfer rate 和 worst-task regret，而不只是平均预测分数。

### 9.4 为什么这是一条有特色的路线

大量相关工作致力于找到“更相似”的样本。本路线研究的是另一件事：

> **相似性方法什么时候不值得相信，以及如何让数据迁移保留拒绝权。**

这使 16D 的可解释性、当前失败案例和尚缺的 exposure 指标共同成为研究资产。即使最终 gate 的提升有限，只要机制控制严谨，这条路线仍能产出可信的适用边界和负迁移分析；而简单 Embedding 替换如果效果有限，通常只会留下一个难解释的负结果。

## 10. 本轮检索来源

### 10.1 直接相关方法

- [Retrieval Augmented Time Series Forecasting / RAF](https://arxiv.org/abs/2411.08249)
- [Retrieval-Augmented Diffusion Models for Time Series Forecasting / RATD](https://arxiv.org/abs/2410.18712)
- [TimeRAG: Boosting LLM Time Series Forecasting via Retrieval-Augmented Generation](https://arxiv.org/abs/2412.16643)
- [Optimizing the Training Diet: Data Mixture Search for Robust Time Series Forecasting](https://arxiv.org/abs/2512.11546)
- [Measuring Time-Series Dataset Similarity using Wasserstein Distance](https://arxiv.org/abs/2507.22189)
- [Model Selection with a Shapelet-based Distance Measure for Multi-source Transfer Learning in Time Series Classification](https://arxiv.org/abs/2409.20005)
- [Stationarity-Aware Retrieval-Augmented Time Series Forecasting / SARAF](https://arxiv.org/abs/2606.04135)
- [TSDS-Toolbox: A Toolbox for Measuring Time-Series Dataset Similarity](https://arxiv.org/abs/2608.08119)

### 10.2 Retrieval、representation 与迁移方向

- [TS-RAG: Retrieval Augmented Generation for Time Series Forecasting](https://arxiv.org/abs/2608.06223)
- [Align-RAG: Alignment Is All You Need for TSFM In-Context Learning](https://arxiv.org/abs/2608.05571)
- [CrossRAG: Retrieval-augmented forecasting with future-consistent contrastive retrieval](https://arxiv.org/abs/2607.29459)
- [TS2Vec: Towards Universal Representation of Time Series](https://arxiv.org/abs/2106.10466)
- [Unsupervised Representation Learning for Time Series with Temporal Neighborhood Coding](https://arxiv.org/abs/2106.00750)
- [Review of Data-centric Time Series Analysis from Sample, Feature, and Period](https://arxiv.org/abs/2404.16886)
- [Estimating Time Series Foundation Model Transferability via In-Context Learning / TimeTic](https://arxiv.org/abs/2509.23695)

### 10.3 检索方法与限制

原检索使用 arXiv API、Crossref API 和 ar5iv 论文 HTML 页面。内置交互式浏览器在该轮环境中没有可用实例，因此没有使用浏览器页面操作。Semantic Scholar 请求受到 429 限流，OpenAlex 请求出现 TLS/连接失败；这些限制不影响上述 arXiv/Crossref 结果，但意味着本文不是所有学术数据库的穷尽检索。

最新的 2026 年工作中，部分仍是 arXiv 预印本，不能与已正式发表的论文等同对待；不过对于评估“投稿时是否容易被指出已有相似思想”，公开预印本仍应纳入 prior-art 风险判断。

## 11. 最终路线决策

当前最高性价比的选择是：

> **停止把 Embedding 替换作为近期主线。固定并保护 16D 可解释空间，优先闭合“reference 异质性 -> source 集中/冲突 -> delivered-window 多样性或效用 -> 负迁移”机制链，再把机制转化为知道何时偏向 Rand、何时偏向 OT、何时收缩的选择性迁移方法。**

近期所有新增实验都应服务于以下三个问题：

1. OT/KDE 是否比 Rand 更集中，集中是否随 reference heterogeneity 增强；
2. 这种集中是否真实改变了 delivered local windows 的覆盖、重复和 utility；
3. 在保持其他条件不变时，控制 concentration 或 lambda_task 是否按预测方向改变最终性能。

如果这三步成立，TS3 的独立贡献来自一套可解释、future-blind 的 negative-transfer diagnosis 与 selective transfer 方法；如果不成立，也能用控制实验明确排除集中机制，并把后续资源定向到 delivery mismatch、regime conflict 或表示不足。无论哪种结果，都比现在直接切换到 Embedding 更可解释、更节省时间，也更能利用已有实验资产。
