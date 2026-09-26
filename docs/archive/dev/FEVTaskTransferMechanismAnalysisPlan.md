# FEV Task 上 OT/KDE 相对 Random 差异的机制诊断与验证计划

> 历史归档：保留当时的设计、操作或实验证据，不作为当前执行指令。当前入口见 `docs/README.md`。

> 本文档是后续开发与实验指导，不是当前阶段的分析结果。当前只定义问题、机制假设、审计接口、验证顺序和验收标准；不启动训练、评估或离线统计。

## 0. 先用大白话理解

如果只记住一句话，请记住：

> Ours 的风险不是“不会找相似数据”，而是“它可能非常有信心地找到了看起来相似、但对未来预测并不真正有用的数据”。

可以把三种方法理解成学生复习：

- **Random**：从一个大题库里比较均匀地做题。它不够聪明，但见过的题型多，所以比较稳。
- **OT/KDE（Ours）**：先看这次考试的大纲，再多做看起来最像大纲的题。大纲准确时，复习很有效；大纲不准时，会集中做错方向的题。
- **FEV task**：每个 task 都是一场不同的考试。虽然题库、学生、复习时间都一样，但不同考试的大纲、题型变化速度、题目长度和隐藏条件不同，所以“聪明挑题”有时比随机做题好，有时反而更差。

### 0.1 我目前最怀疑的七个简单原因

| 要检查什么 | 通俗解释 | 如果成立，Ours 为什么会输给 Random |
|---|---|---|
| 安全历史是否代表后面的考试 | Ours 用第一个窗口的过去数据当大纲，但真正评估的是很多后续窗口 | 大纲过期后，Ours 仍按旧大纲挑题；Random 因为题型广，反而更抗变化 |
| 六维特征是否太粗糙 | “趋势强”不等于同样上涨；“周期强”不等于同样周期长度 | Ours 把本质不同的序列当成相似序列，集中学习错误模式 |
| 整条 source 与局部窗口是否一致 | Ours 按整本书挑书，模型却只随机读其中一页 | 书的主题对，但随机读到的那一页可能和考试无关 |
| 是否抽得太集中 | Ours 反复选少数 source，Random 广泛覆盖 source | 若挑选不完全正确，重复学习少数模式会伤害泛化 |
| future 是否由没看到的条件决定 | 促销、节假日、天气、门店等可能决定答案 | 历史销量看起来相似，不代表未来条件相同；Ours 因此“自信地匹配错” |
| 时间尺度是否不一样 | 每天、每周、每小时的重复规律不能混为一谈 | 六维只看“季节性强不强”，可能没看见“多久重复一次” |
| 指标是否放大少数坏窗口 | 总 SQL 是很多窗口和 series 混合后的分数 | Ours 也许只伤害少数重要窗口，但总分看起来退化很大 |

### 0.2 最重要的检查顺序

后续不要先花时间问“source 和 reference 的总体距离还有没有下降”。这个已经不是最关键的问题。优先顺序应是：

1. **安全历史是否代表真实预测窗口和未来答案。**
2. **Ours 选中的 source 是否真的交付了相似的局部训练窗口。**
3. **Ours 是否因为太集中而失去训练多样性。**
4. **六维特征是否遗漏方向、周期长度、最近变化和外部条件。**
5. **最终坏结果是否稳定存在于多数窗口、series、seed 和指标中。**

### 0.3 什么证据最能说明不同原因

| 看到的现象 | 最直观的含义 |
|---|---|
| 第一段安全历史和后面窗口很不一样 | Ours 的“考试大纲”过时 |
| 整条 source 很像 reference，但训练窗口并不像 | Ours 选对了书、却没有读到对的页 |
| Ours 少数 source 的出现次数极高 | Ours 复习范围太窄 |
| 打乱 reference 后仍然和 Ours 一样好 | Ours 的收益可能不是来自真正的 task 匹配 |
| 一个同样集中的 Random 也变差 | 主要问题是过度集中，不一定是 OT/KDE 的相似性定义 |
| 加入周期长度或最近趋势后结果变好 | 原六维特征缺了时间尺度或局部状态 |
| 只有 SQL 变差、MAE/WQL 不同方向 | 需要先检查指标聚合，而不能立刻说方法失败 |

### 0.4 当前最简主解释

目前最合理、但仍需要验证的解释是：

> 对 task17、task15 这类 task，过去历史本身包含对未来有用的结构，所以 Ours 选到更相关的训练题，效果好；对 task14 一类 task，过去 target 的形状不足以说明未来，或者第一个历史不能代表后续窗口，Ours 因而过度偏向“表面相似”的 source，损失了 Random 的广覆盖优势。

后面的章节会把这个简单说法变成可以实际检验的指标、数据记录和控制实验。阅读顺序建议是：先读本节，再读第 4、6、7 节；需要实施时再阅读 Phase A--D。

## 1. 目标与核心判断

实验固定了 source、Chronos、训练预算、窗口规格和评估流程，OT/KDE 与 Random 的主要区别是 source trajectory 的曝光概率。因此 task 差异只能沿这条链产生：

```text
FEV task -> reference 安全历史 -> 六维表示 -> OT/KDE 概率
         -> source trajectory -> 局部训练窗口 -> 梯度/模型 -> 评估误差
```

真正的问题不是“距离是否下降”，而是：六维 reference 是否代表模型真实需要预测的对象；trajectory-level 重加权是否交付了相似的局部窗口；相关性收益是否超过多样性损失。

类比：task 是考试，reference 是大纲，六维特征是大纲的六个标签，OT/KDE 是按标签挑书，source trajectory 是整本书，训练窗口是随机翻到的一页，future 是试题。整本书标签相似，不保证随机翻到的页面包含考点；只读少数几本书也可能失去通用能力。

## 2. 当前流程中的关键断点

当前 reference extractor 只读取 `task.get_window(0)` 的 past target；evaluator 却遍历所有 `task.iter_windows()`，并通过 `as_univariate=True` 只把 target 交给 Chronos。至少存在：

- `H0`：OT/KDE 使用的第一个完整安全历史；
- `R0`：`H0` 最后 512 点，首个窗口真正看到的 context；
- `Rk`：第 k 个 rolling window 最后 512 点；
- `Yk`：第 k 个窗口的真实 future；
- `E`：所有 future 按时间拼接的 evaluation tail。

于是 `H0` 可能不代表 `R0`，`R0` 可能不代表后续 `Rk`，而 `Rk` 与 `Yk` 的关系也可能因 task 而不同。

当前六维特征还存在 protocol 风险：STL/period 相关路径观察前缀（最长 50K），shift/event 使用完整序列；`period` 主要是周期强度而不是周期 lag；没有直接编码趋势方向、shift 位置、局部尺度、零值和尾部。

## 3. 机制假设矩阵

| 假设 | 可能机制 | 决定性验证 |
|---|---|---|
| Reference 过时 | `H0` 不代表 `Rk/Yk`，重加权方向错误 | 测量 `H0→R0→Rk→Yk` 漂移及其与收益关系 |
| 表示不充分 | 相同六维强度不代表方向、周期长度、局部 regime 或 future 转移 | 加入 trend direction、period lag、recent shift、scale/zero/tail 消融 |
| trajectory/window 错配 | 匹配整条 source，实际随机抽局部 512+64 窗口 | 比较 trajectory distance 与 delivered-window distance |
| 多样性坍缩 | `lambda=0.8` 提高相关性但降低 ESS，少数 source 重复曝光 | ESS、entropy、top-k mass；matched-concentration Random |
| 时间尺度错配 | task seasonality/horizon 与训练 512+64 几何不同 | 检查 `horizon/period`、`context/period`、`horizon/64` |
| 条件变量缺失 | promo、holiday、weather、static identity 主导 future，target-only 无法恢复 | target-only 与 covariate-conditioned predictability 对照 |
| token/scale 差异 | per-window scaling 不消除零膨胀、尖峰、clipping、scale ratio | 审计 Chronos context scale、zero ratio、future/context scale |
| metric 聚合放大 | 少数 window/series/quantile 主导 SQL | 分解到 window、series、target、horizon step、quantile |
| 训练随机性 | 单 seed/checkpoint 偶然 | paired multi-seed、曲线 AUC、bootstrap CI |

`shift` 当前是最强任务级关联变量，但暂时只能视为代理变量：它可能代表 target history 自身有可迁移 regime 信息，也可能只是与低噪声、趋势型和 target-only 可预测 task 共现。

## 4. 一致性审计设计

不要要求历史与 future 的六维数值相等；要判断历史是否能预测 future 的结构或转移方式。审计分四层：

1. `Z(H0)` vs `Z(R0)`：提取对象是否等于模型输入对象；
2. `Z(R0)` vs `Z(Rk)`：第一个 reference 是否在 rolling 评估中失效；
3. `Rk` vs `Yk`：历史是否能预测答案的结构与变化方向；
4. `Z(H0)` vs `Z(E)`：安全历史是否代表整个 held-out 区域。

对足够长的片段计算六维特征，并记录 `absolute_delta`、`signed_delta`、六维 L2、逐维相关性、窗口间方差和 feature failure reason。对短 horizon 不能机械调用 STL/ACF：horizon 只有 5、13、24 时，得到的数字不一定有同样统计语义。所有窗口另外计算 normalized slope、direction、curvature、level/scale ratio、volatility ratio、zero ratio、intermittency、tail heaviness、seasonality-lag correlation、边界 jump、event/outlier rate 和 missing pattern。

为避免泄漏，在 `H0` 内构造 pseudo-cutoff，用历史内部过去片段预测历史内部伪 future，估计 history feature 到 future dynamics 的稳定性。真实 `Yk` 只用于事后审计，不能进入 source selection。

## 5. Representation 与时间尺度探索

候选补充信息包括：

- trend direction、normalized slope、curvature、monotonicity、recent/global slope；
- detected period lag、task seasonality、可见完整周期数、`horizon/period`；
- recent shift、shift location、direction、regime duration、boundary proximity；
- event duration、cluster、positive/negative ratio、离预测边界距离；
- local scale、zero ratio、tail、intermittency、token clipping proxy。

采用 feature group 的 add-one 与 leave-one-out 消融，不一次加入所有特征。`period` 强度与周期长度必须分开：两首歌都节奏强，不代表每秒拍数相同；同理两条序列季节性强，不代表 Chronos 应学习同一个时间尺度。

## 6. Delivery Fidelity：整条 trajectory 是否交付了正确窗口

定义：

```text
D_trajectory = reference 与加权 source trajectory 的距离
D_window     = reference/model-context 与实际训练局部窗口的距离
delivery_fidelity = D_trajectory 的改善能否传递为 D_window 的改善
```

如果 `D_trajectory` 降低但 `D_window` 不降，说明 planner 数学目标没有传递到模型输入；如果 `D_window` 变差，说明全局 trajectory 特征误导了局部窗口。后续比较 full-trajectory、last-512、trajectory 内多窗口分布三种 protocol。

## 7. 相关性收益与多样性损失

重加权的净收益可写成：

```text
task-relevant exposure gain - diversity loss - repeated exposure cost
```

必须记录 `ESS=1/sum(pi_i^2)`、entropy/perplexity、最大 probability、最大 exposure ratio、top-10/100/1000 mass、非零 source 数、domain/frequency/length 的加权组成和重复窗口期望次数。

必须加入三个控制：

1. **Shuffled-reference OT/KDE**：保留算法和集中度，打乱 task-reference 对应关系；若仍有效，收益不是 task-specific similarity。
2. **Matched-concentration Random**：保持 ESS/entropy/top-k mass，但不使用 reference；若同样回退，主因是多样性损失。
3. **Lambda sweep**：`0.0, 0.2, 0.5, 0.8, 1.0`；先好后坏表示 relevance-diversity trade-off，所有 lambda 都坏则更像表示或 reference 方向错误。

## 8. Covariate 与 target-only 协议

“存在 covariate”不是充分解释；需要测量 future 中有多少信息可由 target history 恢复，多少只能由 known/past/static 条件解释。重点同源对照：Rossmann 1W/1D、hospital 1W/1D、ETT 15T/1H，以及 Boomlet 三个 task。Rossmann 尤其适合检验长 daily horizon 和 promo/open/holiday/store identity 是否让 target-only matching 产生 negative transfer。

## 9. 分阶段开发计划

### Phase A：只读审计

新增审计数据模型，不改变训练语义：

- `FEVWindowAuditRecord`：task/window/item/target、H0/R0/Rk/Yk/E 边界、特征和转移统计；
- `SourceExposureAuditRecord`：trajectory probability、exposure ratio、domain/frequency/length、global 与 local window 特征；
- `TaskMechanismSummary`：consistency、rolling stability、history-to-future predictability、delivery fidelity、ESS、time-scale geometry、target-only predictability。

### Phase B：代表性 task 干预

先选 task08/09、13/14、15、17、07、12 或 16。每个 task 比较 Random、当前 OT/KDE、shuffled-reference、matched-concentration Random、local-window matching 和 lambda sweep；至少三个 paired seeds。

### Phase C：feature group 消融

按 temporal scale、trend shape、local regime、distribution shape、conditional context 分组进行 add-one 与 leave-one-out，判断新增信息和真实依赖。

### Phase D：全量确认

只有机制在代表性 task 上通过后才扩展 20 task。预先锁定 seeds、checkpoint 规则、主指标、异常处理和 task-family 分组，使用 leave-one-task-family-out，避免用同一批 task 既发现又确认 gate。

## 10. 结果解释表

| 观察 | 更支持的解释 |
|---|---|
| trajectory 距离降、window 距离不降 | trajectory-window 错配 |
| window 距离降但 ESS 极低 | 多样性坍缩或表示错误 |
| shuffled 与真实一样好 | 非 task-specific 相似性收益 |
| matched Random 同样回退 | 集中度/重复曝光代价 |
| 小 lambda 好、大 lambda 坏 | relevance-diversity trade-off |
| `H0→Rk/Yk` 一致性越高收益越大 | reference 时效性是关键 gate |
| 加 period lag 后同源高低频差异缩小 | 时间尺度缺失 |
| 加 recent shift 后 shift task 稳定 | 全局 shift 缺时间定位 |
| target-only predictability 低的 task 普遍回退 | 条件变量缺失 |

## 11. 机制成立的验收标准

一个变量不能仅凭 task-level 相关性被宣布为根因，必须同时满足：

1. 能预测 Ours 相对 Random 的方向，并在留出 task family 上成立；
2. 确实改变 OT/KDE 的 source exposure；
3. exposure 改变确实传递到 delivered local windows 和 token/gradient 分布；
4. 通过 lambda、feature group、window-level matching 或 concentration control 干预后，性能按预期变化，并跨 seed/window/主要指标稳定。

目标闭环是：

```text
task 属性 -> reference/表示 -> source exposure -> 局部训练窗口
          -> 梯度/模型行为 -> evaluation outcome
```

## 12. 最终目标：可泛化的 Task Gate

最终产物不是事后解释 task17 和 task14，而是一个训练前可用、无 future 泄漏的 gate：

```text
if reference_to_context_consistency 低: Random
elif history_to_future_predictability 低: Random 或小 lambda
elif delivery_fidelity 低: local-window matching
elif ESS 太低: 降低 lambda 或正则化
else: OT/KDE
```

gate 必须用 leave-one-task-family-out 验证。最终成功标准是能够回答：某个 task 的 reference 为什么改变了这些 source 的曝光；这些 source 交付了什么局部窗口；这些窗口为什么帮助或伤害真实答案；干预其中一个环节后结果是否按机制预期改变。

## 13. 实施纪律与预期产物

审计脚本与正式训练代码分离；记录所有时间边界、period protocol、segment length 和 failure reason；不使用真实 future 参与 selection；使用 paired seeds 和 window/series 分解；不把 solver convergence、support coverage 或距离下降单独当作预测成功。

预期产物包括：reference/evaluation consistency audit、trajectory/delivered-window audit、task mechanism matrix、per-window/per-series evaluation decomposition、paired multi-seed controls、feature/lambda ablation、leave-one-family-out gate，以及对反例和剩余不确定性的记录。
