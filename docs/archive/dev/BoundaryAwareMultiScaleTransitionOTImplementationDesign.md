# DynamicTransition OT 首版实现文档

> 历史归档：保留当时的设计、操作或实验证据，不作为当前执行指令。当前入口见 `docs/README.md`。

> 状态：Draft 0.2。本文档是独立的实现文档，不修改上游方法设计文档。
>
> 上游设计：docs/dev/BoundaryAwareMultiScaleTransitionOTDesign.md
>
> 当前证据：docs/Progress.md、9M 六策略归档和相关机制分析文档。

本文档把 BAMT-OT 首版的实现范围、核心 DAG、数据契约、缓存边界、抽象类层级和测试要求集中记录下来。后续实现应先更新本文档，再修改生产代码。

首版改动主线严格只有三步：

1. 在 Feature 阶段增加 segment 内 A/B transition 分析，并把结果写入 Feature Bundle；
2. 在 cache build 阶段读取 transition 结果、完成 candidate/OT 求解、按 confidence mode 生成最终质量并写出 cache；
3. 用新的 transition 公式完全替代旧 Dynamic 方法，训练期直接使用 cache 中的最终质量。

其它内容只能服务于这三步，不另行引入独立的风险模型、训练期 annealing 或额外的多尺度流程。

## 0. 更新协议

每次更新按以下顺序进行：

1. 先在决策记录中记录新增或变更的决定；
2. 再更新 DAG、数据契约和接口命名；
3. 再更新测试与验收条件；
4. 最后才修改 ts3、configs 或 tests；
5. 未确认的条目不得由实现者自行解释为已冻结参数。

用户后续修正覆盖本文档中更早的推断。上游设计文档中的建议值，不自动视为首版已确认值。

## 1. 首版目标与边界

### 1.1 首版真正实现的对象

正式 policy ID 为 d-ot；matched-support 对照 policy 为 d-candidate-uniform。两者都不再使用旧 DynamicPair 的相邻段语义：

- 一个当前 AnalysisSegment 就是一个 transition pair；
- 当前 AnalysisSegment 长度为 512；
- 前半段 A 为 segment 内的前 256 个点；
- 后半段 B 为 segment 内的后 256 个点；
- A -> B 是这个 segment 的唯一变化；
- 首版只计算一个尺度；
- 多尺度扩展使用 `segment_specs` 列表；每项包含 `name`、`segment_length`、`segment_stride` 和 `active`，首版列表中恰好只有一个 active spec；
- 规划输出继续映射到现有 segment-level sampler，因此当前实现称为 segment-proxy DynamicTransition OT；
- Feature 阶段从 Load shard 读取 raw A/B，计算并写入 Feature Bundle v2；cache build 阶段只读取已持久化的 transition 字段，不重新计算 raw A/B。
- raw A/B 数值不写入 Feature Bundle、sampling cache 或训练期对象；Feature Bundle 只保存 descriptor 和辅助量结果。
- 标准 source segment 的 geometry 由现有 segment 枚举保证为完整 `512` 点；末尾不足时使用从 trajectory 末尾向前回溯的重叠 segment。只有特殊 reference 兼容路径允许实际长度小于 `512` 的 segment。
- 特殊短 reference segment 直接按实际长度切分：奇数长度时多出的一个点归前半段；任一半小于现有 descriptor 最小有效长度 `8` 时，该 segment invalid。

首版不实现上游文档中同时扫描 32、64、128、256 的多尺度流程。多尺度是后续扩展，不得通过段内多 horizon 重新引入。

### 1.2 明确禁止的语义

- 不比较原始序列的 Naive 数值距离；
- 不把相邻两个 AnalysisSegment 组成新的 pair；
- 不使用旧 DynamicPairOTPlanner 的 state-only candidate 逻辑；
- 不把 pair 质量折回 anchor，再对 terminal segment 做静态 fallback；
- 不使用训练 future、held-out label、SQL、WQL 或 checkpoint 结果选择候选、OT 参数或 Confidence；
- 不把 ESS、entropy、top-mass 等当前 20-task 实验不能解释 allocation 增益或回退的量，凭直觉组合成风险分数。
- 任意一部分分析无效时，整个 segment 对所有 sampling method 都 invalid；不能只对 d-ot 特殊保留。
- Load 只负责检测 `NaN`/`Inf`；不在 Feature 阶段对 Reference 做非有限值 masking，不用填充值修复原始数据。

### 1.3 当前 20-task 证据的角色

20 个 9M、from-scratch、20K、单 seed 小模型任务是首版经验依据，不是普适定理。当前实现把它们作为当前协议下的 provisional calibration：

- SQL 是主要结果指标；
- WQL、MAE 等保留为冲突诊断；
- candidate-uniform 与 full OT 必须作为同 support 的两个独立目标保留；
- Top-256 是首版 transition candidate 的固定默认值；
- Top-64 与 Top-1024 作为未来敏感性接口，不在首版自动按任务结果选择；
- candidate support 相对 Rand 在当前可配对任务中多数获益但并不安全；
- full OT 相对同支持 candidate-uniform 既有增益也有回退，不能宣称 OT allocation 普遍可靠；
- `auto_confidence` 只能使用当前 cache build 的 future-blind OT 数值诊断推断，不能使用训练结果反向调节；`effective_confidence` 手动填写时不启用 Auto Confidence。
- `auto_confidence` 与 `effective_confidence` 是互斥的两个配置字段；Random 两者必须都为空，任意 OT policy 必须且只能填写其中一个。

### 1.4 配置与策略边界

首版继续使用 sampling.dynamic 这个配置块，但它的内容完全重写；旧 DynamicConfig 的字段不再保留。d-ot 与 d-candidate-uniform 都使用同一套 dynamic 和 ot 配置：

- OT policy：执行对应 pure target、uniform-confidence smoothing 和 confidence mode；
- d-candidate-uniform：执行同一 candidate builder，但输出严格的 matched-support candidate-uniform control；
- 所有 sampling method 都删除 `sampling.lambda` 和训练期 lambda annealing；最终 cache mass 在 cache build 阶段一次性确定，训练期不重新混合；
- `auto_confidence` 或 `effective_confidence` 按全局 confidence 契约控制所有 OT policy；Random 不读取任何一个字段；
- 所有 transition、candidate、confidence 和 identity weight 参数集中写在 `configs/scripts/run.template.yaml` 的共享配置块中，方法块只选择 policy，不允许方法块重复覆盖这些参数；
- 配置只做必要的字段存在、类型、范围、六项权重和为 1、以及唯一 active spec 检查，不做全量 exact-schema 或配置 hash 兼容比较；
- training WindowAccessor 和 Window 交付流程保持不变。

### 1.5 集中式 run 配置入口

实现必须把所有用户可控制的首版参数集中暴露在 `configs/scripts/run.template.yaml`。方法条目只声明 policy，不能在方法条目中复制 DynamicTransition 参数。首版共享配置的字段形状固定为：

~~~yaml
sampling:
  auto_confidence: null       # true for Auto Confidence; null for manual mode
  effective_confidence: null  # float in [0, 1] for manual mode; null for auto mode
  dynamic:
    segment_specs:
      - name: default
        segment_length: 512
        segment_stride: 512
        active: true
    candidate_top_k: 256
    weight_before: 0.20
    weight_after: 0.35
    weight_change: 0.35
    weight_level: 0.04
    weight_scale: 0.04
    weight_constant: 0.02
  ot:
    # Existing OTConfig solver fields remain explicitly visible here.
    solver: pot_sinkhorn_unbalanced
    feature_metric: squared_euclidean
    epsilon: 0.1
    mass_regularization: 1.0
    max_iterations: 1000
    tolerance: 1.0e-8
    neighbor_count: 64
    support_threshold: 0.5
~~~

首版不使用 temperature、outside floor 或 exposure cap。手动 `effective_confidence` 的合法范围为 `[0, 1]`；`auto_confidence` 仍按其自身的 future-blind 推断规则输出不超过 `0.95` 的值。`auto_confidence=true` 时必须将 `effective_confidence` 留空；手动模式下必须将 `auto_confidence` 留空并填写 `effective_confidence`。两个字段不能同时非空。

## 2. 当前已确认的首版决策

| 主题 | 首版决定 | 备注 |
|---|---|---|
| policy ID | d-ot、d-candidate-uniform | d-ot 是 full OT；d-candidate-uniform 是严格 matched-support control |
| 实现语义 | DynamicTransition | 替代旧 Dynamic pair 语义；所有 sampling method 都不再使用 lambda |
| 配置块 | sampling.dynamic | 保留 wire 名称，但完全替换旧 Dynamic 字段 |
| 配置继承 | DynamicOTConfig extends OTConfig | OT solver 默认实现和当前静态 OT 一致；DynamicOTConfig 增加 Dynamic 专有参数 |
| 规划粒度 | 当前 AnalysisSegment | 一个 segment 对应一个 transition pair |
| segment geometry | 512，前后各 256 | 复用现有 ANALYSIS_SEGMENT_LENGTH/STRIDE |
| active scale | 单尺度 | `segment_specs` 为列表；首版恰好一个 active spec |
| identity | 16 + 16 + 16 + 3 = 51D | A、B、delta、level、scale、constant |
| identity 存储 | Feature Bundle v2 命名字段 | `before_features`、`after_features`、`change_features`、`level_shift`、`scale_change`、`is_near_constant`；新增字段 float32，旧 full-segment `feature_values` 保留 |
| raw distance | 禁止 | 只按 51D identity 匹配 |
| candidate budget | Top-256 | 固定首版默认值 |
| candidate parameter ownership | `dynamic.candidate_top_k` | d-ot 不使用 `OTConfig.neighbor_count` 或 `support_threshold`；静态 OTConfig 语义保持不变 |
| candidate retrieval | exact blocked Top-K | GPU blocked exact，不能让 hash 决定资格 |
| reference aggregation | joint reference OT | 多 reference 合并为一个联合 reference mass |
| reference mass | 每个 reference catalog 等质量 | catalog 内对有效 segment 按原始 mass 比例重归一化；空 catalog 直接失败 |
| source invalid mass | 有效 source segment 按原始 base mass 比例重归一化 | invalid segment 在完整 cache topology 中质量为 0 |
| comparison target | candidate-uniform 与 full OT 都保留 | 同 support 对照不可删除 |
| confidence mode | `auto_confidence` 或 `effective_confidence` 二选一 | Random 两者都为空；任意 OT policy 必须且只能填写其中一个 |
| effective confidence range | 手动 `[0, 1]`；Auto 不超过 `0.95` | `1.0` 表示不做 uniform-confidence smoothing |
| uniform-confidence smoothing | 全局有效 source segment 均匀先验 | 所有 OT policy 都使用 `q_final = c_eff * q_pure + (1-c_eff) * u_valid`；不再使用 outside floor 或 exposure cap |
| identity weights | 默认 0.20/0.35/0.35/0.04/0.04/0.02 | 可由配置覆盖；缺省时使用该默认值 |
| invalid segment | 所有方法统一 invalid | Load 过滤非有限 trajectory；Feature/transition 任一部分失败则 segment 质量置零并从有效分析集合移除 |
| feature asset migration | 静态方法兼容旧版，d-ot 强制新版 | d-ot 必须检测 transition schema；静态方法不要求新字段 |
| planner hierarchy | OT planner common base | 静态 OT、静态 candidate-uniform、Dynamic OT、Dynamic candidate-uniform 为同一 OT base 的同级子类 |
| sampling.lambda | 所有 sampling method 删除 | 不保留任何训练期 annealing 语义 |
| config validation | 最小必要校验 | 字段、类型、范围、权重和、唯一 active spec，以及 Random/OT 的 confidence 互斥规则；不做全量 exact-schema/hash 比较 |
| empirical scope | 当前 9M from-scratch 20K 协议 | provisional，不宣称跨初始化或训练阶段普适 |

rho_C 定义为 candidate support 在 Random 或 base segment mass 下的覆盖，作为 candidate diagnostics：

~~~text
rho_C = sum(base_mass[i] for i in candidate_support)
~~~

`rho_C` 不再单独生成 confidence threshold，也不读取训练 outcome；Auto Confidence 的具体推断只使用当前 cache build 的 OT 数值诊断。

## 3. 核心方法流程 DAG

### 3.1 三阶段核心 DAG

~~~mermaid
flowchart TD
    L[Load: mark NaN/Inf trajectory dropped] --> F0[Feature stage: current segment geometry]
    F0 --> H[Read A/B halves with existing/DynamicShardDatasetReader]
    H --> I[Compute full feature and 51D transition identity]
    I --> V[Feature Bundle v2; invalid segment excluded]

    V --> S0[DynamicTransitionFeatureSpace]
    S0 --> C[Source-only MAD calibration]
    C --> D[Exact blocked transition distance]
    D --> K[Reference Top-256 candidate retrieval]
    K --> S[Joint candidate support]
    S --> U[Candidate-uniform target]
    S --> O[Joint unbalanced OT target]
    U --> CU[d-candidate-uniform pure target]
    O --> SM[Uniform-confidence smoothing]
    O --> AC[Auto Confidence from current OT diagnostics]
    CFG[Global auto_confidence / effective_confidence] --> CM[Mutually exclusive confidence mode]
    AC --> CM
    CU --> CM
    S0 --> UP[Uniform prior on valid source segments]
    CM --> SM
    UP --> SM
    SM --> M[Final target mass]
    CU --> AR[Immutable segment sampling cache]
    M --> AR
    AR --> SP[Existing segment sampler]
    SP --> TR[Chronos training]
~~~

### 3.2 首版实际执行顺序

1. Load 阶段对 source/reference 一律检测 `NaN`/`Inf`，保留原始 shard bytes 但在 Load index 标记 dropped；
2. 用现有 Bundle geometry 枚举合法 AnalysisSegment；标准 source segment 由末尾回溯保证完整长度；
3. Feature 阶段读取 full segment 与 A/B halves；特殊短 reference segment 按实际长度切分，奇数多出的点归前半段；
4. 同时计算旧 full-segment 16D feature 与 A、B 的 16D production descriptor；任一部分失败则该 segment 对所有方法 invalid；
5. 构造并持久化 51D DynamicTransition identity；invalid segment 不进入有效 transition space；
6. cache build 读取 Feature Bundle v2，构造 `DynamicTransitionFeatureSpace`；不重新读取 raw A/B；
7. 对有效 source identity 的每个分量计算 source-marginal MAD，MAD 过小时尺度 fallback 为 `1.0`；
8. 对每个 reference segment 做 exact blocked Top-256；
9. 合并所有 reference 的 candidate ID，形成 joint support；每个 reference catalog 等质量，catalog 内有效 mass 重归一化；
10. 在完全相同 support 上生成 candidate-uniform control target；
11. 校验全局 confidence 配置：Random 必须 `auto_confidence=null` 且 `effective_confidence=null`；任意 OT policy 必须且只能填写其中一个；
12. 如果 policy 是 d-candidate-uniform，生成 matched-support 的 `p_C` pure target；如果 policy 是 d-ot，在同一 support 上运行 full OT，生成 `q_OT` pure target；
13. 如果 `auto_confidence=true`，根据当前 cache build 的 OT 数值诊断推断 `effective_confidence`；如果手动填写 `effective_confidence`，直接使用该值，不执行 Auto Confidence；
14. 对所有 OT policy 构造全体有效 source segment 上的 uniform prior，并写出
    `q_final = effective_confidence * q_pure + (1 - effective_confidence) * u_valid`；
15. 写出 base mass 与最终 target mass；invalid segment 在完整 cache topology 中为 0，uniform prior 也不向 invalid segment 分配质量；
16. 求解失败、空 reference 或无有效 support 时直接失败，不静默回退 Random；
17. 训练期只读取最终 segment mass，不重新计算 transition，也不执行 lambda annealing。

## 4. Transition identity 数据契约

### 4.1 逻辑结构

首版每个 segment 的 identity 逻辑结构如下：

~~~text
DynamicTransitionIdentity
  before_features: float32[16]
  after_features: float32[16]
  change_features: float32[16]  # after_features - before_features
  level_shift: float32
  scale_change: float32
  is_near_constant: float32     # 0.0 or 1.0
~~~

固定扁平布局：

~~~text
[0:16]   before_features
[16:32]  after_features
[32:48]  change_features
[48]     level_shift
[49]     scale_change
[50]     is_near_constant
~~~

所有 source/reference identity 必须使用同一字段顺序、同一 feature algorithm version 和同一 segment geometry。identity 的 shape、dtype、有限性和版本必须写入 cache manifest。

Feature Bundle v2 在现有 segment record 中直接增加以下命名字段，不再增加额外的 identity 维度：

~~~text
feature_values:      现有 full-segment 16D feature，保留给静态方法
before_features:     命名的 16D float32 struct
after_features:      命名的 16D float32 struct
change_features:     命名的 16D float32 struct
level_shift:         float32
scale_change:        float32
is_near_constant:    float32
~~~

新增字段直接位于 segment record；不再包一层 `transition_identity`。`change_features` 可取负值；三个辅助量只要求有限，`is_near_constant` 必须为 `0.0` 或 `1.0`。旧 Feature Bundle v1 对静态方法继续有效；d-ot 必须检测并要求 Feature Bundle v2 transition 字段。

### 4.2 A/B 特征计算

- before_features = compute_descriptors(A)；
- after_features = compute_descriptors(B)；
- change_features = after_features - before_features；
- 不调用旧 6D descriptor；
- 不把完整 512 segment 的现有 16D feature 当作 transition identity；
- 不把 A/B raw vector 直接拼进 OT cost。

旧 full-segment `feature_values` 与 transition identity 都必须有效；任一计算失败，整个 segment 对所有方法 invalid。Feature 阶段不得通过对 Reference 的 NaN/Inf 做 masking 来制造有效 descriptor；非有限值已由 Load 阶段统一标记为 trajectory dropped。

Feature 阶段用现有 `ShardDatasetReader`/`DynamicShardDatasetReader` 读取 raw halves；cache build 只读取 Feature Bundle v2 中已经持久化的 transition 字段，不重新读取 raw A/B。

### 4.3 三个辅助量

首版采用设计文档中与身份语义一致的定义：

~~~text
joint_median = median(concat(A, B))
joint_scale = max(1.4826 * MAD(concat(A, B)), epsilon)

level_shift = (median(B) - median(A)) / joint_scale
scale_change = log((MAD(B) + epsilon) / (MAD(A) + epsilon))
is_near_constant = 1.0 if joint_scale == epsilon else 0.0
~~~

`is_near_constant` 是结构性指示量，不是训练 outcome 风险分数。浮点实现应使用与 feature protocol 相同的 `NUMERIC_EPS` 判定规则，不能通过比较未归一化的绝对数值重新定义常量。标准 source segment 使用 512 点；特殊短 reference segment 使用实际长度，前半段长度为 `ceil(length/2)`，后半段长度为 `floor(length/2)`。

### 4.4 Distance contract

source/reference pair 的 cost 只由 51D identity 产生。每个 identity 分量的尺度只从有效 source identity 的边际分布计算，不构造 source-source pair distance：

~~~text
scale_j = 1.4826 * MAD(source_identity[:, j])
scale_j = 1.0 if scale_j < NUMERIC_EPS
~~~

归一化后的逐分量距离使用固定 `Huber delta=1`：

~~~text
d_before[j] = Huber((before_source[j] - before_reference[j]) / scale_before[j], delta=1)
d_after[j]  = Huber((after_source[j] - after_reference[j]) / scale_after[j], delta=1)
d_change[j] = Huber((change_source[j] - change_reference[j]) / scale_change[j], delta=1)
d_level     = Huber((level_source - level_reference) / scale_level, delta=1)
d_scale     = Huber((scale_source - scale_reference) / scale_scale, delta=1)
d_const     = abs(is_near_constant_source - is_near_constant_reference)

d_before = mean_j(d_before[j])
d_after  = mean_j(d_after[j])
d_change = mean_j(d_change[j])

cost = 0.20 * d_before
      + 0.35 * d_after
      + 0.35 * d_change
      + auxiliary_cost
~~~

三个辅助量必须参与匹配，不能只写入诊断；A/B/change 的主权重保持 0.20/0.35/0.35；所有尺度只使用有效 source identity 的边际 MAD。MAD 不是 identity 的新增维度；它只用于 segment 内辅助量计算和距离归一化。

首版默认完整 cost 为：

~~~text
cost = 0.20 * d_before
      + 0.35 * d_after
      + 0.35 * d_change
      + 0.04 * d_level
      + 0.04 * d_scale
      + 0.02 * d_const
~~~

这六个权重总和为 1.0。它们写入默认配置模板；如果运行配置没有提供覆盖值，则使用上述默认值。若配置提供覆盖值，必须提供完整六项、每项非负且总和为 1.0，并把最终值写入 manifest。

### 4.5 Raw descriptor failure

如果 Load 之后某个 A、B、辅助量或旧 full-segment feature 无法生成有效结果，则该 segment 对所有方法 invalid，不丢弃整条 trajectory。Source 和每个 Reference catalog 各自对剩余有效 segment 的原始 mass 重新归一化：

- Load 对任何包含 `NaN`/`Inf` 的 source/reference trajectory 保留原始 bytes，但在 Load index 标记 `dropped=true`、`drop_stage=load`、`drop_reason=non_finite`；Feature 和所有 sampling method 均跳过该 trajectory；
- 有效 segment 仍保留原稳定 ID 和 geometry；
- 失败 segment 不进入 identity、candidate 或 OT；
- invalid source segment 在完整 source topology 中保留稳定 ID 和 geometry，但 base/final mass 都为 0；有效 source segment 按原始 base mass 比例归一化；
- invalid reference segment 在其 catalog 内按原始 reference mass 比例重归一化；
- 任一 reference catalog 没有有效 segment 时，cache build 显式失败；
- failure count、failure reason 和重新归一化质量写入 diagnostics；
- training WindowAccessor 不参与该处理；所有方法使用同一 invalid segment 结果。

## 5. Candidate 与 OT 契约

### 5.1 Candidate support

首版不实现 Hash/LSH 作为方法层依赖。候选模块必须有可替换的 retrieval interface，但默认实现为 exact blocked Top-K：

- query：一个 reference DynamicTransitionIdentity；
- source：全部 source transition identity；
- budget：`dynamic.candidate_top_k`，首版为 Top-256；
- distance：第 4.4 节的完整 transition cost；
- GPU：按 source block 计算，不能分配完整 source-reference dense matrix；
- CPU：使用可验证的 exact fallback；
- tie-break：按稳定 segment ID 排序，保证 determinism。

source segment 数量小于 256 时，Top-K 使用实际数量并记录 effective_k。某个 reference 没有任何有限 candidate 时，整个 build 失败，不静默回退 Random。

多个 reference 的候选集合取 ID union，形成一个 joint support。candidate support 必须记录：

- candidate ID；
- source base mass；
- reference 命中次数；
- 最小和平均 transition distance；
- rho_C；
- 每个 reference 是否有有效 candidate；
- candidate 数量和 source 总 segment 数。

### 5.2 Candidate-uniform target

在相同 joint support 上保留现有 Random/base 几何：

~~~text
p_C[i] = base_mass[i] / sum(base_mass[k] for k in C), i in C
p_C[i] = 0,                                             i not in C
~~~

它是首版的 matched-support control，不得重新检索或重新生成 candidate ID。

d-candidate-uniform 的 pure target 语义是严格 matched-support control：

- `p_C` 是其 `q_pure`，candidate 之外的 pure target mass 为 0；
- 它与 d-ot 使用相同的 global confidence 配置契约；填写 confidence 字段后，同样执行 uniform-confidence smoothing；
- 不执行 temperature、outside floor 或 exposure cap；
- manifest 必须记录它与对应 d-ot artifact 使用相同 candidate support hash。

### 5.3 Full OT target

full OT 在同一 candidate support 与联合 reference identity mass 上运行，继续使用当前静态 OT 的 unbalanced entropic OT solver 实现。source marginal 先在 candidate support 内按 base mass 归一化为 `p_C`；reference marginal 按每个 catalog 等质量、catalog 内有效 segment 原始 mass 比例构造。`DynamicOTConfig` 是 `OTConfig` 的子类，承载 `candidate_top_k`、identity weights、confidence 和 geometry；不使用 OTConfig 中旧的候选字段控制 DynamicTransition。

OT 输出必须保留：

- source row marginal；
- normalized target mass；
- solver convergence diagnostics；
- candidate IDs；
- reference mass normalization；
- transport cost 摘要；
- 与 candidate-uniform 的 support equality hash。

OT 输出的 `q_OT` 与完整 source topology 对齐：candidate support 外和 invalid segment 的质量为 0。它不再执行 exposure cap、outside floor 或 temperature；后续只由全局 confidence mode 与 uniform prior 生成最终 target。

OT convergence 只证明数值求解完成，不证明训练有效。full OT 与 candidate-uniform 的差异必须作为独立 allocation effect 报告。

## 6. Confidence mode 与 uniform smoothing

### 6.1 Uniform prior

设 `V` 为完整 source topology 中有效的 source segment 集合，`C` 为 joint candidate support。定义只在有效 source segment 上均匀分配的先验：

~~~text
u_valid[i] = 1 / |V|, i in V
u_valid[i] = 0,       i not in V
~~~

`q_pure` 是任意 OT policy 的 pure target：d-ot 为完整 source topology 对齐的 `q_OT`，d-candidate-uniform 为 `p_C`。candidate support 外和 invalid segment 的 pure target 质量为 0；支持集外的非零质量和支持集内的平滑由同一个 uniform prior 完成。如果 `C` 覆盖全部有效 source segment，则不构造额外的支持集外质量，但仍执行全体有效 segment 上的 uniform smoothing。

### 6.2 Mutually exclusive confidence modes

配置文件中的两个字段是全局模式控制字段：

~~~text
auto_confidence: true | null
effective_confidence: float in [0, 1] | null
~~~

二者严格互斥：

- Random policy：两个字段都必须为 `null`，不执行 uniform-confidence smoothing；
- 任意 OT policy：必须且只能有一个字段非空；
- `auto_confidence=true` 时，`effective_confidence` 必须为 `null`，由当前 cache build 的 OT 数值诊断推断最终值；
- 手动模式下，`auto_confidence` 必须为 `null`，`effective_confidence` 必须填写 `[0, 1]` 内的数值；`1.0` 表示直接使用 pure target；
- 两者同时非空、OT 两者同时为空、Random 任一非空，均为配置错误并直接失败。

最终质量统一为：

~~~text
q_final = effective_confidence * q_pure
          + (1 - effective_confidence) * u_valid
~~~

当 `effective_confidence < 1.0` 时，OT policy 按比例保留全局 uniform prior。`effective_confidence=1.0` 时不保留额外 uniform prior，直接使用 pure target。对于任意有效 candidate-support 外 segment，若 `C` 未覆盖全部有效 source segment，则：

~~~text
q_final[i] = (1 - effective_confidence) / |V|
~~~

这同时保证支持集外概率非零且均匀，并压低 candidate support 内的 pure target 尖峰。该 smoothing 适用于所有 OT policy；Random policy 不接受这两个字段，也不通过该机制改变自身分布。

Auto Confidence 只在 `auto_confidence=true` 时启用，并由当前 cache build 的 future-blind OT 数值诊断推断：

1. 读取当前 candidate/OT 求解产生的数值诊断；
2. 按冻结的 Auto Confidence 规则推断一个 `[0, 0.95]` 内的值；
3. candidate/support/solver/归一化的 correctness 检查失败时直接失败，不将数值错误转换为 confidence；
4. 手动模式不读取 Auto Confidence 规则，也不使用 OT 数值诊断覆盖用户填写值。

首版不把 ESS、entropy、top-mass、训练结果或 checkpoint 表现作为额外加权项。它们仍可记录为 diagnostics，供后续证据决定是否解冻。

### 6.3 Auto Confidence 的数值边界

Auto Confidence 的输入只能来自当前 cache build 已产生的 candidate/OT 数值诊断；不得读取 held-out evaluation、训练 loss、checkpoint 或其它训练 outcome。具体诊断字段、组合规则和阈值必须在生产实现前冻结并写入测试；不能由手动 `effective_confidence` 反向推断，也不能在手动模式下运行。

`rho_C` 继续作为 candidate coverage diagnostic 记录，但不再通过一次性 static k256 中位数单独决定 confidence。

## 7. 当前代码边界与拟议接口

### 7.1 复用的现有接口

- SourceBundleReader / ReferenceBundleReader：保留 Catalog、Bundle 和 segment geometry 验证；
- `dataset_processing/shard_dataset.py` 中的 DynamicShardDatasetReader：紧邻并继承现有 ShardDatasetReader，只服务 Feature 阶段的 Dynamic transition raw half 读取；
- CompactRuntimeDatasetComputationPart：继续承载 segment geometry 和 raw mass；
- OTConfig：保留当前静态 OT solver 的默认实现和数值语义；
- DynamicOTConfig：OTConfig 子类，承载 Dynamic 专有 candidate、confidence mode、geometry 和 identity weights；不包含 temperature、floor、exposure cap、lambda 或旧 Dynamic 字段；
- DynamicTransitionFeatureSpace：承载完整 source topology 以及仅包含 valid segment 的 transition identity/mass arrays；
- SamplingTransferResultCache：扩展 manifest 和 diagnostics，不替换整体 cache protocol；
- sampling.safetensors：继续保存 segment geometry、raw mass 和 target mass；
- SegmentWeightedWindowSampler：不修改训练期 hierarchical sampling 语义。

### 7.2 新接口的责任边界

以下符号名称已经冻结，生产实现不得自行改名：

| 拟议接口 | 责任 |
|---|---|
| DynamicShardDatasetReader | `dataset_processing/shard_dataset.py` 中现有 ShardDatasetReader 的子类，供 Feature 阶段读取 Dynamic raw half |
| DynamicTransitionSegment | 一个 segment 的稳定 ID、A/B geometry、validity、raw mass 和临时 identity |
| DynamicTransitionIdentity | 固定 51D 数值布局、版本、校验和 |
| DynamicTransitionFeatureBuilder | 从 raw A/B 生成 51D identity |
| DynamicTransitionFeatureSpace | 完整 source topology、valid segment IDs、51D identity、base mass 和 geometry 的不可变计算空间 |
| DynamicTransitionDistanceCalibration | source-marginal MAD 尺度校准；零 MAD fallback 为 1.0；固定 Huber delta=1 |
| DynamicTransitionCandidateSupport | candidate IDs、coverage 和 support diagnostics |
| DynamicTransitionCandidatePlanner | exact Top-K 与 joint support union |
| DynamicTransitionPlan | 完整 source topology 对齐的 base mass、pure target、final target、support 和 diagnostics |
| DynamicTransitionConfidence | Auto/Manual mode、effective confidence 和选择原因 |
| References2SourceOTPlannerBase | 与当前静态 OT 共享的 OT planner 基类，负责公共输入校验、space/support/质量流程和 plan 约束 |
| References2SourceSegmentOTPlanner | 当前静态 OT planner；`References2SourceOTPlannerBase` 的同级子类 |
| CandidateUniformSegmentPlanner | 当前静态 matched-support control；`References2SourceOTPlannerBase` 的同级子类 |
| References2SourceDynamicTransitionOTPlanner | 新 Dynamic full OT planner；与静态 OT planner 同级继承 `References2SourceOTPlannerBase` |
| References2SourceDynamicTransitionCandidateUniformPlanner | 新 Dynamic matched-support control；与静态 OT planner 同级继承 `References2SourceOTPlannerBase` |
| DynamicOTConfig | `OTConfig` 子类；Dynamic 参数和 Dynamic-specific validation |

Planner 继承关系固定为：

~~~text
Policy
└── References2SourceOTPlannerBase
    ├── References2SourceSegmentOTPlanner
    ├── CandidateUniformSegmentPlanner
    ├── References2SourceDynamicTransitionOTPlanner
    └── References2SourceDynamicTransitionCandidateUniformPlanner
~~~

默认 OT 数值实现、candidate distance 计算和 solver 调用与当前静态 OT 一致；Dynamic 只替换输入 identity、candidate support 和 target 后处理公式。Dynamic 不再另建一套与静态 OT 平行的 solver 基类。

公共 planner base 只冻结 OT 方法族的输入校验、source/reference mass 处理、candidate support 约束和 solver 调用边界；返回值允许保持当前静态 OT 的 `AnalysisSegmentMassResult`，或 Dynamic 的 `DynamicTransitionPlan`。两者都必须提供完整 source topology 对齐的最终 segment mass，且都不再接受 `lambda` 参数。

### 7.2.1 核心计算对象字段

`DynamicTransitionFeatureSpace` 必须同时保存完整 topology 和有效分析子集，但只有有效子集进入 identity/candidate/OT 数组：

~~~text
DynamicTransitionFeatureSpace
  role: source | reference
  reference_id: str | None
  all_segment_ids: stable ordered IDs
  all_segment_starts: int64[N]
  all_segment_lengths: int64[N]
  all_base_mass: float64[N]          # invalid row=0, total=1
  valid_indices: int64[M]
  valid_segment_ids: stable IDs[M]
  identities: float32[M, 51]
  valid_base_mass: float64[M]        # normalized over valid rows
~~~

`DynamicTransitionPlan` 只保存算法结果，不负责文件 IO：

~~~text
DynamicTransitionPlan
  source_segment_ids: stable ordered IDs[N]
  base_mass: float64[N]
  pure_target: float64[N]          # q_OT before uniform-confidence smoothing
  final_target: float64[N]
  candidate_support: DynamicTransitionCandidateSupport
  effective_confidence: float
  diagnostics: mapping
~~~

其中 `pure_target` 是 uniform-confidence smoothing 之前的 target：d-ot 为完整 source topology 对齐的 `q_OT`，d-candidate-uniform 为 `p_C`；`final_target` 是实际写入 cache 的最终质量。两者都与完整 source topology 对齐，invalid segment 行始终为 0。所有 OT policy 的 `final_target` 按 `q_final = effective_confidence * q_pure + (1 - effective_confidence) * u_valid` 生成。

`DynamicTransitionDistanceCalibration.fit(source_space)` 只读取 source 有效 identity 并生成每个 identity 分量的 MAD scale；`compute_distance(source_identity, reference_identity)` 只读取两个 51D identity 和已拟合 scales。它不接收 reference outcome、训练配置或 checkpoint 信息。

`References2SourceOTPlannerBase.plan(source_space, references, config)` 负责公共校验、构造 calibration、候选 support、reference mass、confidence mode 校验和完整 topology 对齐；具体子类只实现 `_build_pure_target`。所有 OT planner 在其后解析 Auto/Manual confidence mode、构造 `u_valid` 并执行 uniform-confidence smoothing；candidate-uniform planner 的 `p_C` 仍作为其 pure target。

模块间最小函数接口也已经冻结：

~~~text
References2SourceOTPlannerBase.plan
References2SourceSegmentOTPlanner.plan
CandidateUniformSegmentPlanner.plan
References2SourceDynamicTransitionOTPlanner.plan
References2SourceDynamicTransitionCandidateUniformPlanner.plan
DynamicShardDatasetReader.read_segment_halves_into
DynamicTransitionFeatureBuilder.build_identity
DynamicTransitionDistanceCalibration.fit
DynamicTransitionDistanceCalibration.compute_distance
DynamicTransitionCandidatePlanner.retrieve_exact_top_k
DynamicTransitionCandidatePlanner.build_joint_support
References2SourceDynamicTransitionOTPlanner._build_pure_target
References2SourceDynamicTransitionCandidateUniformPlanner._build_pure_target
build_uniform_source_prior
apply_effective_confidence_smoothing
select_effective_confidence
~~~

DynamicOTConfig 的用户可控字段就是第 1.5 节集中配置入口列出的字段；不再保留旧 DynamicConfig 的 `state_support_radius`、`state_bandwidth`、`transition_bandwidth` 或 `segment_horizon`。

旧 ts3/core/transfer/dynamic.py 的 pair-specific implementation 直接由新 transition implementation 替换；旧 Dynamic symbols 不保留兼容 shim，因为旧实现已经在其他 branch 有 commit 存档。

### 7.3 建议的实现 DAG 模块

~~~text
ts3/core/transfer/
  dynamic_transition.py       # identity, builder, segment records
  dynamic_transition_cost.py  # source MAD calibration and exact distance
  dynamic_transition_support.py
  dynamic_transition_confidence.py
  dynamic_transition_planner.py
  solver.py                   # existing OT solver boundary and orchestration
  cache_builder.py             # source/reference raw read and artifact build
  cache.py                     # artifact IO and manifest extension

dataset_processing/shard_dataset.py
  DynamicShardDatasetReader    # ShardDatasetReader subclass
~~~

`DynamicShardDatasetReader` 的模块位置已经固定为 `dataset_processing/shard_dataset.py`；其余 `ts3/core/transfer/` 文件按上述职责拆分实现，不得把 Feature 阶段 raw reader 逻辑塞入 `cache_builder.py`。

DynamicShardDatasetReader 的新增方法固定为：

~~~text
read_segment_halves_into(
    trajectory_id,
    segment_start,
    segment_length,
    before_output,
    after_output,
)
~~~

方法要求 segment_length 为正整数；标准 segment 使用两个等长的 float64 输出数组，特殊短 reference segment 允许奇数长度，此时 before 长度为 `ceil(segment_length/2)`、after 长度为 `floor(segment_length/2)`。它只负责 Feature 阶段 raw 读取；训练期仍由现有 WindowAccessor.read_window_into 交付完整 context 和 prediction。

## 8. Cache 与生命周期

### 8.1 Compute-only state

cache build 期间允许存在：

- source/reference Feature Bundle v2 readers；
- 51D source/reference identity matrices；
- source-marginal MAD calibration；
- candidate index；
- joint reference mass；
- OT coupling 或 row marginal。

identity storage 使用 float32；组件 reduction、distance summary、mass normalization 和最终 target 使用 float64。full dense source-reference distance matrix 不得分配。

raw shard reader 和 raw A/B blocks 只在 Feature 阶段短暂存在；cache build 不重新读取 raw A/B。所有 compute-only state 在写出 cache 后释放。

这些对象在写出 artifact 后必须释放。Reference raw values、reference identity 和 source identity 不得被训练 worker 反序列化。

### 8.2 Persistent artifact

首版 persistent cache 只需要保留现有 sampling payload 加上 manifest diagnostics：

- cache schema `ts3.transfer-sampling-cache.v3`；所有旧版本 cache 直接拒绝加载并重建；
- source catalog hash；
- reference catalog hash；
- policy ID d-ot 或 d-candidate-uniform；
- segment geometry；
- transition identity schema/version 来源摘要；
- active segment spec；
- candidate top-k；
- candidate support hash；
- rho_C 和实际 effective confidence；
- target kind：candidate-uniform 或 full OT；
- confidence mode：auto 或 manual；
- effective_confidence、uniform prior cardinality 和选择原因；
- base mass、final mass summaries；`pure_target` 只在 `DynamicTransitionPlan` 内存对象中保留，不把完整 pure target 写入 persistent cache；
- determinism and normalization audit。

不持久化：

- raw A/B values；
- cache build 内存中的 51D identity matrix（identity 结果本身已在 Feature Bundle v2 持久化）；
- full pairwise distance matrix；
- OT coupling；
- reference training data；
- held-out outcome。

### 8.3 训练期不变量

- 当前 sampler 只看到最终 segment mass；
- segment ID 必须与当前 source topology 一致；
- target_masses 必须有限、非负且归一化；
- cache schema 不是 v3、policy 不匹配或 source topology 不匹配时拒绝加载；不执行全量配置 hash 比较；
- training future 只进入正常 loss，不进入 cache build 或 confidence。
- training 期不读取纯 OT target、不重新判断 confidence、不执行 lambda annealing；只使用最终 target mass。

## 9. 测试与验收

### 9.1 Identity tests

- 512 长 segment 精确切成 A/B 两个 256 长 block；
- 标准 source 尾段通过末尾回溯产生完整 segment；特殊短 reference 按实际长度切分，奇数多出的点归前半段；
- 任一半小于 8 点时 segment invalid；
- A、B descriptor 各为 16D，delta 为逐维差；
- 最终 identity 恰好为 51D，顺序和 dtype 固定；
- Feature Bundle v2 直接保存命名的 before/after/change/auxiliary 字段，并保留旧 full-segment feature_values；
- constant、near-constant、NaN、infinity 和短 block 按 feature protocol 明确失败或返回诊断；
- level_shift、scale_change、is_near_constant 具有手工可验证结果；
- 同一输入重复构造 identity 完全一致。

### 9.2 Matching tests

- raw sequence 不会进入 cost；
- A、B、change、auxiliary 四类信息任一变化都能改变对应 distance；
- source-marginal MAD calibration 不读取 reference outcome，zero-MAD fallback 为1.0；
- Huber delta 固定为1，六组 distance 先逐分量计算再组内均值；constant mismatch 为0/1绝对差；
- exact Top-256 与小样本 dense reference 实现一致；
- GPU blocked 与 CPU exact fallback 的 candidate ID 一致；
- tie-break 稳定；
- full OT 和 candidate-uniform 的 candidate support hash 完全一致。

### 9.3 Mass tests

- candidate-uniform、full OT、uniform prior 和 final smoothing 各自归一化；
- `u_valid` 只在有效 source segment 上均匀分配，invalid segment mass 始终为 0；
- Random policy 的 `auto_confidence` 与 `effective_confidence` 都为空；任意 OT policy 恰有一个非空；
- 手动模式的 `effective_confidence` 在 `[0, 1]` 内；Auto 模式的推断结果在 `[0, 0.95]` 内；
- 手动 `effective_confidence` 不触发 Auto Confidence，Auto 模式不接受手动值；
- candidate support 外的有效 segment 获得相同的 `(1 - effective_confidence) / |V|` 质量；
- C 覆盖全部有效 source 时不制造额外的支持集外质量；
- invalid source segment mass 为0，有效 segment 按原始 base mass 比例重归一化；
- invalid reference segment 在 catalog 内按原始 mass 比例重归一化，空 reference catalog 直接失败；
- Auto Confidence 按冻结的当前 OT 数值规则输出 `[0, 0.95]` 内的值；
- `q_final = effective_confidence * q_pure + (1 - effective_confidence) * u_valid` 与手工计算一致；
- 求解失败、空 reference 或无支持时直接失败，不回退 base Random；
- 所有 sampling method 不再执行 lambda mixing 或 annealing。

### 9.4 Integration tests

- d-ot 配置可以从 cache builder 走通；
- d-candidate-uniform 配置可以从 cache builder 走通，并对其 `p_C` pure target 执行相同的 confidence mode 规则；
- DynamicShardDatasetReader 与基类 reader 的读取结果一致；
- Feature Bundle v2 可以驱动 d-ot 与 d-candidate-uniform；静态方法可继续读取旧 Feature Bundle v1；
- `DynamicOTConfig` 单一配置入口同时提供 OT solver 与 Dynamic 参数；
- `References2SourceOTPlannerBase` 的静态 OT、静态 candidate-uniform、Dynamic OT 和 Dynamic candidate-uniform 子类共享公共 support/质量约束；
- 旧 DynamicConfig 不会悄悄改变新 transition 语义；
- v3 SamplingTransferResultCache 可以加载新 artifact，旧版本 cache 被拒绝；
- sampler 从 segment mass 生成合法 window；
- planned/delivered segment mass audit 可以运行；
- cache reload、不同 block size 和 CPU/GPU fallback 具有同一 topology 和 mass hash；
- Feature 阶段 Load-invalid trajectory 不进入任何方法，Feature-invalid segment 在所有方法中一致为零质量。

### 9.5 研究验收

首版代码验收只证明：

- transition identity、candidate support、OT target、uniform-confidence smoothing 的数据流正确；
- candidate-uniform 与 full OT 是 matched-support；
- artifact 可复现、可加载、可交付。

它不因为 solver convergence、descriptor distance 下降或 cache 构建成功，就声称 forecast utility。20-task 结果仍需作为独立实验协议审计。

## 10. 已决定、待执行的准备工作

当前设计决策已冻结；进入生产代码前只需完成以下已确定工作：

1. 冻结 Auto Confidence 使用的当前 candidate/OT 数值诊断、组合规则和阈值，并写入对应测试；
2. 将 `auto_confidence` 与 `effective_confidence` 写入共享配置入口并实现互斥校验；
3. 将共享 Dynamic/OT 参数集中到 run template；方法条目只选择 policy，不重复覆盖 transition 参数；
4. 删除所有 sampling method 的 lambda 配置和训练期 annealing 接线；
5. 不创建额外 calibration artifact，不增加 calibration version，不把训练 outcome 作为 confidence 输入；
6. 生产代码实现必须沿用本节之前确定的 `References2SourceOTPlannerBase` 同级 planner 层级，不把 Dynamic 再拆成独立的平行 solver 体系。
