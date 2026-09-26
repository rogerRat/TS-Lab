# TS3 决策记录

> 历史归档：保留当时的设计、操作或实验证据，不作为当前执行指令。当前入口见 `docs/README.md`。

## 2026-07-31 — Heartbeat 仅在日志静默时输出固定存活摘要

- Status: accepted
- Problem: 全局 `_Heartbeat` 保存最近一条普通日志的完整正文；`ProcessingProgress` 的进度条因此会被再次嵌入全局 heartbeat，造成进度条和 ETA 噪声重复。两个 30 秒周期也会在同一长任务中产生重复心跳。
- Decision: 全局 heartbeat 将间隔解释为日志静默阈值，只有超过阈值且期间没有业务或进度活动时才输出；输出固定的配置 stage、`idle_for` 和 `uptime`，不再复制任意上一条日志。进度日志通过 `ts3_event=progress` 与 `ts3_progress_event` 标记事件类型；`ProcessingProgress` 对未变化的 `(completed, total, detail)` 快照跳过周期 heartbeat，但保留 start、checkpoint 和终态记录。
- Consequences: 有正常日志或变化中的进度时不额外产生全局 heartbeat；真正静默时仍能看到简短的存活记录。日志格式和 telemetry 线程行为改变，但不改变训练、采样、下载、评估或数据处理语义。

## 2026-07-31 — TS3 两层直接抽样索引与轨迹 sampler 常驻缓存

- Status: accepted
- Problem: `RuntimeDataset` 已在内存保留 Grid、Trajectory、length 与 locator，但原 TS3 路径在每个 slot 选中 grid 后重新为其全部 trajectory length 构建累计表和 `TrajectorySampler`。对同一 grid 的重复构建会丢弃，在 TS3 的多 grid 抽样中形成 CPU 供给空档；Random 只有一个 global grid 并已长期复用其 sampler。
- Decision: `GridSampler` 与长度加权 `TrajectorySampler` 各自持有私有 `_direct_sample_index` 累计抽签索引，以随机数和 `searchsorted` 直接完成加权抽样。`TS3Core` 在初始化时构建并保留 `_trajectory_samplers_by_grid_id: dict[int, TrajectorySampler]`；其键为已有 grid id，值为该 grid 的可直接抽样索引。
- Consequences: 原始数值不会读入该缓存，不修改 COI、数据结构、窗口 IO 或 checkpoint 格式；缓存由已校验 RuntimeDataset 完全派生，无需写入 checkpoint。各级的概率分布保持不变，但 Grid 改为使用直接累计索引而非 `rng.choice(..., p=...)`，固定 seed 的具体 TS3 抽样 trace 可能不同。

## 2026-07-31 — Random 批量抽样复用全局长度加权 sampler

- Status: accepted
- Decision: Batch 独立 RNG 路径对 RandomCore 继续复用其启动时创建的全局 `TrajectorySampler`，不为每个 slot 重建全轨迹长度累积数组。
- Consequences: 每个 slot 仍使用独立 RNG 子流并保持相同的长度加权概率；移除每 batch 对全部 Random 轨迹的重复 O(N) 累积构造。

## 2026-07-31 — Chronos 预训练权重固定为项目本地缓存

- Status: accepted
- Decision: Profile-1B 的 `google/t5-efficient-tiny` 固定为 revision `3441d7e8bf3f89841f366d39452b95200416e4a9`，物化于项目 `cache/pretrained_models/google/t5-efficient-tiny/`。Adapter 只用 `local_files_only=True` 从该项目缓存加载，绝不联网下载或回退到 Hugging Face cache。
- Consequences: 训练启动不依赖 Hugging Face 网络或可变的 `main` 引用；`cache/` 不进入 Git。若两级本地缓存都不存在，训练立即失败并给出加载错误。

## 2026-07-31 — 首两个训练 batch 输出分段性能计时

- Status: accepted
- Decision: 训练 loop 与 Chronos adapter 仅对前两个成功 batch 记录 Core 调度、批量请求、tensor 组装、优化和接受边界耗时。
- Consequences: 可定位 CPU/I/O 与 GPU 计算停顿，不改变抽样、窗口、模型或审计语义，也不会在正式长训练中产生逐步日志开销。

## 2026-07-31 — BatchWindowContent 与独立 slot RNG 的批量训练交付

- Status: accepted
- Decision: 新增并集中定义 `BatchWindowContent` 于 `ts3.spec.models.training_protocol`，训练主路径一次交付按 ticket 顺序排列的连续 float64 `[batch_size, window_length]` 数组。每个 slot 使用由 `(seed, slot_number)` 派生的独立 RNG 子流，Core 并行完成 batch locator 抽样，并依据 locator 直接读取窗口区间，不实例化完整 trajectory 或单条 `WindowContent`。原 `WindowContent`/`request_window` 保留为兼容入口。
- Consequences: 固定 seed 不再与旧的单 RNG 顺序实现产生相同 trace，但 slot 概率分布、batch 行顺序、模型目标与 optimizer 边界保持不变；新 checkpoint 仅与该 batch-RNG 协议兼容。审计由 `audit.flush_interval_steps` 控制常规 flush，checkpoint、失败和结束边界强制 flush。

## 2026-07-31 — Core 请求窗口时物化 WindowContent

- Status: accepted
- Decision: Core 保留已选 `Window` 用于内部审计与交付确认，并通过现有 `WindowAccessor.read()` 读取数值后返回 `WindowContent` 给训练 Adapter。
- Consequences: Core-to-adapter 窗口交付完成从 locator 到实际数值的协议接线，不改变窗口抽样或数据读取实现。

## 2026-07-31 — WindowAccessor 按 Bundle 复用 ShardDatasetReader

- Status: accepted
- Decision: WindowAccessor 为每个唯一 Bundle 目录只创建一个 ShardDatasetReader；遍历同一 Bundle 下的多条 trajectory 时复用已有 reader，不重复解析 load/index.arrow。
- Consequences: 训练启动阶段的索引读取从“每条 trajectory 一次”降为“每个 Bundle 一次”，不改变窗口采样、轨迹选择或训练数值语义。

## 2026-07-31 — 补齐 Reference Fetch 与 Catalog 主流程日志

- Status: accepted
- Decision: Reference Fetch、Source Catalog、Reference Catalog 入口统一配置项目 logger；Catalog 构建按记录数输出阶段进度、dropped 统计、最终记录数和输出路径。
- Consequences: 数据链在 Fetch/ref 和 Catalog 阶段也能通过控制台及 `logs/data/{fetch/ref,catalog,catalog/ref}/` 观察，不改变数据处理与 Catalog 内容。

## 2026-07-31 — 训练窗口资格由 Catalog 状态与长度确定

- Status: accepted
- Decision: 训练仅接纳未 `dropped` 且长度不少于 `context_length + prediction_length` 的 Source 轨迹。窗口数量按长度与 stride 公式计算，窗口起点由抽样 rank 直接换算；训练不再扫描轨迹数值或枚举窗口检查有效值。
- Consequences: Load/Feature 的 `dropped` 契约成为训练数值有效性的唯一依据；启动和逐样本路径不再重复读取或扫描原始数值。

## 2026-07-31 — Random 与 TS3 RuntimeDataset 构建输出分策略进度日志

- Status: accepted
- Decision: Random 记录 Source 轨迹扫描、窗口资格、dropped 数量、全局 Grid 人口和完成进度；TS3 分别记录 Source/Reference 扫描、特征无效、网格分组、混合前后网格集合、无对应 Reference 网格及舍弃质量。
- Consequences: 训练启动阶段可区分两种策略的实际处理阶段与统计，不改变 Catalog、窗口资格、网格混合或采样语义。

## 2026-07-31 — 训练配置以 optimizer steps 指定预算

- Status: accepted
- Decision: 顶层训练配置使用 `training_steps`；内部 `effective_slot_budget` 由 `training_steps * batch_size` 自动推导，不再要求配置填写 slots 数。
- Consequences: 配置表达直接对应 optimizer 更新次数，Core 和 adapter 仍沿用内部 slot budget 协议。

## 2026-07-31 — 训练与评估统一记录详细阶段日志和近似进度

- Status: accepted
- Decision: 训练日志写入 `logs/train/`，评估日志写入 `logs/evaluate/`；训练按 optimizer step、FEV/TIME 按任务输出带百分比和 ASCII 进度条的 INFO 日志，同时输出到控制台。
- Consequences: 长时间启动、训练和评估阶段均可通过控制台或对应日志文件观察当前阶段、进度、复用任务、结果路径和异常位置；不引入 tqdm 或改变算法语义。

## 2026-07-31 — SOP-2 不规定训练进程管理方式

- Status: accepted
- Decision: SOP-2 仅保留 `python -m train.train` 的训练入口及其参数说明，不再规定 tmux 会话、日志重定向、远端目录切换或 Conda 激活前缀。
- Consequences: 训练进程管理交由运行环境或操作者决定，SOP-2 只表达训练配置与项目 CLI 的实验语义。

## 2026-07-31 — Profile-1B 训练配置绑定现有 Source Catalog

- Status: accepted
- Decision: `configs/train/chronos/ts3-profile-1b-fev.yaml` 和 `random-profile-1b.yaml` 均使用已发布的 `configs/catalog/src/1b.yaml`；TS3 保留 FEV Reference Catalog、`n_bins` 和 `lambda`，Random 不引入这些 TS3 专用字段。
- Consequences: 两个训练入口均可解析到当前 Profile-1B Source Bundle；训练配置不再引用不存在的 `source.yaml`。

## 2026-07-30 — Dedup 与 Features CLI 输出阶段日志

- Status: accepted
- Decision: Dedup 和 Features CLI 统一配置项目 logger，向 stderr 及 `logs/data/{dedup,features}/` 输出启动参数、Bundle 校验、每 1000 条轨迹的进度、结果统计、输出路径和完成状态；不改变数据处理逻辑或产物 schema。
- Consequences: CLI 不再静默运行，运维可通过控制台和时间戳日志定位长时间处理阶段；已有输出仍按严格不覆盖规则处理。

## 2026-07-30 — Catalog V6 显式表达 dropped 状态

- Status: accepted
- Decision: Catalog V6 使用 `state is not None` 表达是否存在已丢弃状态，不使用状态字典的隐式布尔值。
- Consequences: dropped 状态的业务判断与状态对象内容解耦，保持 Load、Dedup、Feature 阶段优先级不变。

## 2026-07-31 — SOP-1.1 明示 Load/ref 的内嵌安全提取

- Status: accepted
- Decision: 在 Reference Load/ref 命令旁明确说明其会在进程内安全提取正式 test 起点前的历史、立即转换为 Bundle，并删除临时原生数据。
- Consequences: 操作人员不会误以为仍须执行独立 Extract/ref，或误以为 Load/ref 直接把未经安全边界处理的 Fetch raw 写入 bins。

## 2026-07-30 — SOP-1 与 SOP-1.1 同步到 Bundle 数据链

- Status: accepted
- Decision: Source SOP 与 Reference SOP 只描述 Fetch as-is、单一 ShardDatasetBundle、Arrow-only Dedup/Feature 和 Catalog V6；Reference Extract 明确为 Load 内部临时步骤。
- Consequences: 操作人员不再得到 RuntimeDataset、build-id、独立 Extract 或 Catalog V4/V5 命令，Reference 训练读取边界也在 SOP 中明确为 Feature-only。

## 2026-07-30 — 移出旧持久化数据树

- Status: accepted
- Decision: 将已废弃的 `data/src/runtime`、`data/ref/runtime` 和 `data/ref/extracted` 从项目数据树移至远端用户回收目录；Fetch raw 数据未改动。
- Consequences: 项目内不存在旧 RuntimeDataset 或独立 Extract 产物，运行只能从新的 Bundle 路径开始；回收目录保留仅用于平台级恢复，绝不被代码读取。

## 2026-07-30 — 删除旧 RuntimeDataset 与独立 Reference Extract 入口

- Status: accepted
- Decision: 删除旧 `RuntimeDataset` 文件契约、发布器、Dedup/Feature 实现及其独立 Reference Extract CLI；Reference Extract 仅作为 `load.ref` 的内部安全解析步骤，临时原生表示在 Load 返回前删除。
- Consequences: 不提供旧路径、旧 build-id、旧 Catalog V5 或旧 stage tree 的读取兼容；维护者只能通过 `ShardDatasetBundle` 的 bins+Arrow 契约进入后续处理。

## 2026-07-30 — RuntimeDataset 全量替换为 ShardDatasetBundle

- Status: accepted
- Decision: 废弃所有外部 RuntimeDataset 目录、身份、父链与兼容读取。一个 ShardDatasetBundle 固定由 Load 持有的二进制 shards 与 Arrow 索引组成，并可附加不复制数值数据的 Dedup 和 Feature Arrow 阶段；scope 仅保存在 `load/meta.arrow`。默认单 shard 上限为 256 MiB，轨迹不可跨 shard。Reference Fetch 保留原始文件，Extract 成为 Load/ref 的内部安全读取步骤；Reference Dedup 入口暂时显式 `NotImplementedError`。
- Consequences: Catalog、训练与数据处理必须迁移到 `ts3.spec.files.shard_dataset` 契约；旧 RuntimeDataset、Catalog 和已发布派生产物均不兼容，须从 Fetch 原始数据重建。

## 2026-07-30 — Reference Fetch 支持显式 benchmark 输出目录

- Status: accepted
- Context: 运行时需要将同一冻结 Reference spec 的 benchmark 数据写入一个明确指定的目录，而不是只能使用 spec 中的默认路径。
- Decision: `dataset_processing.fetch.ref` 的 `fetch(spec_path, output_dir=None)` 与 CLI 都接受可选 `--output-dir`；传入时作为 adapter 的数据根，未传入时仍解析并使用 `upstream.data_root`。
- Consequences: 既有无 `--output-dir` 调用的输出位置与行为不变；spec 中的 benchmark、revision、declaration 和 task selection 仍是唯一的实验语义来源。

## 2026-07-30 — Source Features 自动批处理全部 Dedup scope，工件仍按 scope 发布

- Status: accepted
- Decision: `dataset_processing.features` 改为只接受 `--input-root`，自动发现 `dedup/` 下同一 build ID 的全部 scope 并依次执行既有逐轨迹特征计算。
- Consequences: 用户不再输入每个 Dedup 目录；Feature 数值、父链、RuntimeDataset ID、输出目录与 Catalog 记录仍保持 dataset/subset scope 隔离。`--skip-existing` 仅供中断后的同 build 恢复，默认严格拒绝已有输出。旧单 `--input` 模式不再接受，须由新的批处理入口重建全部 Feature 和 Catalog。

## 2026-07-30 — Source Dedup 的重复判定范围覆盖整个 Load RuntimeDataset

- Status: accepted
- Decision: `dataset_processing.dedup` 改为只接受 `--input-root`，一次发现其 `load/` 下的全部 Load RuntimeDataset，并在全部轨迹间建立同一个量化 SHA-256 重复分组。输出仍按原 dataset/subset scope 写入，保证 Features 的直接父工件布局不变。
- Consequences: 跨 dataset/subset 的重复轨迹现在只保留全局字典序最小 `(dataset, subset, series_id)` winner；其他 scope 的 `dedup.arrow` 将该 winner 记录为外部 scope。旧的单 `--input` scope 模式不再接受，所有 Source 派生工件必须从 Load 后重新执行全局 Dedup、Features 和 Catalog。

## 2026-07-30 — SOP 的训练默认实例同时覆盖 Random 与 TS3

- Status: accepted
- Decision: 新增 `configs/train/chronos/random-profile-1b.yaml` 作为 Profile-1B Source Catalog 的 Random 可运行基线；SOP-2 同时以此配置和既有 `configs/train/chronos/ts3-fev.yaml` 为唯一默认训练命令。TS3 固定消费 Profile-1B Source Catalog 与 FEV 20 个任务安全历史生成的 Reference Catalog。
- Consequences: SOP 不再要求用户在命令中替换抽象路径或手写训练 YAML。Random 配置严格只含 `sampling.policy=random` 和 `source_catalog`；TS3 配置才含 `n_bins`、`lambda` 和 `reference_catalog`，不改变两种策略的实现逻辑。

## 2026-07-30 — Random 采用 V5 GlobalGrid 与状态恢复身份

- Status: accepted
- Decision: Random 与 TS3 继承共同 `Core`，但作为并列策略实现。Random 使用 V5 Source Catalog 的 `data_*` locator（通常为 Dedup），不读取或校验 Feature 值/有效性，不构建 FeatureGrid、Reference 或 Grid 概率分布。所有存在合法窗口的轨迹进入唯一 `GlobalGrid`，并在其中按轨迹长度加权抽样；窗口仍按现有协议在轨迹内均匀抽取。
- Consequences: Random 与 TS3 共用 ticket、窗口、审计、批次事务和 checkpoint 流程。V5 恢复兼容性使用 Catalog location、Dataset `resume_state`、规范化配置及 TS3 的显式概率映射；不重新引入 SHA-256 或内容 fingerprint 校验。

## 2026-07-30 — 仅 Dedup 进程内允许 SHA-256，所有派生产物采用无 Hash schema

- Status: accepted
- Decision: RuntimeDataset v2、Catalog v5、reference manifest v2、训练 checkpoint 和评测 metadata 禁止任何 SHA/checksum/fingerprint 字段，也不重读文件内容计算摘要。唯一例外是 Dedup 对量化轨迹在进程内计算完整 SHA-256 作为去重键；该键和截断值均不得写入产物。
- Consequences: 旧 Hash schema 被明确拒绝；需删除并从原始输入重建所有派生数据、Catalog、训练和评测产物。第三方代码一律不修改。

## 2026-07-30 — 训练内 Catalog 严格校验只执行一次并输出工件进度

- Status: accepted
- Decision: 同一 `CatalogReader` 实例成功校验同一个 Catalog 后，后续 `records()` 复用该成功结果，不重复重算 RuntimeDataset 内容 SHA-256；训练入口另配置全局 `catalog` Logger，使每个 RuntimeDataset 和文件哈希进度输出到控制台及 `logs/train/`。
- Consequences: 每次训练启动仍在首次读取时完整校验 Catalog 与所有声明文件，但不会在 RuntimeDatasetBuilder 阶段重复全量 I/O；已经启动的进程需重启才会获得此行为和日志。

## 2026-07-30 — 训练入口使用全局 Logger 记录启动与循环阶段

- Status: accepted
- Decision: `train.train` 启动时配置项目全局 `train` Logger，日志写入 `logs/train/` 并输出到控制台；runner 记录 Catalog 读取、运行时数据集构建、混合分布、manifest、adapter/Core 初始化和训练循环的阶段性状态，异常记录完整 traceback。
- Consequences: Catalog 校验或模型初始化等长时间阶段不再无输出；已经启动的旧进程不会动态获得新日志，必须在停止后以新代码重启。

## 2026-07-30 — Chronos 子目录训练配置以自身位置解析 Catalog

- Status: accepted
- Decision: `configs/train/chronos/ts3-fev.yaml` 位于 `chronos/` 子目录，因此 source 和 reference Catalog 相对路径使用 `../../catalog/...`。
- Consequences: 训练配置解析后固定指向 `configs/catalog/src/source.yaml` 与 `configs/catalog/ref/fev/fev.yaml`，不再误指向不存在的 `configs/train/catalog/`。

## 2026-07-30 — TS3 训练不再依赖独立 FeatureSpecification

- Status: accepted
- Decision: 训练 YAML 移除 `feature_spec`；训练启动和 `RuntimeDatasetBuilder` 不再加载、校验或记录独立的 FeatureSpecification 指纹。source/reference 仍必须具有同一 descriptor 顺序与 `n_bins`，并由 Catalog 固定其实际 Feature RuntimeDataset 工件。
- Consequences: 删除 `configs/train/feature_spec.json`；训练输入缩减为 Catalog、窗口、采样和 adapter 配置，特征实现的可追溯性由 Catalog 绑定的 Feature RuntimeDataset 文件及其 metadata 提供。

## 2026-07-30 — 首个 FEV TS3 训练配置固定已发布工件与基线预算

- Status: accepted
- Decision: 新增 `configs/train/ts3-fev.yaml`，固定 profile-1b Source Catalog、FEV Reference Catalog、`n_bins=8`、`lambda=0.5`、`512/64` 窗口、batch size 32 与 3,200,000 个 slot 预算；新增 `configs/train/feature_spec.json`，快照当前 `ts3/features/*.py` 的内容 SHA-256、当前数值阈值及 `[0,1]` 六维 descriptor 范围。
- Consequences: 可用单一训练 YAML 启动当前已发布的数据工件；该实验是 target-domain-history-aware/adaptation，后续若改动特征实现或实验超参数，必须发布新的配置文件而非覆盖本文件。

## 2026-07-30 — Catalog/ref 只依赖 Reference Feature RuntimeDataset

- Status: accepted
- Decision: Catalog/ref 只接受 `--runtime-root` 和 `--output`。发布器从 Feature `meta.json` 的 v2 reference provenance 恢复 spec、benchmark、manifest hash 和全量 asset IDs，并拒绝缺失、重复或跨 population 的 Feature 工件；不读取 Extract/ref 目录或 manifest。
- Consequences: Catalog/ref 的唯一数据输入边界是 RuntimeDataset；既有不含 v2 population 的 Feature 工件需要按新 Load/Feature 协议重新发布。

## 2026-07-30 — Catalog/ref 默认从 Extract 输出目录读取固定 manifest

- Status: accepted
- Decision: Catalog/ref 的 `--manifest` 改为可选，默认使用 `--ex-dir` 并只读取 `<ex-dir>/manifest.json`；显式 `--manifest` 仅读取指定文件，不执行目录发现。Load/ref 与 Catalog/ref 复用同一个集中式 manifest 输入解析器。
- Consequences: ReferenceCatalog 仍严格校验 manifest 与 Feature provenance，但用户不必手动填写契约文件名。

## 2026-07-30 — Features/ref 区分缺失 Load 目录与错误目录层级

- Status: accepted
- Decision: Features/ref 对不存在的 `--load-dir` 抛出明确的 `FileNotFoundError`；目录存在但不符合 `<runtime-root>/load/reference.<benchmark>` 契约时才抛出层级错误。
- Consequences: 用户可直接区分路径拼写/层级遗漏与 RuntimeDataset 布局不匹配。

## 2026-07-30 — Load/ref 使用 `--ex-dir` 表示 Extract 输出目录

- Status: accepted
- Decision: Load/ref 将默认 manifest 输入参数从 `--ref-dir` 更名为 `--ex-dir`，明确该目录是 Extract/ref 的发布结果；`--manifest` 仍是显式文件覆盖入口。
- Consequences: CLI 不再使用含义宽泛的 reference-directory 名称；旧 `--ref-dir` 参数不再接受。

## 2026-07-30 — Features/ref 只依赖 Load 固化的 reference population

- Status: accepted
- Context: Features/ref 若重新读取 Extract/ref 的 `manifest.json` 才能确定全量 asset，则 Load 在转换时丢失了下游所需信息，破坏了 RuntimeDataset 的阶段边界。
- Decision: Reference Load provenance 升级为 v2，在每个 Load `meta.json` 中固化 `spec_id`、`benchmark`、manifest hash、当前 asset ID 与完整排序 asset ID 集合。Features/ref 仅扫描 `--load-dir` 的 Load metadata，要求其形成一个完整且一致的 reference population；不读取原生 extraction 目录、manifest 或 native assets。
- Consequences: 批量 Features/ref 只需 `--load-dir`；现有 v1 Reference Load 工件不含完整 population，必须用新协议重新 Load 后才能批量 Feature。

## 2026-07-30 — Features/ref 支持按 reference manifest 批量发布

- Status: accepted
- Decision: Features/ref 只接受 `--ref-dir` 批量模式，读取固定 `manifest.json` 的全部 asset；`--load-dir` 直接指定 `<runtime-root>/load/reference.<benchmark>`，每个 asset 在其子目录中必须恰有一个已发布 Load RuntimeDataset，Feature 输出根从该结构自动推导。
- Consequences: 对 `Load/ref --asset-id all`，用户可用一次 Features/ref 命令完成所有 asset，无需输入 Load build ID 或 Feature 输出根；缺少或多于一个预期 Load 工件立即失败，不扫描未在 manifest 声明的目录，也不提供单 asset 入口。

## 2026-07-30 — Features/ref 输入参数使用目录语义

- Status: accepted
- Decision: `dataset_processing.features.ref` 将必填 `--input` 改为 `--input-dir`；该参数仍指向单个 reference Load RuntimeDataset 目录，`--output-root` 保持发布根语义。
- Consequences: Reference Features CLI 的输入/输出目录含义更明确；旧 `--input` 参数不再接受。

## 2026-07-30 — Load/ref 使用全局 Logger 输出资产发布进度

- Status: accepted
- Decision: `dataset_processing.load.ref` 通过 `configs.runtime.configure_logger` 配置 `load.ref` logger，控制台和 `logs/data/load/ref/` 文件均记录 manifest 解析、hash、asset 选择、逐 asset 发布输入/计数/输出和完整失败 traceback。
- Consequences: Load/ref 不再静默执行；日志格式、时区、文件位置和 handler 生命周期与其他项目 CLI 保持一致。

## 2026-07-30 — Reference 输出采用固定 spec 与 manifest 契约名称

- Status: accepted
- Decision: Extract/ref 始终将规范化 extraction spec 写为 `spec.yaml`，将 extraction manifest 写为 `manifest.json`。Load/ref 的 `--manifest` 改为可选：默认必须提供 `--ref-dir`，并且只检查 `<ref-dir>/manifest.json`；显式传入 `--manifest` 时只读取该指定文件，不执行目录扫描或后备发现。
- Consequences: 用户不必手动拼接 manifest 文件名；自定义 manifest 仍是显式、可审计的覆盖。旧的 `extraction_spec.*` 和 `extraction_manifest.json` 不再是活动输出名称。

## 2026-07-30 — FEV 负 cutoff 转换为绝对安全历史边界

- Status: accepted
- Context: FEV 默认 `initial_cutoff` 可为负数，表示相对序列末尾的 Python slicing 边界；Reference manifest 的 `position` 只接受非负绝对位置，直接写入该负值会在 manifest 校验阶段失败。
- Decision: FEV adapter 对负 cutoff 使用已导出的 `past_data` 历史长度，并取所有输出 target 的最小长度作为保守的绝对 `training_boundary`/`earliest_test_boundary`；非负 cutoff 和 timestamp cutoff 保持其原有语义。
- Consequences: 不读取 future，不改变 FEV 的历史数据内容；manifest 边界与实际输出的安全历史一致，并满足非负 position schema。

## 2026-07-30 — FEV Extract 先完成 provider 数据加载

- Status: accepted
- Context: FEV `Task.freq` 等元数据属性要求先调用 `Task.load_full_dataset()`；Extract/ref 在打印任务元数据前访问 `freq`，导致第一个任务在 manifest 生成前失败。
- Decision: FEV adapter 创建 `Task` 后立即调用 `load_full_dataset()`，再访问 `freq`、读取第一个 evaluation window 和生成安全历史 asset。
- Consequences: 不改变 FEV 任务选择、边界或 reference 内容；确保 provider 的生命周期顺序符合 FEV API。

## 2026-07-30 — Catalog metadata fingerprint 统一使用 meta.json 文件 SHA-256

- Context: Feature、Dedup 和 RuntimeDataset parent pin 使用 `file_sha256(meta.json)`，而 Catalog 生成/读取曾使用规范化 metadata mapping 的 SHA-256，导致新生成的 Reference Feature Catalog 在 parent pin 校验处失败。
- Decision: Catalog 的 RuntimeDataset metadata 指纹、Feature/Dedup parent pin 校验和 CatalogReader 工件验证统一使用对应 `meta.json` 文件字节的 `file_sha256`；Catalog 字段名称和 `--skip-parent-hash` 行为保持不变。
- Consequences: 现有按文件 SHA-256 生成的 Feature/Dedup parent pin 可直接通过 Catalog 校验；依赖旧 canonical mapping SHA-256 的 Catalog metadata 字段需要重新生成。内容文件和实验数据语义不变。

## 2026-07-30 — 统一 Source、Reference 与 Benchmark 数据目录

- Status: accepted
- Decision: 所有 source 原始训练数据存放于 `data/src/raw`，source Load/Dedup/Feature RuntimeDataset 存放于 `data/src/runtime`；reference 原生资产及其 RuntimeDataset 存放于 `data/ref`；benchmark 评测使用的官方原始数据存放于 `data/bench`。新代码、配置、脚本和实验不得使用旧的 `data/raw`、`data/runtime`、`data/benchmarks`、`data/reference` 或 `data/runtime-reference` 活动路径。
- Consequences: `.gitignore` 保留四个根目录及 `data/src/raw`、`data/src/runtime` 的结构占位，但忽略其中全部数据内容；现有远端旧目录不自动搬迁，后续运行必须使用新目录。

## 2026-07-30 — Reference extraction CLI 增加详细运行时日志

- Context: `dataset_processing.extract.ref` 原先只在异常时依赖命令行 traceback，长时间 FEV 提取期间无法观察当前 spec、任务、资产写入和最终 manifest 状态。
- Decision: CLI 使用项目统一 logger，同时输出 INFO 到 stderr、DEBUG/INFO 日志文件；runner 记录 spec、路径、适配器、阶段和 manifest 生命周期；FEV adapter 按任务记录加载、保存、边界、指纹、计数和耗时，并使用周期性任务进度心跳。
- Consequences: 日志只增加运行时可观测性，不改变任务选择、历史边界、asset ID、输出格式或失败语义。日志文件位于 `logs/data/extract/ref/`。

## 2026-07-29 - Catalog CLI reuses global logger

- Status: accepted
- Decision: Catalog CLI and CatalogReader use the project-wide `configure_logger` setup and emit ASCII messages with `[stage]content` message prefixes. Logs cover discovery, unique artifact hashing, Feature progress, record counts, writes, verification, completion, and failures.
- Consequences: Console output is available during long Catalog builds, and the full DEBUG log is written under `logs/data/catalog/` by the shared logger.

## 2026-07-29 — Catalog 工件指纹按唯一 RuntimeDataset 计算和校验

- Status: accepted
- Context: CatalogTrajectoryRecord 共享 Feature、Dedup 和 Load RuntimeDataset；旧实现按每条 trajectory 重复读取 Arrow 和计算完整工件 ContentHash，且发布后的立即回读再次执行完整工件校验。
- Decision: Catalog 发布阶段每个唯一 RuntimeDataset 只计算一次并将指纹写入 Catalog；发布后的结构回读不重新校验外部 RuntimeDataset。后续严格 Catalog 读取按唯一 RuntimeDataset 缓存元数据、内容指纹和 Arrow 行，记录级校验只执行行定位与字段一致性检查。
- Consequences: Catalog 仍保存并可在后续访问时验证数据版本，但不再按 trajectory 重复扫描共享工件；`CatalogReader.read(..., verify=False)` 仅适用于发布后的内部结构回读。

## 2026-07-29 — 按当前 V4 实现修正 Reference 与训练 SOP

- Status: accepted
- Context: `SOP-1.1-reference.md` 和 `SOP-2-tr-ev.md` 仍描述已删除的 runtime index、linear distribution 工件、Catalog v3 和独立 GridSpecification/test_split 字段；当前代码使用 Catalog V4 trajectory table，并在训练启动时构建内存 RuntimeDataset/DistMixer。
- Decision: 两份 SOP 统一以当前 `dataset_processing.catalog.ref`、Catalog V4、`sampling.n_bins`、`sampling.source_catalog`/`reference_catalog` 和 TS3-only `train.train` 路径为准。
- Consequences: 文档不再指导生成或配置 runtime index、linear distribution、grid_spec、test_split；评估仍只读取 checkpoint，不读取 Catalog 或 RuntimeDataset。

## 2026-07-29 — Catalog 支持跳过父元数据 Hash

- Status: accepted
- Context: RuntimeDataset 元数据 Hash 算法变更后，已有 Feature/Dedup 工件的 parent pin 可能与当前算法计算值不同，但仍需要生成 Catalog 进行后续迁移或验证。
- Decision: `dataset_processing.catalog` 和 `dataset_processing.catalog.ref` 增加 `--skip-parent-hash`。该开关只跳过 Feature→父数据和 Dedup→Load 的元数据 SHA-256 比对，仍校验父数据 ID、stage、文件内容、Arrow 结构及 Catalog 记录一致性；跳过时保留工件中原有的 parent pin。
- Consequences: 默认行为仍严格拒绝 Hash 不匹配；使用 `--skip-parent-hash` 生成的 Catalog 明确依赖调用者确认工件关系，不能证明当前 Hash 算法下的元数据完整性。

## 2026-07-29 — 将 Catalog CLI 类型参数命名为 `--type`

- Status: accepted
- Context: `python -m dataset_processing.catalog` 的 Catalog 类型参数原名为 `--catalog-kind`，与 SOP1 中的简短操作示例不一致，也使命令行显得冗余。
- Decision: 将 CLI 参数名改为必填的 `--type`，继续接受 `source` 和 `reference`；内部函数参数与 Catalog YAML 的 `catalog_kind` 字段保持不变。
- Consequences: 所有 Catalog CLI 调用必须使用 `--type source` 或 `--type reference`；旧的 `--catalog-kind` 参数不再接受。

## 2026-07-29 — 暂停 Audit 分析层直至实现 V4 读取器

- Status: accepted
- Context: 旧 `ts3.analysis.audit` 的 reader、legacy 兼容与统计说明针对已删除的 v1/v2 training-audit 格式；当前训练端只生成 Catalog V4 的 `ts3.v4-training-audit.v1` 事件。
- Decision: 删除旧 Audit 分析实现，不保留 legacy/current reader 或报告兼容入口。保留最小可导入包与 `analyze_audit`/CLI 占位入口，均立即抛出 `NotImplementedError`；源码注释标记为实现当前 V4 分析的 TODO。
- Consequences: 当前训练 audit 仍正常写入和恢复，但不能经 `ts3.analysis.audit` 读取或生成报告，直到实现基于 V4 manifest 和事件协议的新分析器。

## 2026-07-29 — 训练审计硬切换至 V4 细粒度事件协议

- Status: accepted
- Context: 原 `ts3/spec/audit.py` 要求旧 Core、旧 distribution artifact 和旧窗口尝试流程的字段；当前 Catalog V4 Core 不再产生这些事实，且旧事件不能可靠解释 V4 的原子 batch 接受语义。
- Decision: 以 `ts3.v4-training-audit.v1` 替换旧 schema，只接受 `slot_scheduled`、`window_selected`、`slot_delivered` 和 `batch_finalized`。事件记录当前 Core 可直接证明的 trajectory/scope/grid、抽样概率、合法窗口 rank、窗口边界和 slot/batch 接受结果。manifest 仅引用并固定当前 Catalog V4 与 RuntimeSourceDataset 工件；分析必须通过这些当前设施校验并 join，不复制数据资产。删除旧事件名、旧字段、legacy reader、迁移器和 fallback。
- Consequences: 新 writer 不能读取或转换此前 audit 输出；adapter-facing protocol Spec 不增加 audit 字段，细粒度选择上下文只存在于 Core 私有状态和 audit JSONL。失败 batch 以受控 ASCII 原因落盘，但 traceback 和原始数据仍只进入训练日志。

## 2026-07-29 — 清理训练与实验入口中的旧 GridIndex/LinearDistribution 链路

- Status: accepted
- Decision: `dataset_processing.catalog` 和 `run_10pct_experiment.sh` 统一使用 Catalog V4；删除实验脚本的 CatalogGridIndex 构建阶段；checkpoint manifest 改用 V4 Catalog、RuntimeDataset 和混合概率指纹，不再读取 `linear_distribution` 或 `source_index` artifact。
- Consequences: 训练启动和恢复只依赖 `ts3.spec.catalogs`、`ts3.spec.models` 生成的 RuntimeDataset 及 `DistMixer` 输出；`train.schemas` 中保留的旧字段仅用于拒绝过时配置，不代表活动实现。

## 2026-07-29 — 将 RuntimeTrajectoryReader 去旧为 RuntimeShardReader

- Status: accepted
- Context: `RuntimeTrajectoryReader` 实际按 `values`/`validity` 文件 shard 进行 mmap 读取，并不针对完整 Trajectory 或旧 Grid index；其 `runtime_grid_index` 参数和 `descriptor_registry` fallback 属于旧 API 遗留。
- Decision: 将类重命名为 `RuntimeShardReader`，构造参数统一为显式 `descriptors` mapping，并删除旧的 Grid index 兼容 fallback。
- Consequences: Core、数据处理、Runtime 导出层、测试和文档统一使用新的 shard-oriented API；读取行为、窗口合法性和 mmap LRU 行为保持不变。

## 2026-07-29 — 将核心训练协议循环集中到 `ts3/core/loop.py`

- Status: accepted
- Context: `train/runner.py` 同时负责训练启动组装和 `schedule -> consume -> accept_report` 协议循环。
- Decision: 新增 `ts3.core.schedule.loop.run_core_loop()` 负责执行通用 Core-to-adapter 循环；训练入口只通过回调保留 checkpoint 等外层编排。具体模型和 Adapter 仍由外部 Adapter 包实现。
- Consequences: Core loop 不依赖具体 Adapter 类或模型实现；训练入口的启动、恢复、审计和最终报告行为保持不变。

## 2026-07-29 — 删除未接入的 Core Scheduler 模块

- Status: accepted
- Context: `ts3/core/scheduler.py` 中的 `SlotScheduler` 没有被运行时代码或测试引用；当前实际调度逻辑由 `TS3Core.schedule_tickets()` 执行。
- Decision: 删除 `ts3/core/scheduler.py`，保留现有 `TS3Core` 调度行为，不保留兼容入口。
- Consequences: Core 的调度、采样和 pending 状态继续由 `ts3/core/core.py` 管理；项目文档不再把已删除的 `SlotScheduler` 描述为运行时组件。

## 2026-07-29 — 缩短 RuntimeDatasetBuilder 模块文件名

- Status: accepted
- Decision: 将 `ts3/core/runtime_dataset_builder.py` 重命名为 `ts3/core/runtime_builder.py`；保留 `RuntimeDatasetBuilder` 类名、公共导出和运行行为不变。
- Consequences: Core 导入和项目文档统一使用 `runtime_builder.py`，不保留旧模块路径兼容入口。

## 2026-07-29 — 将共享领域模型集中到 `ts3.spec.models`

- Status: accepted
- Decision: 将 `ts3/core/data_model.py` 与 `ts3/spec/training_protocol.py` 迁移到 `ts3/spec/models/`，由 `ts3.spec.models` 统一导出；Core 保留调度、采样、窗口访问和交付等行为实现。
- Consequences: `Grid`、`Trajectory`、`RuntimeDataset` 以及 Adapter-Core 协议对象拥有统一的模型入口；旧的 `ts3.core.data_model` 和 `ts3.spec.training_protocol` 导入路径不再提供。

## 2026-07-29 — 将外部 RuntimeDataset 工件协议命名为 `runtime_artifact.py`

- Status: accepted
- Context: `ts3/spec/runtime.py` 定义的是 Core 外部持久化 RuntimeDataset 工件的 meta、Arrow schema、文件和指纹校验，不是 Core 内存对象。
- Decision: 将文件重命名为 `ts3/spec/runtime_artifact.py`，保留现有 schema、校验逻辑和导出符号不变；文件头明确说明其外部 RuntimeDataset artifact 职责。
- Consequences: 全仓引用统一使用 `ts3.spec.runtime_artifact`；Core 内存对象继续由 `ts3.core.data_model` 定义，外部工件读取继续由 `ts3.runtime.dataset` 负责。

## 2026-07-29 — 将 Load 临时 DataFrame 契约命名为 `load_dataframe.py`

- Status: accepted
- Context: `ts3/spec/contract.py` 定义的是 raw 到 RuntimeDataset 转换期间的临时 DataFrame，而不是泛化的跨层 Contract 或 RuntimeDataset 结构。
- Decision: 将文件重命名为 `ts3/spec/load_dataframe.py`，保留现有字段、校验逻辑和导出符号不变；文件头明确说明其 Load 阶段临时 DataFrame 定义职责。
- Consequences: 全仓引用统一使用 `ts3.spec.load_dataframe`；RuntimeDataset 的内存结构由 `ts3.spec.models` 定义，持久化结构仍由 `ts3.spec.runtime_artifact` 定义。

## 2026-07-29 — Reference Catalog 统一迁移到 V4

- Status: accepted
- Decision: 删除 `ts3/spec/ref_catalog.py`，reference Catalog 发布与读取统一使用 `ts3.spec.catalogs` 的 `Catalog`、`ReferenceCatalog` 和 `CatalogReader`。
- Consequences: reference 仍在发布前校验 extraction manifest、asset scope 和 runtime provenance，但持久化格式统一为 Catalog V4；旧的独立 ReferenceCatalog v1 字段和 runtime index 不再生成。

## 2026-07-29 — 删除 V3 `ts3/spec/catalog.py`

- Status: accepted
- Context: `ts3/spec/catalog.py` 仍定义已废弃的 Catalog V3 `RuntimeCatalog`，当前 Catalog 已迁移为 V4 并由 `ts3.spec.catalogs` 负责。
- Decision: 删除 `ts3/spec/catalog.py`，ReferenceCatalog 已统一使用 `ts3.spec.catalogs`，不再保留旧的独立校验实现。
- Consequences: V3 `RuntimeCatalog` 不再可导入；Source/Reference Catalog 统一使用 V4 `ts3.spec.catalogs` 数据模型和读取器。

## 2026-07-29 — 规范化 DistMixer 模块名称

- Status: accepted
- Decision: 将 `ts3/core/distribution_mixer.py` 重命名为 `ts3/core/dist_mixer.py`；`DistMixer` 类名和运行行为保持不变。
- Consequences: Core 导出层、训练入口和项目文档统一引用 `dist_mixer.py`，不保留旧模块导入路径。

## 2026-07-29 — 分离 Grid 构建逻辑与分析逻辑

- Status: accepted
- Context: 原 `ts3/core/grid.py` 同时包含 Core 采样所需的特征校验/离散化，以及 analysis 使用的 occupancy/coverage 统计，职责边界不清晰。
- Decision: 新增 `ts3/analysis/grid.py` 持有离散化分析接口、occupancy 统计和 coverage 计算；`ts3/core/runtime_builder.py` 内部持有构建 Grid 对象所需的特征校验、坐标离散化和 `grid_id` 计算；删除独立的 `ts3/core/grid.py`。
- Consequences: Core 采样构建不依赖 analysis；analysis feature 模块从 `ts3.analysis.grid` 获取分析函数；Grid 数据对象由 `ts3.spec.models` 定义。

## 2026-07-29 — 重组 Spec、Catalog 与 Grid 包布局

- Status: accepted
- Context: 原 `ts3/schemas`、`ts3/catalog` 和 `ts3/grid` 的边界与当前 V4 结构不一致；Spec 是跨层协议定义，Catalog 是 Spec 下的数据资产声明，Grid 数学属于 Core 内部运行时能力。
- Decision: 将 `ts3/schemas` 更名为 `ts3/spec`；将原 `ts3/catalog` 全部迁移到 `ts3/spec/catalogs`；将 `ts3/grid/grid.py` 迁移到 `ts3/core/grid.py`，Grid 数据对象继续由 `ts3/core/data_model.py` 定义；删除旧包入口和旧导入路径。
- Consequences: 全仓导入、脚本、测试和项目文档统一使用 `ts3.spec`、`ts3.spec.catalogs` 与 `ts3.core.grid`；旧包路径不再提供兼容入口，行为和持久化 schema 版本保持不变。

## 2026-07-29 — 删除未使用的 `ts3/assets` 命名空间

- Status: accepted
- Context: `ts3/assets` 仅剩 legacy `__init__.py`，没有活动代码导入；旧的 `runtime.py` 源文件已删除，当前 Catalog V4 链路使用 `ts3.spec.catalogs`。
- Decision: 删除整个 `ts3/assets` 包及其残留缓存，不保留兼容入口。同步移除项目文档中对该包和旧 `runtime.py` 的描述。
- Consequences: `ts3.assets` 不再是可导入的项目命名空间；资产与 Catalog 运行时统一通过现有 `ts3.spec.catalogs` 和 `ts3.core` 模块访问。

## 2026-07-29 — 删除旧的 `ts3/sampling` 包

- Status: accepted
- Context: 旧 Sampling 包已不再属于 Catalog V4 活动链路，目录及其旧采样/索引构建入口已由用户删除。
- Decision: 删除整个 `ts3/sampling` 包，不保留兼容导入入口；当前采样实现统一使用 `ts3.core`。
- Consequences: `ts3.sampling` 不再是可导入的项目命名空间；旧采样模块、旧索引构建器和旧线性分布构建器不再提供运行时或外部兼容路径。

## 2026-07-29 — 删除数据泄露审计旧实现

- Status: accepted
- Decision: 删除 `ts3/sampling/contamination_auditor.py` 及其当前流程描述；项目不再生成或依赖 contamination/leakage report。
- Consequences: 训练启动不再提供多余的数据泄露审计入口；training audit、去重审计明细和目录授权状态继续保留。

## 2026-07-29 — 删除旧协议跳过测试入口

- Status: accepted
- Context: `test_visualize_audit.py`、`test_audit_analysis.py` 和 `test_runtime_window_delivery.py` 仅包含旧协议的模块级 skip 声明，保留它们会继续暴露已废弃测试入口。
- Decision: 直接删除上述旧测试文件；V4 行为由 `tests/test_catalog_v4_core.py` 和现有有效项目测试覆盖。
- Consequences: 项目测试目录不再包含旧协议 skip 项，V4 不提供回退测试路径。

## 2026-07-29 — 测试入口同步到 Catalog V4 Core 链路

- Status: accepted
- Context: 原测试直接构造已删除的 CatalogGridIndex、LinearDistributionArtifact 和旧 TS3Core 参数，无法验证 V4 计划要求的对象边界。
- Decision: 新增 `tests/test_catalog_v4_core.py`，按 Catalog、Builder、Mixer、Sampler、Accessor、Core 分层测试 V4 行为；旧 V3 专属测试不再作为 TS3 当前测试入口。
- Consequences: 测试明确验证 V3/未知版本拒绝、Feature-invalid 与零窗口排除、显式 lambda 校验、Source 概率采样、长度加权、stride 窗口和 opaque ticket；旧实验协议不会被兼容测试重新引入。

## 2026-07-29 — TS3 Catalog V4 取代旧索引与线性分布活动链路

- Status: accepted
- Context: 旧 TS3 活动链路以 CatalogGridIndex、旧 GridSpecification 和 LinearDistributionArtifact 连接数据资产与 Core，并通过运行时 lineage 读取窗口；这与 Catalog 作为冻结数据资产、Core 作为采样运行时的边界不一致。
- Decision: 仅接受 Catalog V4；使用扁平 Arrow trajectory table、直接 TrajectoryLocator、RuntimeDatasetBuilder、DistMixer、V4 samplers 和 WindowAccessor。Source/Reference 先分别构建 RuntimeDataset，再按显式 lambda 生成新的 Source；Core 不接收 Catalog、旧 index、旧 distribution 或 lineage 对象。
- Consequences: V3/未知/缺失 Catalog 版本立即失败；旧索引、旧分布和隐式父链兼容路径离开 TS3 活动 import graph。Feature-invalid 与零合法窗口轨迹保留在 Catalog 审计中但不进入 Source GridSet；Reference 非 Source Grid 的 mass 被丢弃并记录。
- Implementation: `ts3/spec/catalogs/`、`ts3/spec/models/data_model.py`、`runtime_builder.py`、`dist_mixer.py`、`samplers.py`、`window_accessor.py`，以及 Catalog publisher 和 TS3 training runner。

## 2026-07-29 — 内存 Catalog Mapping 必须携带来源内容 SHA-256

- Status: accepted
- Context: `build_catalog_grid_index` 支持已解析的内存 Mapping。此前当 Mapping 没有文件时，会对默认路径字符串计算 SHA-256 并将其记录为 `catalog_sha256`，这不是内容校验。
- Decision: 文件路径和带 `source_path` 的 Catalog 对象使用真实 Catalog 文件内容 SHA-256；内存 normalized Mapping 必须显式提供来源文件的 lowercase SHA-256，缺失或格式错误立即拒绝。禁止使用路径字符串生成伪 checksum。
- Consequences: 正式文件路径流程不变；测试 fixture 和其他内存调用方必须补充真实来源 digest，或明确改用文件输入。Catalog provenance 不再因路径变化而伪变化，也不会遗漏同一路径下的内容变化。

本文是影响数据协议、Schema、CLI、默认值、兼容性或实验语义的追加式历史记录。更正必须以新记录形式出现，不得静默重写既有 accepted 决策。

## 2026-07-30 — FEV provider 边界统一为 manifest 原生标量

- Context: FEV 的 `Task.cutoffs[0]` 可能返回 NumPy 整数。它与 Python `int` 数值相同，但 manifest 校验按原生 JSON 标量类型严格检查，因此完整提取在写 manifest 时失败。
- Decision: FEV adapter 在构造 extraction manifest 前，将整数边界规范化为 Python `int`；非字符串的非整数边界规范化为字符串，以满足 position/timestamp schema，不改变边界数值或安全历史范围。
- Consequences: 现有 FEV YAML spec 和 CLI 不变；provider 返回的整数标量可以正常写入并通过 manifest 校验。

## 2026-07-30 — Reference 支持完整冻结任务集的一次性提取

- Context: SOP-1.1 对“所有任务的安全历史是否可以一次性提取”说明不够直接，容易将一次提取运行、一个 Reference asset 和跨任务合并混为一谈。
- Decision: 一个冻结 extraction spec 可以在一次 `Extract/ref` 运行中处理数据集的全部所选任务。发布时仍按 `(source identity, target projection)` 分组；同组任务共享一个安全历史人口并采用共同最早安全边界。不同实体集合、target 列、维度展开或安全边界的任务必须分别发布 asset；不得仅因数据集名称相同而合并，也不得按 task 重复计数或等权加权。
- Consequences: “一次性”只表示一次冻结流程和一次 manifest 生成，不保证只有一个 asset；完整任务集应显式冻结，不能依赖可能漂移的 `all`。本决策澄清 SOP，不改变当前 benchmark adapter 的既有抽取实现。

## 2026-07-30 — Reference 输出目录中的 spec ID 仅为命名约定

- Context: SOP-1.1 使用 `data/reference/<benchmark>/<spec-id>` 作为示例，可能被理解为用户必须手动计算并输入 SHA-256 目录名。
- Decision: `--output-dir` 接受任意不存在的目录；`spec_id` 由 Extract/ref 从冻结 spec 自动计算并写入 `extraction_manifest.json`。使用 spec ID 作为目录名只是推荐的隔离和可追溯命名约定，后续流程以 manifest 路径为准，不从目录名推断身份。
- Consequences: CLI 不增加手动 hash 参数，也不改变已有输出目录兼容性；正式实验推荐用 shell 或 Python 自动生成 spec ID 目录，避免人工复制。

## 2026-07-29 — Random 可从 TS3 source Catalog 回溯 Load 数据

- Status: accepted
- Context: 为使 Random 与 TS3 选择同一 Profile-1B 数据集范围，Random 需要使用与 TS3 相同的 Catalog，而不是在训练 YAML 重复维护一份 Load 目录列表。
- Decision: Random 的 `sampling.source_catalog` 接受标准 Catalog v3。每个 Catalog entry 仍固定 Feature RuntimeDataset；Random 验证 Feature -> Dedup -> Load 的已 pin 父链，并从唯一对应的 Load RuntimeDataset 的 index/shard 构建 global-grid source index，不读取或使用 feature 值。旧的单路径或路径列表输入继续兼容。
- Consequences: Catalog v3 不新增字段，Feature 工件及其 RuntimeDataset parent pin 间接固定 Random 使用的 Load 数据。若 sibling-stage 路径、父 runtime ID 或 meta SHA-256 任一项不匹配，Random 启动立即失败。与 TS3 同样使用同一数据集范围，但 Random 仍不按 feature 有效性过滤轨迹。

## 2026-07-29 — Random 基线可聚合多个 Load RuntimeDataset

- Status: accepted
- Context: Random 原先只能配置一个 `source_runtime_dataset`，无法与使用多个 Feature RuntimeDataset 的 TS3 在相同 Profile-1B 数据集范围内训练。
- Decision: Random 配置新增非空 `sampling.source_runtime_datasets` 路径列表；旧的单路径 `sampling.source_runtime_dataset` 保持兼容。启动阶段直接聚合各 Load RuntimeDataset 的 index/shard，在一个 global grid 中按原有窗口合法性和长度质量抽样；随后由同日的 Catalog 决策新增优先的 `source_catalog` 输入，不读取 Feature 值或 Reference。
- Consequences: 这使 Random 与 TS3 可选择相同的数据集范围，但不承诺轨迹集合完全一致：TS3 会排除 feature 无效轨迹，Random 会保留其窗口合法的 Load 轨迹。生成的 Random source index 以稳定的多输入 index hash 固定输入集合，并拒绝重复目录和跨目录重复 runtime ID。

## 2026-07-29 — Reference spec 不再 pin 评测 declaration SHA-256

- Status: accepted
- Context: extraction spec 先前要求用户为 `evaluation_declaration` 手工提供 SHA-256，却没有提供项目命令生成该值，给正常的 benchmark reference 流程增加了不可执行的人工步骤。
- Decision: spec 仅保存相对的 `evaluation_declaration.path`；Fetch/ref 与 Extract/ref 不再校验 declaration SHA-256。spec、extraction manifest、native asset 和 RuntimeDataset 的既有 hash/provenance 校验保持不变。
- Consequences: 同一 spec 路径指向的 task declaration 若被修改，之后的 extraction 可能遵循新声明；因此正式实验应将 task declaration 与 spec 一同版本控制或使用新的 spec 输出目录。

## 2026-07-28 — 拆分 Source 与 Reference 数据处理 SOP

- Status: accepted
- Context: 原 `SOP-1-data.md` 同时描述 source/test 的普通数据资产链和 benchmark reference 的无泄露历史链，二者的输入、工件、Dedup 规则与 Catalog 类型不同，容易将普通 Catalog v3 与 ReferenceCatalog 混用。
- Decision: `SOP-1-data.md` 定义且只定义 Source/Test 的 `Fetch -> Load -> Dedup -> Features -> Catalog v3`；新增 `SOP-1.1-reference.md`，定义且只定义 `spec -> Fetch/ref -> Extract/ref -> Load/ref -> Features/ref -> ReferenceCatalog`。
- Consequences: SOP-2 明确分别引用 SOP-1 与 SOP-1.1；reference 的研究性选择（task selection、数据 revision）和无泄露边界在独立 SOP 中可见，不改变现有 CLI、schema 或运行行为。

## 2026-07-28 — Benchmark reference 使用不可变训练历史前缀

- Status: accepted
- Context: TS3 需要无预测 horizon 泄露的目标特征分布；既有训练已接受 reference Catalog，但没有提取和处理边界。
- Decision: 通过 benchmark 官方训练前缀 API 导出原生历史。相同原始 asset 且相同 target projection 的任务按最早安全边界合并为一个人口。
- Alternatives: validation prefix 因可能需要模型选择而拒绝；按 term 复制和 task 等权会在没有新人口时改变质量，因而拒绝。
- Consequences: 实验属于 target-domain-history-aware，不是未知目标域 zero-shot；reference 永不成为 source batch。
- Implementation: `ts3.spec.ref_extraction`、`dataset_processing.extract.ref` 与 `docs/dev/BenchmarkReferenceDataPipeline.md`。

## 2026-07-28 — Reference 复用 RuntimeDataset 并跳过 Dedup

- Status: accepted
- Context: source 已有不可变 Load/Dedup/Feature/Catalog 链。
- Decision: reference 使用 Load/ref -> Feature -> ReferenceCatalog，并复用既有 shard、Feature schema、grid index、污染审计与分布构建器；不执行 Dedup。
- Alternatives: 单独存储格式或 sampler 会重复基础设施；把 reference 当 source 会弱化角色和泄露边界，均拒绝。
- Consequences: 新持久化协议只包括集中式 extraction 与 ReferenceCatalog schema；source 行为不变。
- Implementation: `ts3.spec.ref_catalog`、`dataset_processing.load.ref` 与 `dataset_processing.catalog.ref`。

## 2026-07-28 — Dataset Processing 的 Source/Reference 物理分支与 Benchmark Adapter

- Status: accepted
- Context: Reference 不能继续作为顶层聚合模块；它必须与 Source 同样具有清晰、可检查的阶段边界，并且 benchmark 的无泄露规则不能混在统一 dispatcher 中。
- Decision: Fetch、Load、Dedup、Features、Catalog 均以 `src/` 与 `ref/` 子包表示两个分支。Reference Fetch/Dedup 为空边界；`extract/ref/fev.py`、`gift_eval.py`、`time.py` 是唯一的 benchmark Adapter。Load/ref 与 Features/ref 只复用已经由 extraction spec 标准化的原生格式和特征计算基础设施。
- Naming: 除 `reference.py` 外，Python 文件名中的 Reference 前缀一律写作 `ref_`；schema 定义位于 `ts3/spec/ref_extraction.py` 和 `ts3/spec/ref_catalog.py`。
- Consequences: 不保留 `dataset_processing.reference`、`dataset_processing.reference_catalog` 或旧 schema 文件路径；source 默认 CLI 仍保持为各阶段包根入口。

## 2026-07-28 — Reference Fetch 负责 Benchmark 数据获取

- Status: accepted
- Context: Extract/ref 只能抽取本地 benchmark 数据，无法承担下载职责；此前 Fetch/ref 的空边界使完整 Reference 管线缺少获取入口。
- Decision: `dataset_processing.fetch.ref` 使用已有 extraction spec 的 `upstream.data_root` 和冻结 revision 路由 `fev.py`、`gift_eval.py`、`time.py` 下载官方数据；Extract/ref 使用相同 spec 和本地数据根，仅执行无泄露历史前缀抽取。
- Consequences: 不新增 fetch receipt 或第二份 schema，避免与 extraction manifest 重复；Reference Dedup 仍为空边界。
## 2026-07-29 — 冻结 FEV 二十个完整任务的 extraction spec

- Context: 需要为当前 FEV RQ3 的 20 个完整任务提供可执行的 reference extraction spec，且配置必须集中在 `configs/` 下。
- Decision: 新增 `configs/ex_spec/fev/extraction_spec.json` 作为实际冻结 spec，显式选择 task index `0`–`19`；新增 `configs/ex_spec/fev/template.json` 作为对应模板，并在同目录保存 `default.yaml` declaration snapshot。由于 spec schema 禁止 declaration 路径穿越，spec/template 引用同目录 snapshot；该 snapshot 必须与 `configs/eval/fev/default.yaml` 同步。两者使用 `training_common_earliest_test` 和 `huggingface-dataset-on-disk.v1`。
- Upstream: provider 为 `autogluon/fev_datasets`，FEV code/data revision 记录为 lock 文件中的 `96535744f69e066598004632fe73f0bc534100e0`；spec 的 native cache 路径为 `../../../data/fev`，解析后为项目根目录的 `data/fev/`，Fetch/ref 直接写入该目录，不执行下载后的移动。
- Consequences: 当前 FEV reference 任务集合不会因声明文件的隐式 `all` 选择而漂移；Fetch/ref 与 Extract/ref 可直接使用该 spec。native cache、manifest 和后续 RuntimeDataset 仍必须在远端生成，不能复制到本地 Mac。
## 2026-07-29 — Reference spec 允许受控的相对路径回溯

- Context: FEV extraction spec 位于 `configs/ex_spec/fev/`，其权威 task declaration 位于 `configs/eval/fev/`；原 schema 禁止 `..`，迫使 extraction 目录保存重复的 declaration snapshot。
- Decision: `_relative` 只拒绝绝对路径，允许包含 `..` 的相对路径。FEV spec/template 改为引用 `../../eval/fev/default.yaml`，删除 `configs/ex_spec/fev/default.yaml`。
- Consequences: declaration 仍由 `configs/eval/fev/default.yaml` 单一维护；路径解析仍以 spec 目录为基准，不能使用绝对路径。调用者必须将 spec 与其引用文件一起版本控制，避免引用目标漂移。
## 2026-07-29 — Reference extraction spec 使用 YAML 作为 FEV 配置格式

- Context: extraction spec 的字段是层级配置，FEV 项目配置已经普遍使用 YAML；JSON 不是 reference schema 的实验语义要求。
- Decision: `ReferenceExtractionSpecification.from_file` 同时支持 `.json`、`.yaml` 和 `.yml`；当前 FEV 实际 spec 与模板迁移为 `configs/ex_spec/fev/extraction_spec.yaml` 和 `template.yaml`。`spec_id` 仍由规范化 JSON mapping 计算，Extract 发布的 spec snapshot 保留输入扩展名。
- Consequences: JSON 旧 spec 继续可读，YAML 成为当前 FEV 配置格式；YAML 解析后的数据类型仍接受同一严格 schema 校验。
# Decision: Standardize stage boundaries and heartbeats for CLI logging

- Date: 2026-07-31
- Change: `configs.runtime.configure_logger` now starts a daemon heartbeat for every configured project CLI logger. It records the most recent non-heartbeat log message as the current stage and emits a heartbeat to both console and the corresponding log file every 30 seconds.
- Rationale: Long-running data processing, training, and evaluation stages must visibly report that they are alive even when a single operation produces no intermediate output. Existing application behavior and experimental logic are unchanged.
- Scope: Applies automatically to all project entrypoints using `configure_logger`; third-party code that does not use the project runtime logger is unchanged.
- Hub upload/download CLIs were migrated to the same shared logger so they also receive the same file naming, console output, stage tracking, and heartbeat behavior.
# Decision: One configured logger per manually run process

- Date: 2026-07-31
- Change: Each project CLI configures exactly one top-level logger. Modules use child logger names that propagate to that logger; they do not create additional configured loggers or handlers.
- Rationale: Multiple configured loggers in one process created duplicate heartbeats and independent stale current-stage values, especially during training and Reference Catalog construction.
- Consequences: A process has one console stream, one log file, and one heartbeat. Module-specific logger names remain available for source identification through normal logger propagation.
# Decision: Standardize lightweight progress bars and ETA

- Date: 2026-07-31
- Change: `ProcessingProgress` now includes an ASCII progress bar and estimated remaining seconds whenever a total is known; unknown-total operations continue to report heartbeat, elapsed time, rate-independent counters, and current detail.
- Rationale: Heavy loading, conversion, feature, training, and evaluation stages need visible progress and an approximate ETA without requiring TQDM or changing processing behavior.
- Consequences: Existing data-processing callers automatically receive ETA-bearing progress records; callers with known totals expose the most useful estimate while indeterminate stages remain explicitly marked unknown.
# Decision: Use human-readable telemetry units

- Date: 2026-07-31
- Change: Global runtime formatting now presents durations and ETAs as compact units such as `1.2m` or `1.0h`, and byte counts as binary units such as `1.0 GiB`; progress logs use these formatters instead of raw seconds or bytes.
- Rationale: Operators can interpret progress, remaining time, and artifact sizes directly from console and log output without mental conversion.
- Consequences: Machine-readable values remain available in structured artifacts where required, while operational logger messages prioritize human readability.
## 2026-08-03 — 训练主链从 Grid 采样迁移到轨迹级 OT 质量传输（feat/ot）

- Status: accepted（方法决策；实现迁移至 feat/ot 分支）
- Context:
  - 网格在大规模主体区域估计尚可，但边界普遍存在 singleton grid，导致边界轨迹被重复曝光，
    造成边界结构估计偏差与训练分布大范围偏移。
  - FEV Reference 各任务的数据量不代表最终评测权重；当前将全部 Reference 原始轨迹合并为单一
    分布，与 FEV 任务级等权平均的评测口径不一致。
- Decision: 训练主链从 Grid-based 采样迁移到轨迹级 OT（最优传输）质量传输：
  - 每条轨迹的质量定义为其时间点数量 `trajectory.length`；不再用合法窗口数或 grid population。
  - "Task" 退化为数据事实："一个参考数据集 = 一个 catalog"。数据资产侧在 Load 阶段就把每个
    FEV task 分离为独立 bundle → 独立 catalog，即使 spec 指定多个 task 也分离。计算类只处理
    单个参考数据集；多参考集的 for 循环与等权平均在编排层完成。
  - 每个参考数据集独立完成：集内质量归一化、Source 支持过滤（质量转移距离阈值，默认 0.5，
    squared euclidean，6 维 [0,1]）、单对 OT 求解。各参考集 OT 目标最后等权平均（分母 N =
    全部参考集），再与 Random 基础分布按 `π=(1-λ)p+λq_OT` 混合。λ=0 严格等价 Random，λ=1 全 OT。
  - 超出 Source 支持的 reference 轨迹在运行时丢弃；任一参考集过滤后无质量则训练启动失败。
  - Core 改为通用 `ScheduleCore`，Random 与 OT 共用；Core 不感知 OT/Reference/λ/Grid，只消费
    轨迹级采样概率。Grid 不再属于主动训练调用图。
  - Reference 数据资产与运行时对象在 checkpoint/resume 全程只读。
- Insights:
  - Singleton 边界 grid 是离散化的结构性产物，任何 bin 数都无法同时兼顾主体密度与边界稀疏；
    轨迹级传输绕开离散化，直接在连续 6 维特征上定义质量流动，从根本上消除边界过曝。
  - 把"任务等权"作为评测口径的一部分，就必须在训练目标里显式体现：先分任务归一、再等权平均，
    而不是让数据量大的任务隐式支配目标分布。
  - 现有 20 个 FEV task 在 extract 阶段已是 20 个 1:1 独立 asset，合并只发生在 Load/Catalog 阶段，
    因此"数据层分离"是低成本改动，且无需给 Catalog 增加 task_id 字段（保持 V6）。
- Catalog: 保持 V6 不变，不新增 task_id，不升 V7。参考数据集身份即 catalog 本身。
- Migration: 代码实现在 feat/ot 分支进行；实现指导文档见该分支
  `docs/dev/TS3TrajectoryOTImplementationGuide.md`。master 仅保留本决策记录。
- Consequences:
  - 从主动训练调用图移除：FeatureGrid、GlobalGrid、GridSampler、DistMixer、TS3Core、RandomCore、
    RuntimeSourceTS3Dataset、RuntimeSourceRandomDataset、两个 grid builder、`_grid_coordinate/_grid_id`。
  - 保留复用：CatalogReader(V6)、WindowSampler、WindowAccessor(仅改构造)、ChronosTS3TrainAdapter、
    AdapterTrainingContract、ScheduledTicket/BatchWindowContent/BatchConsumptionReport、
    checkpoint/audit 框架(升版)、Trajectory/Window/WindowLocator/WindowSpecification。
  - 新增 POT(python optimal transport) 依赖，用于非平衡 Sinkhorn。
  - audit schema 升版，移除 grid 专属字段，新增轨迹级质量与概率字段。

## 2026-08-05 — Reference 缺失值宽容策略的派生工件重建与防御边界

- Status: accepted
- Context: `395528a` 将 Reference Load 从“轨迹含任意非有限值即整条标记 `load/non_finite`”改为“移除非有限观测并保留其余有限历史”。Reference 仅用于特征空间质量，训练时不作为 model context 读取，因此一个缺失值不应使整条安全历史失效。但已发布的 Bundle/Catalog 保存了旧的严格决策，代码修改不会回溯改写这些 Arrow 工件。
- Incident audit: 2026-08-05 对 `configs/catalog/ref/fev/` 的 20 个 FEV Catalog 做全量审计。发现 `jena_weather_1H` (21/21)、`rohlik_orders_1D` (7/7)、`uci_air_quality_1H` (4/4) 因旧 `load/non_finite` 规则变为空 Reference；`rossmann_1D` 还错失 1/1115 条，`uk_covid_nation_1D` 错失 8/12 条。各受影响轨迹均仍含有足够的有限观测：最小保留长度分别为 8303、362、4525、461 和 144。
- Decision:
  - Reference Load 对一维 target 过滤 `NaN` 和 `+/-inf`，仅在过滤后为空时跳过该轨迹；每条跳过值和每个 asset 汇总必须记录告警。Feature 阶段保留同等的 Reference 非有限值防御性过滤，以防止合法的直接 Bundle 输入传入不可计算特征。
  - 派生工件不提供隐式的旧规则兼容或运行时“反丢弃”回退。`load/index.arrow` 中的 `dropped` 决策是不可变输入契约；旧规则工件必须从保留的 Fetch raw 重建 `Load -> Feature -> Catalog V6`。这避免训练在未记录的情况下修改已发布的人口、feature 和质量。
  - 恢复操作只允许定向删除受影响 asset 的 `data/ref/shard_dataset/fev/<asset_id>` 和 `configs/catalog/ref/fev/<asset_id>`，不修改 `data/bench/fev` 的冻结原始 Fetch 数据。重建时按当前 spec 和冻结缓存运行 `load.ref --asset-id <asset_id>`、`features`、`catalog.ref`；离线模式可以使用已存在的 FEV 缓存，不需重新下载。
  - Training startup 继续对每个 Reference Catalog 拒绝“无 eligible trajectories”的状态；这是数据质量问题的显式防线，而不是可以静默忽略的空分布。只有在 Catalog 重建后经审计证明为空的轨迹才会在 Load 时被显式记录并跳过。
- Recovery result: 已定向重建上述 5 个 asset。完成后 `jena_weather_1H=21/21`、`rohlik_orders_1D=7/7`、`rossmann_1D=1115/1115`、`uci_air_quality_1H=4/4`、`uk_covid_nation_1D=12/12` 条有效轨迹。全体 20 个 FEV Catalog 审计为 2767 条记录、2758 条 eligible、0 个空 Catalog；同时使用 `discover_catalogs` + `RuntimeReferenceDatasetBuilder(..., "count")` 重新构建训练运行时对象，结果为 20 个非空 Runtime Reference、2758 条轨迹。
- Consequences: 此类修复的回退边界是“丢弃受影响的派生工件并从冻结 raw 重建”，而不是撤销缺失值宽容语义。现有实验配置、Catalog V6 和 Reference 发现协议不变；但任何在 2026-08-04 23:27 +0800 之前按旧规则生成、且含 `load/non_finite` 轨迹的 Reference 派生工件都不可视为对新语义兼容，须先审计再重建。
- Evidence: `/root/autodl-tmp/ts3-ref-rebuild-20260805.log` 以及 `logs/data/{load/ref,features,catalog/ref}/` 中 2026-08-05 的运行日志。
