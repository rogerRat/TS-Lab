# SOP-2: TS3 训练与评估

> 历史归档：保留当时的设计、操作或实验证据，不作为当前执行指令。当前入口见 `docs/README.md`。

> 历史 SOP：旧 OT/KDE 训练工作流及 scripts/run.py 已退役，以下命令不再是操作入口。
> 当前工具见 [scripts/STD.md](../../../scripts/STD.md)，安装见 [requ/INSTALL.md](../../../requ/INSTALL.md)。

本 SOP 描述当前可执行的 Random、OT、KDE、Dynamic-OT 与 Dynamic-KDE 训练和评估路径。训练
接收 Catalog V6：Random 只读取 Source Catalog；非 Random 先由传输缓存构建读取同一 benchmark
下的 Source/Reference Catalog，再由训练加载该缓存。非 Random 训练必须手动写明这些 reference
对应的 benchmark task index，防止后续忘记模型参考了哪个任务。模型训练只从缓存恢复的最终
source 采样分布取窗口，绝不在训练启动中重新求解 OT/KDE。

## 当前训练与评估契约

本节是当前实现的权威约定，优先于本文后续尚未迁移的旧 grid/`ts3` 示例。Random 的 sampling
只含 Source；OT/KDE 及其 Dynamic 变体必须额外声明 Reference Catalog 发现根和它们代表的
评估任务。非 Random 配置先生成一次传输采样缓存，训练仅加载并验证这一工件。

| 配置字段 | 含义解释 | 示例 | 是否必须 | 默认值 | 允许输入值类型 |
| --- | --- | --- | --- | --- | --- |
| `sampling.policy` | 训练采样方法 | `ot` | 是 | 无 | `random`、`ot`、`kde`、`d-ot` 或 `d-kde` 字符串 |
| `sampling.source_catalog` | Source Catalog V6 文件 | `../../catalog/src/1b.yaml` | 是 | 无 | 相对路径字符串 |
| `sampling.lambda` | Reference target 的混合权重 | `0.5` | OT/KDE 是；Random 禁止 | 无 | `[0,1]` float |
| `sampling.reference_catalog_dir` | Reference Catalog V6 发现根；单任务实验必须是该任务唯一 catalog 的目录 | `../../catalog/ref/fev/<long_bundle_derived_name>/` | OT/KDE 是；Random 禁止 | 无 | 相对路径字符串 |
| `sampling.reference_task_record.benchmark` | Reference 对应的评测 benchmark | `fev` | OT/KDE 是 | 无 | `fev`、`time` 或 `gifteval` 字符串 |
| `sampling.reference_task_record.task_indices` | 人工维护的 Reference 评测任务索引 | `[13]` | OT/KDE 是 | 无 | 非空、无重复的非负整数 list |
| `sampling.ot` | OT 求解器参数 | `{solver: pot_sinkhorn_unbalanced, ...}` | OT 是 | 无 | mapping |
| `sampling.kde` | KDE 求解器参数 | `{feature_metric: squared_euclidean, ...}` | KDE 是 | 无 | mapping |

FEV task 13 的 Rossmann 单任务训练声明示例：

```yaml
sampling:
  policy: ot
  lambda: 0.5
  source_catalog: ../../catalog/src/1b.yaml
  reference_catalog_dir: ../../catalog/ref/fev/<long_bundle_derived_name>/
  reference_task_record:
    benchmark: fev
    task_indices: [13]
```

缓存构建与训练启动会把 `reference_task_record` 写入 cache/run manifest 和 checkpoint 的
`ts3_checkpoint_manifest.json`。这项记录是人工一致性标记，不会从 catalog 路径或 asset 名称
推断，也不实现 hash 级血缘追踪。

> **强制规则：`task_indices` 不会筛选 Reference Catalog。** 当前传输缓存构建实现只对
> `sampling.reference_catalog_dir` 调用 catalog discovery，并构建该目录之下发现的**全部**
> Reference Catalog；`sampling.reference_task_record.task_indices` 仅写入 cache/run/checkpoint manifest，
> 用于后续评估一致性警告。因而单任务 FEV 实验不得把目录写为
> `../../catalog/ref/fev/` 再期待 `task_indices: [i]` 过滤到 task `i`。这会混入全部 20 个
> FEV catalog，并可能因无有效轨迹的无关 reference 在启动阶段失败。必须将目录写为 task `i`
> 对应 asset 的唯一子目录，并让 `task_indices: [i]` 与该目录保持一一对应。

FEV 的公共入口仍是 `python -m evaluate.run`。常规配置以 `expected_tasks` 评估前 N 个任务；
需要精确选择任务时，使用 `task_indices`，并要求其长度等于 `expected_tasks`：

```yaml
bench_config:
  task_file: default.yaml
  expected_tasks: 1
  task_indices: [13]
```

| 评估字段 | 含义解释 | 示例 | 是否必须 | 默认值 | 允许输入值类型 |
| --- | --- | --- | --- | --- | --- |
| `bench_config.task_file` | FEV 20-task declaration | `default.yaml` | FEV 是 | 无 | 相对路径字符串 |
| `bench_config.expected_tasks` | 本次任务数 | `1` | FEV 是 | 无 | 正整数 |
| `bench_config.task_indices` | 指定真实 FEV task index 的 selector | `[13]` | 否 | 未设置时选择前 N 个 | 非空且无重复的非负整数 list |

FEV 评估会从 checkpoint 读取 `reference_task_record`。选择的任务集合与训练 reference 不相同时，
会输出带 `====== REFERENCE TASK MISMATCH ======` 分隔线的非阻塞警告。使用旧 checkpoint、
无法读取该字段时也会给出同样显眼的不可校验警告。

在 AutoDL 上需要访问 Hugging Face 的 FEV 数据前，执行：

```bash
source /etc/network_turbo
export HF_ENDPOINT=https://hf-mirror.com
python -m evaluate.run --checkpoint <checkpoint> --config <config> --output-dir <output>
```

已创建但因网络中断未完成的评估，必须使用相同 checkpoint、config 和 output directory 加
`--resume` 继续，不能覆盖或重建已有运行元数据。

## 1. 训练前交接

先按 SOP-1 发布 `configs/catalog/src/1b.yaml`，再按 SOP-1.1 发布同一 benchmark 下的
Reference Catalog 目录，例如 `configs/catalog/ref/fev/<long_bundle_derived_name>/`。其中 `<long_bundle_derived_name>` 必须是以 `fev-%2F` 开头、由实际 bundle 路径编码得到的目录名；不能使用 dataset_config 别名。Catalog YAML 使用 `version: 6`；
每个 reference asset 是一个独立的子目录和 Catalog header。训练不使用网格或 `n_bins`。

训练启动会将下列新工件写入 output；非 Random 训练在此前还必须已有相配的传输缓存：

```text
run_manifest.json
training_audit/manifest.json
training_audit/events.jsonl
checkpoints/checkpoint-final/
final_report.json
```
`audit.compression: true` 时事件文件名为 `training_audit/events.jsonl.gz`；
`audit.level`（verbose/standard/minimal）决定记录的事件子集，`audit.enabled: false`
时不创建 `training_audit/`。

非 Random 缓存位于 `outputs/cache/<run_id>/<policy>/`，包含 `manifest.json` 与
`sampling.safetensors`；它保存 Source 分段几何、原始质量和 lambda=1 的目标质量，不保存
Feature 向量、Reference 原始数据或活的 sampler。训练不写 source/reference runtime index、
linear distribution 或 CatalogGridIndex 工件。
不要提供 `cata_index`、Canonical/Feature Parquet、`grid_spec` 或
`catalog_grid_index`；这些字段和接口已移除。

### Reference 输入边界

`sampling.reference_catalog_dir` 必须是 SOP-1.1 中由 `dataset_processing.catalog.ref` 发布的
Reference Catalog 发现目录；不能是 Source Catalog、原生 benchmark 目录、Load/Dedup/Feature
目录或 extraction manifest。传输缓存构建自动发现该目录中的 Reference Catalog V6 headers。对于
**单任务** FEV 训练，该目录必须只包含对应 task 的一个 header，例如 task 13 使用
`configs/catalog/ref/fev/<long_bundle_derived_name>/`，而不是共同父目录；运行入口会拒绝短名称目录
`configs/catalog/ref/fev/`。OT/KDE 还必须手动声明 `sampling.reference_task_record`，它记录这些
reference 要对应的 benchmark 和 task index，不以 catalog 目录名作自动推断或筛选。

启动单任务 OT/KDE 训练前，必须完成下面的交接检查：

1. 从 `configs/eval/fev/default.yaml` 查出 task index 对应的 `dataset_config`；
2. 确认 `reference_catalog_dir` 只发现这一个 catalog，而不是 FEV 的共同父目录；
3. 确认该 catalog 存在至少一条 feature 有效、未 dropped 的 reference trajectory；
4. 确认 YAML 的 `reference_task_record.task_indices` 与该 task index 完全相同。

第 2 项不能由第 4 项替代：`reference_task_record` 是审计声明，不是运行时筛选器。

reference 只影响传输缓存中保存的 target 分段质量；训练启动按 `sampling.lambda` 将该质量与
缓存保存的 source 原始分段质量混合。Core 之后只从混合后的 source-backed RuntimeDataset
抽样；reference 原始数据、Feature、Catalog 和 manifest 不会在任何 batch、epoch、checkpoint
恢复或评估中被修改。

使用 benchmark 历史构建 reference 的实验必须标注为 target-domain-history-aware/adaptation，
不能宣称为未知目标域 zero-shot。

### 传输缓存：训练前的 OT/KDE 阶段

`assets.py`、SOP-1 和 SOP-1.1 的职责止于发布 Source/Reference Bundle 与 Catalog。非 Random
的 OT/KDE（包括 Dynamic-OT 和 Dynamic-KDE）在独立的训练前步骤计算；入口为：

```bash
python -m ts3.core.transfer.cache_builder --config <train.yaml> --run-id <run-id>
```

`<run-id>` 必须与随后 `train.train --output-dir <output-dir>` 推导的运行标识完全一致，即
`run-<output-dir 的最后一个路径分量>`。例如输出目录为
`outputs/train/fev/task-03/d-ot-task-03` 时，缓存命令必须使用
`--run-id run-d-ot-task-03`，并发布到
`outputs/cache/run-d-ot-task-03/d-ot/`。不同 task 绝不能共用相同的输出目录末段，否则会共用
缓存键并覆盖或误读彼此的缓存。

缓存构建会验证 Source/Reference Catalog，读取 Feature，计算 lambda=1 的目标质量，并优先
使用 CUDA 执行可迁移的 OT/KDE 数值计算；CUDA 或 PyTorch 不可用时自动回退 CPU，结果工件格式
保持相同。构建成功后缓存默认不可覆盖；只有确认同一未完成运行需要重新计算时才显式传入
`--overwrite`。不得为已经完成或正在恢复的运行随意覆盖其缓存。

训练端会重新验证当前 Source runtime、窗口规格、策略及 Source/Reference 配置与缓存 manifest
一致，再按 YAML 的 `sampling.lambda` 将缓存内的 raw/target 质量混合为采样分布。它不会重新
读取 Reference Feature 或重新执行 OT/KDE；缓存缺失、不匹配或损坏都会使训练在模型构建前失败。
Random 不产生也不读取传输缓存。

对 `configs/scripts/run.yaml` 定义的 FEV 批量实验，优先使用：

```bash
python scripts/run.py [TASK ...]
```

该编排器为每个非 Random task 建立任务唯一输出目录，并在调用 `train.train` 前自动运行
`ts3.core.transfer.cache_builder`。`scripts/assets.py` 只在需要新建 Source/Reference 资产时运行，不能替代
缓存构建入口。

## 2. 固定的可运行训练 YAML

批量 FEV 实验的当前入口是 `configs/scripts/run.yaml` 与 `python scripts/run.py`；脚本会为每个
task/method 生成训练 YAML 并发布其所需缓存。下列直接训练 YAML 是单次运行示例：它们均相对于
自身目录解析 Catalog 路径，使用同一个 Profile-1B Chronos 初始化模型、窗口几何和优化器，唯一
变量是 `sampling.policy`。直接运行任一非 Random YAML 前必须先按上一节构建同名缓存。

| 算法 | 配置文件 | 数据输入 | 适用含义 |
| --- | --- | --- | --- |
| Random | `configs/train/chronos/random-profile-1b.yaml` | `configs/catalog/src/1b.yaml` | 仅在 Profile-1B Source 上随机抽样的基线。 |
| OT | `configs/train/chronos/ts3-profile-1b-fev.yaml` | `configs/catalog/src/1b.yaml` 与 `configs/catalog/ref/fev/` | 以 FEV 20 个任务的 Reference 与 Source 混合，再用 unbalanced Sinkhorn OT 求解。 |
| KDE | `configs/train/chronos/ts3-profile-1b-fev-kde.yaml` | `configs/catalog/src/1b.yaml` 与 `configs/catalog/ref/fev/` | 以同一 Reference/Source 输入，用 KDE（平方欧氏距离）构建目标分布。 |

训练 YAML 字段如下。

| 配置字段 | 含义解释 | 示例 | 是否必须 | 默认值 | 允许输入类型 |
| --- | --- | --- | --- | --- | --- |
| `sampling.policy` | 当前训练策略 | `random`、`ot` 或 `kde` | 是 | 无 | 枚举字符串 |
| `sampling.lambda` | reference target 的混合权重；仅 OT/KDE | `0.5` | OT/KDE 是；Random 禁止 | 无 | `[0,1]` float |
| `sampling.source_catalog` | Source Catalog V6 | `../../catalog/src/1b.yaml` | 是 | 无 | 相对路径字符串 |
| `sampling.reference_catalog_dir` | FEV Reference Catalog V6 发现根；仅 OT/KDE | `../../catalog/ref/fev/` | OT/KDE 是；Random 禁止 | 无 | 相对路径字符串 |
| `context_length` | context 长度 | `512` | 是 | 无 | 正整数 |
| `prediction_length` | future 长度 | `64` | 是 | 无 | 正整数 |
| `window_stride` | 合法窗口起点间隔 | `1` | 是 | 无 | 正整数 |
| `missing_value_policy` | 记录到窗口资格和 manifest 的策略标识 | `reject` | 是 | 无 | 非空 ASCII 字符串 |
| `seed` | scheduler 与进程 RNG seed | `0` | 是 | 无 | 非负整数 |
| `batch_size` | 一次 optimizer step 的 slot 数 | `32` | 是 | 无 | 正整数 |
| `training_steps` | optimizer 更新总次数；slot 总预算自动计算为 `training_steps * batch_size` | `100000` | 是 | 无 | 正整数 |
| `adapter_class` | 适配器类 | `adapters.chronos.train.adapter:ChronosTS3TrainAdapter` | 是 | 无 | `module:Class` 字符串 |
| `adapter_config` | Chronos adapter 配置 | `{model: ..., chronos: ..., optimizer: ..., training: ...}` | 是 | 无 | mapping |
| `runtime_shard_mmap_cache_capacity` | Core shard mmap 缓存容量 | `8` | 否 | `1` | 正整数 |
| `save_checkpoint_steps` | 中途 checkpoint 间隔 | `10000` | 否 | `0` | 非负整数 |
| `resume_checkpoint` | 需要严格恢复的项目 checkpoint | `output/old/checkpoints/checkpoint-step-0000010000` | 否 | 无 | 路径字符串 |

`adapter_config.model` 必须包含 `random_init`、`tie_embeddings`、`device`，并在
`model_id` 与 `init_checkpoint` 中二选一。`chronos` 必须包含 model/tokenizer 规定字段；
`optimizer` 必须为 `{name, kwargs}`；`training` 当前只允许 `drop_prob`。顶层或 nested
配置中的 `n_bins`、`alpha`、`metadata-balanced`、`ts3-e`、`catalog_grid_index` 均无效。

Chronos 训练 adapter 的子字段如下；`optimizer.kwargs` 原样传给 `torch.optim.AdamW`
或 `torch.optim.Adam`，因此其中允许的键由所选 PyTorch optimizer 决定。

| 配置字段 | 含义解释 | 示例 | 是否必须 | 默认值 | 允许输入类型 |
| --- | --- | --- | --- | --- | --- |
| `adapter_config.model.model_id` | Hugging Face 模型标识；与 `init_checkpoint` 二选一 | `google/t5-efficient-tiny` | 二选一 | 无 | 非空 ASCII 字符串 |
| `adapter_config.model.init_checkpoint` | 新训练使用的本地模型目录；与 `model_id` 二选一 | `models/base` | 二选一 | 无 | 已存在目录路径 |
| `adapter_config.model.random_init` | 是否只读取 model config 后随机初始化 | `false` | 是 | 无 | 布尔值 |
| `adapter_config.model.tie_embeddings` | 随机初始化时是否绑定词嵌入 | `false` | 是 | 无 | 布尔值 |
| `adapter_config.model.device` | 模型加载与训练设备 | `cuda` | 是 | 无 | 非空 ASCII 字符串 |
| `adapter_config.chronos.model_type` | Chronos 模型类型；当前训练实现最终只接受 seq2seq | `seq2seq` | 是 | 无 | `seq2seq` 或 `causal` 字符串 |
| `adapter_config.chronos.tokenizer_class` | Chronos tokenizer 类名 | `MeanScaleUniformBins` | 是 | 无 | 非空 ASCII 字符串 |
| `adapter_config.chronos.tokenizer_kwargs` | tokenizer 构造参数 | `{low_limit: -15.0, high_limit: 15.0}` | 是 | 无 | mapping |
| `adapter_config.chronos.n_tokens` | tokenizer 总 token 数 | `4096` | 是 | 无 | 正整数 |
| `adapter_config.chronos.n_special_tokens` | special token 数 | `2` | 是 | 无 | 正整数 |
| `adapter_config.chronos.pad_token_id` | padding token 编号 | `0` | 是 | 无 | 非负整数，且小于 special token 数 |
| `adapter_config.chronos.eos_token_id` | EOS token 编号 | `1` | 是 | 无 | 非负整数，且小于 special token 数 |
| `adapter_config.chronos.use_eos_token` | 是否使用 EOS token | `true` | 是 | 无 | 布尔值 |
| `adapter_config.optimizer.name` | optimizer 类型 | `AdamW` | 是 | 无 | `AdamW` 或 `Adam` |
| `adapter_config.optimizer.kwargs` | 所选 optimizer 的原生参数 | `{learning_rate: 0.001, weight_decay: 0.0}` | 是 | 无 | mapping |
| `adapter_config.training.drop_prob` | 保留在规范化配置中的概率字段；当前 Chronos adapter 不读取它，不施加缺失增强 | `0.0` | 是 | 无 | `[0,1]` float |

## 3. 启动训练

Random 基线（只使用 Source）命令：

```bash
python -m train.train --config configs/train/chronos/random-profile-1b.yaml --output-dir outputs/train/random-profile-1b --seed 0
```

OT 命令（先构建使用 Source 与 FEV Reference 的缓存）：

```bash
python -m ts3.core.transfer.cache_builder \
  --config configs/train/chronos/ts3-profile-1b-fev.yaml \
  --run-id run-ts3-profile-1b-fev
python -m train.train \
--config configs/train/chronos/ts3-profile-1b-fev.yaml \
--output-dir outputs/train/ts3-profile-1b-fev \
--seed 0
```

KDE 命令（先构建使用同一 Source 与 FEV Reference 的缓存）：

```bash
python -m ts3.core.transfer.cache_builder \
  --config configs/train/chronos/ts3-profile-1b-fev-kde.yaml \
  --run-id run-ts3-profile-1b-fev-kde
python -m train.train \
  --config configs/train/chronos/ts3-profile-1b-fev-kde.yaml \
  --output-dir outputs/train/ts3-profile-1b-fev-kde \
  --seed 0
```

Random 训练严格按以下顺序：解析 YAML；读取并验证 Source Catalog；构建内存 RuntimeDataset
和随机分布；写 run manifest；创建 training audit；固定 Python/NumPy/Torch CPU/CUDA RNG；构建
adapter；由 adapter contract 构建 Core；随后循环
`schedule_tickets -> consume -> request_window -> accept_report`，直到成功 slot 数正好
达到 budget。非 Random 训练的差异仅在启动时：读取并验证 Source Catalog 与已发布的传输缓存，
从缓存重建混合采样分布，然后进入完全相同的 manifest、audit、adapter 和 Core 流程。每个 batch
只有一次 optimizer step；当前不支持 gradient accumulation 或
部分 batch 消费。

Core 的窗口抽样权不可移交给 adapter：adapter 只按顺序用 opaque ticket ID 请求窗口，
Core 在合法 start 中均匀取样并返回 values。adapter 的成功报告必须带回完全相同、同一
顺序的 delivery ID；任何窗口异常、模型异常、报告失败或不匹配都会终止运行，绝不重采样。

## 4. Checkpoint 与恢复

每次成功训练都会写 `checkpoints/checkpoint-final`。`save_checkpoint_steps > 0` 时，
每个已确认且整除该间隔的 optimizer step 另写一个 step checkpoint。checkpoint 原子
包含模型文件、`ts3_training_state.pt` 和 `ts3_checkpoint_manifest.json`；后两者固定
adapter/optimizer、Core scheduler、RNG、审计 offset、source RuntimeDataset 直接状态和分布
状态，不计算或保存 checksum。运行根目录的 `training_audit/` 是唯一审计副本；checkpoint
仅以相对 `audit_source` 和 `audit_offset` 引用它，不复制审计文件。因此 checkpoint 必须保留在
其所属运行目录中才能恢复。旧 checkpoint 若自带 `training_audit/`，仍按旧格式兼容恢复。

恢复必须使用新的 `--output-dir`，在 YAML 顶层填写 `resume_checkpoint`。除总 budget
必须大于已有 progress、`save_checkpoint_steps` 可以改变、以及新增 resume 路径外，当前
配置与 checkpoint 内 run manifest 必须完全相同。不能把 `init_checkpoint` 当作恢复，且
它不能与 `resume_checkpoint` 同时出现。

## 5. FEV 与 TIME 评估

训练完成后以项目 checkpoint 运行统一入口。`configs/eval/fev/default.yaml` 是 FEV
20-task 声明，不是 `--config` 可直接接收的运行配置；运行配置通过其
`bench_config.task_file` 引用该任务声明。假设当前位于项目根目录：

Random 基线：

```bash
python -m evaluate.run \
--checkpoint outputs/train/random-profile-1b/checkpoints/checkpoint-final \
--config configs/eval/fev/random-profile-1b.yaml \
--output-dir outputs/eval/random-profile-1b
```

OT：

```bash
python -m evaluate.run \
--checkpoint outputs/train/ts3-profile-1b/checkpoints/checkpoint-final \
--config configs/eval/fev/ts3-profile-1b.yaml \
--output-dir outputs/eval/ts3-profile-1b
```

KDE：FEV adapter 与 OT 完全相同，评估配置只记录模型标识，因此可复制现有配置并修改
`model.id`（例如改为 `ts3-profile-1b-fev-kde`），然后运行：

```bash
python -m evaluate.run \
  --checkpoint outputs/train/ts3-profile-1b-fev-kde/checkpoints/checkpoint-final \
  --config configs/eval/fev/ts3-profile-1b-fev-kde.yaml \
  --output-dir outputs/eval/ts3-profile-1b-fev-kde
```

建议在项目根目录创建该配置：

```bash
cp configs/eval/fev/ts3-profile-1b.yaml configs/eval/fev/ts3-profile-1b-fev-kde.yaml
sed -i 's/ts3-profile-1b/ts3-profile-1b-fev-kde/g' configs/eval/fev/ts3-profile-1b-fev-kde.yaml
```

如需比较中间 checkpoint，分别对三种模型重复上述 FEV 命令，并为每个 checkpoint 使用独立
的 `--output-dir`；已有未完成目录必须加 `--resume`，不得覆盖 provenance。

TIME：

```bash
python -m evaluate.run --checkpoint outputs/train/ts3-profile-1b-fev/checkpoints/checkpoint-final --config configs/eval/time/official_all.yaml --output-dir outputs/eval/ts3-profile-1b-fev/time-official-all
```

| CLI 参数变量名称 | 含义解释 | 示例 | 是否必须 | 默认值 | 允许输入值类型 |
| --- | --- | --- | --- | --- | --- |
| `--checkpoint` | 完整项目 checkpoint 目录 | `outputs/train/ts3-profile-1b/checkpoints/checkpoint-final` | 是 | 无 | 已存在的目录路径 |
| `--config` | 严格 FEV 或 TIME 运行 YAML | `configs/eval/fev/random-profile-1b.yaml` 或 `configs/eval/time/official_all.yaml` | 是 | 无 | 已存在的 YAML 文件路径 |
| `--output-dir` | 新评估输出目录 | `outputs/eval/ts3-profile-1b` 或 `outputs/eval/ts3-profile-1b/time-official-all` | 是 | 无 | 不存在的目录路径 |
| `--resume` | 在 provenance 一致时复用已有评估目录 | `--resume` | 否 | 关闭 | 布尔开关 |

评估 YAML 顶层固定为 `schema_version`、`benchmark`、`model`、`adapter`、
`bench_config`。checkpoint 只能通过 CLI 传入。FEV 运行 YAML 的
`bench_config.task_file` 指向只含 `tasks` 顶层字段的 FEV 任务声明（当前为
`configs/eval/fev/default.yaml`），不能将该任务声明直接传给 `--config`。FEV/TIME 不读取 Catalog、Core、
RuntimeDataset 或 training audit；它们只从 checkpoint 加载模型并执行各自 benchmark。

| 评估 YAML 字段 | 含义解释 | 示例 | 是否必须 | 默认值 | 允许输入值类型 |
| --- | --- | --- | --- | --- | --- |
| `schema_version` | 统一评估 schema 版本 | `1` | 是 | 无 | 当前仅整数 `1` |
| `benchmark` | 选择评估后端 | `fev` | 是 | 无 | `fev` 或 `time` |
| `model.id` | 记录结果的模型标识 | `ts3_run_001_final` | 是 | 无 | ASCII 安全标识字符串 |
| `adapter.class` | 对应 benchmark 的公开适配器类 | `adapters.chronos.eval.fev:ChronosFEVEvaluationAdapter` | 是 | 无 | `module:Class` 字符串 |
| `adapter.config` | 适配器模型加载与推理参数 | `{model: ..., inference: ...}` | 是 | 无 | mapping |
| `bench_config` | benchmark 私有数据、任务和来源参数 | `{task_file: default.yaml, expected_tasks: 20}` | 是 | 无 | mapping |

FEV adapter 的 `adapter.config` 必须有 `model.device`、`model.torch_dtype` 与
`inference.batch_size`、`num_samples`、`temperature`、`top_k`、`top_p`、`seed`；
`num_samples >= 2`，`top_k` 为 `0` 或至少 `2`，`top_p` 在 `(0,1]`。TIME adapter 在
此基础上还要求 `inference.context_length` 和严格递增、含 `0.5` 的
`inference.quantile_levels`。各 benchmark 的 `bench_config` 由对应 backend 继续严格
校验，现有示例见 `configs/eval/fev/` 与 `configs/eval/time/`。

## 6. 当前禁止的旧接口

不得使用 Catalog v1/v2、旧 `runtime_index`、`cata_index`、Canonical/Feature Parquet、
预生成 CatalogGridIndex、`legacy grid index builder`、`metadata-balanced`/`ts3-e` policy、
`alpha`、`--action`、`--index`、旧 adapter factory 或失败 batch 的替代采样。当前
`train.train` 当前支持 `sampling.policy=random`、`ot` 与 `kde`。Random 仅使用 Source Catalog；
OT/KDE 使用 Source Catalog 和显式 FEV Reference Catalog V6 目录。三者都不接受任何旧 Catalog
或 Hash 兼容字段。

## 增量更新（2026-08-14）

当前训练实现新增或固化以下运行事实：

1. `python scripts/run.py` 是批量 FEV 入口，支持 `TASK`、`--run-id`、`--dry-run` 和
   `-N/--processes`；它为每个 task/method 隔离输出目录，并在非 Random 训练前构建不可变
   transfer cache。
2. 训练循环显示 tqdm 进度，并按 `logging_interval_steps` 输出带 experiment、run、policy、
   optimizer steps、delivered slots、loss 和 ETA 的统一日志。loss 同时追加到 `losses.csv`，
   并生成 `checkpoints/losses_by_step.png`；这些是可归档的结果工件。
3. lambda-annealed sampling 使用有序 Core 调度；online DataLoader 在该模式下自动关闭，
   以保持采样刷新顺序。刷新事件会记录 optimizer step、progress、lambda 和 refresh interval。
4. FEV 评估入口仍为 `python -m evaluate.run`。同一 checkpoint、config 和 output directory
   的中断评估用 `--resume`；已存在且完成的评估输出可由当前显式复用契约读取，不应覆盖已有
   运行元数据。
5. 统一日志配置由 `configs/logger.py` 提供；训练、缓存构建和编排器应使用同一日志格式，
   归档时保留 run manifest、final report、loss 曲线和评估元数据即可。checkpoint 与
   `training_audit` 是否保留由实验归档策略单独决定。
