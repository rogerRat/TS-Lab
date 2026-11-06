# 通用时序数据集来源与获取指南

> 最后修改日期：2026 年 11 月 7 日。
> 本文优先采用项目论文、官方 GitHub、官方 Hugging Face 数据仓库和原始数据发布页。大型聚合仓库的目录可能继续更新，因此本文同时提供静态分类和自动枚举方法。

## 1. 目标与口径

本文整理可用于时序分析、预测、分类、异常检测及其他任务的公开数据集来源，覆盖专题数据库、基准集合、聚合语料、派生数据和合成数据。内容重点是识别权威入口、理解数据组织方式、评估适用条件，并为可复现的数据获取和登记提供参考。

不同项目使用 `observations`、`time points`、`timesteps`、`timestamps`、`tokens` 等不同计量口径。数据点数量不能直接代表独立序列数、有效样本数或任务难度，应结合来源、变量、频率、时间范围和切分方式理解。

本文使用以下术语：

- **配置（configuration）**：数据仓库中可单独选择的数据子集、版本或任务配置。
- **子数据集（sub-dataset）**：聚合语料内部具有独立来源、许可或语义的数据集。
- **目录分组（folder/domain shard）**：仓库按领域、年份或来源组织的目录；它不一定等价于可加载配置。
- **派生数据**：由真实数据切窗、重采样、插值、缩放、混合或增强产生的数据。
- **合成数据**：由规则、统计模型、仿真器或其他生成机制构造的数据。
- **评测切分**：用于验证或测试的数据部分；其用途和访问边界应与训练数据区分。

## 2. 来源类型与适用性

下表用于描述来源的使用条件，不代表数据质量的绝对排名。

| 标记 | 一般含义 | 使用时重点检查 |
| --- | --- | --- |
| **A：稳定来源** | 来源清晰、文档完整、获取方式稳定，元数据和许可可追踪 | 固定版本，确认字段、时间范围和许可条款 |
| **B：大规模或多来源集合** | 覆盖广或规模大，但下载、存储、来源核验或处理成本较高 | 先按来源和目录抽样，估算资源并核验重叠 |
| **C：条件性来源** | 专用领域、派生、合成或任务混合数据 | 明确研究问题，记录生成链和适用边界 |
| **D：仅作评测或审计** | 测试集、泄漏风险较高的数据，或来源与许可不清的数据 | 隔离评测切分，避免未经核实地用于训练或汇总规模 |

等级针对数据使用场景和处理成本，不表示数据集本身优劣。

## 3. 来源总览

下表列出本文涉及的主要公开时序数据仓库与基准集合。入口、配置和仓库内容可能更新；实际使用前应检查原始发布页和当前版本说明。

| 数据源 | 权威入口 | 组织方式 | 使用提示 |
| --- | --- | --- | --- |
| LOTSA / Moirai | [HF: Salesforce/lotsa_data](https://huggingface.co/datasets/Salesforce/lotsa_data)；[GitHub: uni2ts](https://github.com/SalesforceAIResearch/uni2ts) | 大量来源级目录和配置 | 按原始来源、领域和版本筛选 |
| GIFT-Eval Pretrain | [HF: Salesforce/GiftEvalPretrain](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain)；[GitHub: gift-eval](https://github.com/SalesforceAIResearch/gift-eval) | 多个来源，部分按年份或处理方式拆分 | 区分训练来源与评测切分 |
| Time-300B | [HF: Maple728/Time-300B](https://huggingface.co/datasets/Maple728/Time-300B)；[GitHub: Time-MoE](https://github.com/Time-MoE/Time-MoE) | 领域目录、来源目录和大文件分片 | 先估算存储及处理成本 |
| RMISC | [HF: nju-zhangsq/RMISC](https://huggingface.co/datasets/nju-zhangsq/RMISC)；[GitHub](https://github.com/zhangsq-nju/RMISC) | 版本分支及来源级子数据集 | 核对版本差异、元数据和多变量结构 |
| Chronos datasets | [HF: chronos_datasets](https://huggingface.co/datasets/autogluon/chronos_datasets)；[extra](https://huggingface.co/datasets/autogluon/chronos_datasets_extra)；[GitHub](https://github.com/amazon-science/chronos-forecasting) | 原始、增强、合成和评测配置并存 | 按数据生成方式和任务切分区分 |
| UTSD | [HF: thuml/UTSD](https://huggingface.co/datasets/thuml/UTSD)；[GitHub: OpenLTM](https://github.com/thuml/OpenLTM) | 多种规模版本 | 核对规模版本之间是否嵌套或重叠 |
| Time Series PILE | [HF: AutonLab/Timeseries-PILE](https://huggingface.co/datasets/AutonLab/Timeseries-PILE)；[MOMENT GitHub](https://github.com/moment-timeseries-foundation-model/moment) | 多任务、多来源聚合语料 | 先审查任务类型、来源与切分 |
| Monash | [HF: Monash-University/monash_tsf](https://huggingface.co/datasets/Monash-University/monash_tsf)；[官方站](https://forecastingdata.org/) | 频率、来源和缺失值版本构成的配置 | 核对基准版本及标准切分 |
| Quito Corpus / QuitoBench | [Corpus](https://huggingface.co/datasets/hq-bench/quito-corpus)；[Bench](https://huggingface.co/datasets/hq-bench/quitobench)；[GitHub](https://github.com/alipay/quito) | `hour`、`min` 两种配置及独立 benchmark | 区分原始语料和评测基准 |

---

# 4. 主要时序数据来源

## 4.1 LOTSA / Moirai

### 4.1.1 权威源头

- 数据仓库：[Salesforce/lotsa_data](https://huggingface.co/datasets/Salesforce/lotsa_data)
- 官方代码：[SalesforceAIResearch/uni2ts](https://github.com/SalesforceAIResearch/uni2ts)
- 论文：[Unified Training of Universal Time Series Forecasting Transformers](https://arxiv.org/abs/2402.02592)

LOTSA 是 Moirai 的公开预训练归档，论文报告超过 270 亿 observations，覆盖九个领域。Hugging Face 仓库以来源级目录存放 Arrow 数据，目录名通常比论文中的领域标签更细。

### 4.1.2 配置与子数据集分组

LOTSA 的根目录数量较多，而且可能随仓库维护发生变化。与其依赖静态目录截图，不如按原始来源和领域登记配置。当前可识别的主要类别如下。

| 领域 | 代表性目录或来源家族 | 适用性 | 说明 |
| --- | --- | --- | --- |
| 能源 | `australian_electricity_demand`、电力负荷、电价、太阳能、风电、建筑能耗类 | **A/B** | 优先保留现有语料中未覆盖的国家、频率和测量对象 |
| 交通 | `PEMS03`、其他 PEMS、`BEIJING_SUBWAY_30MIN`、`HZMETRO`、出租车和道路流量类 | **B** | 与 UTSD、Monash、Chronos 重叠概率高，必须做源级排除 |
| 气象与气候 | ERA5 年份目录、CMIP6 年份目录、WeatherBench/气象站类 | **B/C** | 数据量极大；适合后续 scaling，不适合当前全量处理 |
| 零售与销售 | 商店销售、产品需求、竞赛销售类 | **A/B** | 若能确认未被 Monash/Chronos 收录，可补充非物理过程 |
| 金融与经济 | 汇率、宏观经济、股票、银行和竞赛数据 | **A/B** | 高频金融数据需单独处理非平稳性和交易日历 |
| Web 与云服务 | Web traffic、服务器、请求量、云指标类 | **A** | 可补充不同于气象数据的业务与过程类型 |
| 健康 | 医疗监测、疾病统计、生命体征类 | **A/C** | 价值较高，但需逐项核验使用许可和隐私处理 |
| 自然科学 | 海洋、地球科学、天文或环境监测类 | **B** | 适合长周期和多尺度结构扩展 |
| 竞赛数据 | M 系列、Tourism、NN5、Kaggle 等 | **D/B** | 与 Monash 等聚合库高度重叠；默认不重复纳入 |

已在官方仓库中可直接观察到的目录示例包括：

```text
australian_electricity_demand
BEIJING_SUBWAY_30MIN
HZMETRO
PEMS03
cmip6_<year>
era5_<year>
```

这些名称只表示当前仓库中的部分配置，不应被视为完整清单。

### 4.1.3 推荐策略

**优先使用：**

- Web、云服务、工业监控、健康和未覆盖地区的能源数据；
- 明确具有独立原始来源、独立采样频率和独立许可证的目录；
- 能够保留原始 source ID、变量组和时间范围的数据。

**留作后续大规模扩展：**

- ERA5、CMIP6 和大规模交通传感器；
- 大量同质化气象网格或空间节点；
- 需要 TB 级落盘和高成本窗口去重的目录。

**默认排除：**

- 已经通过 Monash、UTSD 或 Chronos 纳入的竞赛和经典基准；
- 无法从目录名或元数据恢复原始来源的派生副本；
- 评测测试段。

### 4.1.4 自动枚举当前目录

```python
from huggingface_hub import HfApi

api = HfApi()
entries = api.list_repo_tree(
    repo_id="Salesforce/lotsa_data",
    repo_type="dataset",
    recursive=False,
)

configs = sorted(
    item.path for item in entries
    if item.path and "/" not in item.path and item.path not in {
        "README.md", ".gitattributes"
    }
)

print("\n".join(configs))
```

建议把枚举结果保存为带 revision 的清单：

```text
manifests/lotsa_<commit_sha>.csv
```

---

## 4.2 GIFT-Eval Pretrain

### 4.2.1 权威源头

- 预训练语料：[Salesforce/GiftEvalPretrain](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain)
- 评测集：[Salesforce/GiftEval](https://huggingface.co/datasets/Salesforce/GiftEval)
- 官方代码：[SalesforceAIResearch/gift-eval](https://github.com/SalesforceAIResearch/gift-eval)
- 论文：[GIFT-Eval](https://arxiv.org/abs/2410.10393)

官方说明将预训练语料描述为 71 个单变量和 17 个多变量来源数据集，共 88 个来源，约 450 万条序列和 2300 亿数据点。仓库中的实际配置数可能多于 88，因为 ERA5、CMIP6 等来源可能按年份或处理版本拆分。

### 4.2.2 配置家族

| 配置家族 | 常见命名形式或代表来源 | 适用性 | 处理建议 |
| --- | --- | --- | --- |
| 经典竞赛 | `m1_*`、`m3_*`、`m4_*`、Tourism、NN5 等 | **D/B** | 与 Monash 高度重叠，原则上源级排除 |
| 电力与能源 | electricity、solar、wind、load、price、building energy | **A/B** | 只保留未覆盖地区和不同测量对象 |
| 交通 | traffic、PEMS、metro、taxi、rideshare | **B** | 与 LOTSA、UTSD 和 Chronos 交叉核验 |
| 气象再分析 | `era5_<year>` | **B** | 年份是分片，不是独立科学来源；按年份采样即可 |
| 气候模拟 | `cmip6_<year>` 或模型/情景分片 | **B/C** | 适合大规模扩展，避免全部并入主池 |
| 金融与经济 | FRED、exchange、stock、bitcoin、macro 类 | **A/B** | 注意交易日历、缺失值和非平稳性 |
| Web 与 IT | web traffic、server、cloud、requests | **A** | 对结构覆盖有较高增量价值 |
| 健康与人口 | hospital、medical、mortality、epidemic 类 | **A/C** | 逐项核验许可与时间切分 |
| 零售与需求 | sales、retail、product demand | **A/B** | 可补充非物理过程结构 |
| 多变量环境与工业 | 多传感器、环境、工业控制类 | **A/B** | 可用于多变量数据分析 |

官方仓库中可观察到的配置示例包括：

```text
nn5_daily_with_missing
cmip6_1865
cmip6_1885
era5_1989
era5_1990
era5_1992
era5_1993
era5_1996
```

这些年份目录应在 `canonical_source_id` 层合并为同一来源家族，而不是被视为完全独立的数据集。

### 4.2.3 推荐策略

若使用 GIFT-Eval 评测，应核对其官方训练数据与评测切分说明，避免训练和评测来源交叉。

否则：

- 优先抽取 Web、工业、多变量环境、未覆盖能源和零售配置；
- 经典 Monash 配置默认排除；
- ERA5/CMIP6 只做分层抽样；
- GIFT-Eval 本身的 benchmark 配置标记为 `evaluation_exclusion=true`，不得进入训练。

### 4.2.4 自动枚举

```python
from huggingface_hub import HfApi

api = HfApi()
entries = api.list_repo_tree(
    repo_id="Salesforce/GiftEvalPretrain",
    repo_type="dataset",
    recursive=False,
)

configs = sorted(
    item.path for item in entries
    if item.path and "/" not in item.path
)

for name in configs:
    print(name)
```

还可用 `datasets` 获取可加载配置：

```python
from datasets import get_dataset_config_names

configs = get_dataset_config_names("Salesforce/GiftEvalPretrain")
print(len(configs))
print("\n".join(sorted(configs)))
```

---

## 4.3 Time-300B / Time-MoE

### 4.3.1 权威源头

- 数据仓库：[Maple728/Time-300B](https://huggingface.co/datasets/Maple728/Time-300B)
- 官方代码：[Time-MoE/Time-MoE](https://github.com/Time-MoE/Time-MoE)
- 论文：[Time-MoE](https://arxiv.org/abs/2409.16040)

Time-300B 是按领域和来源组织的大规模语料。论文报告超过 3000 亿时间点，覆盖九个以上领域，并含少量合成数据。项目仓库不是一个简单的几十个 Hugging Face 配置集合，更接近经过处理的大型文件树。

### 4.3.2 领域配置

| 领域 | 内容 | 适用性 | 建议 |
| --- | --- | --- | --- |
| `Nature` / 自然环境 | 气象、气候、地球与环境过程 | **B/C** | 占比极高，适合研究偏斜；不宜直接按原比例进入主池 |
| `Energy` | 负荷、发电、价格、设备与建筑能耗 | **A/B** | 优先选取未覆盖国家、设备和频率 |
| `Transport` | 道路、公共交通、轨迹和流量 | **B** | 与 UTSD/LOTSA 重叠核验后使用 |
| `Web` | 网站访问、云服务、请求量和系统指标 | **A** | 建议优先，通常能补充不同于物理传感器的结构 |
| `Finance` / `Economics` | 市场、汇率、宏观与业务指标 | **A/B** | 建议按来源分层抽样 |
| `Health` | 医疗、人口和生理数据 | **A/C** | 许可明确时优先 |
| `Sales` / `Retail` | 商品、门店和需求 | **A/B** | 可增加零售、门店和需求类序列 |
| `Industrial` / IoT | 机器、设备和工业传感器 | **A** | 若源信息可追踪，应优先 |
| `Synthetic` | 项目生成或汇总的合成序列 | **C** | 与真实数据分开建库，用于消融 |
| 其他/Misc | 无法直接归入主要领域的数据 | **C** | 必须先恢复原始来源 |

### 4.3.3 推荐策略

**当前推荐：**

```text
Web
-> Industrial / IoT
-> Health
-> Sales / Retail
-> 未覆盖的 Energy
-> 未覆盖的 Finance / Economics
```

**留作大规模实验：**

```text
Nature
-> 大规模 Weather / Climate
-> 全量 Transport
```

Time-300B 的 Nature 数据占比较高。使用这类大规模领域数据时，可分别评估领域构成和数据规模的影响：

1. 保持原始分布，验证结构调度是否能纠正极端来源偏斜；
2. 在相同数据预算下进行领域均衡或结构均衡采样。

合成部分必须设置：

```text
is_synthetic = true
wrapper_corpus = "Time-300B"
```

不得与真实观测的规模直接合并报告。

### 4.3.4 自动枚举

```python
from huggingface_hub import HfApi

api = HfApi()
entries = api.list_repo_tree(
    repo_id="Maple728/Time-300B",
    repo_type="dataset",
    recursive=False,
)

for item in entries:
    print(item.path)
```

对于 TB 级仓库，建议先生成目录统计，不下载数据文件：

```python
from huggingface_hub import HfApi

api = HfApi()
files = api.list_repo_files(
    repo_id="Maple728/Time-300B",
    repo_type="dataset",
)

top_level = {}
for path in files:
    root = path.split("/", 1)[0]
    top_level[root] = top_level.get(root, 0) + 1

print(top_level)
```

---

## 4.4 RMISC

### 4.4.1 权威源头

- 数据仓库：[nju-zhangsq/RMISC](https://huggingface.co/datasets/nju-zhangsq/RMISC)
- 官方代码：[zhangsq-nju/RMISC](https://github.com/zhangsq-nju/RMISC)
- 论文：[RMISC: A Large-scale Real-world Multivariate Corpus for Time Series Foundation Models](https://arxiv.org/abs/2607.06504)

RMISC 是较新的真实世界多变量语料，官方报告约 200 个子数据集、200 万个原始文件、160 亿 timesteps 和 1420 亿标量 time points。其重要价值不是单纯规模，而是保留多变量组和较完整的来源信息。

### 4.4.2 官方版本/分支

| 分支或版本 | 含义 | 适用性 | 建议 |
| --- | --- | --- | --- |
| `recommended_corpus` | 项目方推荐用于训练和复现实验的语料组合 | **A** | 可优先评估的版本 |
| `smaller_version` | 更小、便于验证管线的版本 | **A** | 先用于格式、速度和特征抽取测试 |
| `main` | 完整来源组织或主版本 | **B** | 仅按子数据集选择性下载 |
| `zipped_version` | 压缩归档版本 | **B/C** | 适合批量归档，不利于流式和选择性读取 |
| Parquet 转换分支 | Hugging Face 自动或项目提供的 Parquet 版本 | **A/B** | 优先用于结构扫描，但需记录转换 revision |

### 4.4.3 子数据集领域

RMISC 的约 200 个子数据集不应被压缩成几个 Hugging Face 配置名。应按原始数据源登记。官方描述覆盖的主要领域包括：

| 领域 | 典型内容 | 适用性 |
| --- | --- | --- |
| Energy | 建筑能耗、电网、发电、储能和设备 | **A** |
| Industry | 机器、多传感器、生产过程和状态监控 | **A** |
| Traffic | 道路、车辆、公共交通和流量 | **B** |
| Environment | 气象、空气质量、水文和环境传感器 | **A/B** |
| Finance | 市场、经济和业务指标 | **A/B** |
| Healthcare | 生理信号、医疗监测和健康统计 | **A/C** |
| IoT / Smart Building | 房间、设备、家居和建筑传感器 | **A** |
| Other | 其他真实世界多变量过程 | **C** |

仓库示例中可见按独立来源命名的子目录，例如：

```text
ApplianceEnergy
```

实际完整名称应从所选 revision 自动枚举。

### 4.4.4 推荐策略

比较不同版本时，可按以下顺序核验：

```text
smaller_version
-> recommended_corpus
-> main 中与现有来源不重叠的 Industry / IoT / Health
-> main 中其余来源
```

使用 RMISC 时应保留：

```text
_original_filename
canonical_source_id
variable_group_id
original_split
license
```

多变量数据不应在入库前无条件拆成互不相关的单变量序列。建议同时保留：

- 原始多变量轨迹；
- 可用于当前单变量实验的变量视图；
- 从变量视图回溯到原始多变量组的索引。

### 4.4.5 自动枚举指定分支

```python
from huggingface_hub import HfApi

api = HfApi()

for revision in [
    "recommended_corpus",
    "smaller_version",
    "main",
    "zipped_version",
]:
    print(f"\n## {revision}")
    try:
        entries = api.list_repo_tree(
            repo_id="nju-zhangsq/RMISC",
            repo_type="dataset",
            revision=revision,
            recursive=False,
        )
        for item in entries:
            print(item.path)
    except Exception as exc:
        print("Unavailable:", exc)
```

---

## 4.5 Chronos datasets

### 4.5.1 权威源头

- 主仓库：[autogluon/chronos_datasets](https://huggingface.co/datasets/autogluon/chronos_datasets)
- 补充构建源：[autogluon/chronos_datasets_extra](https://huggingface.co/datasets/autogluon/chronos_datasets_extra)
- 官方代码：[amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)
- 论文：[Chronos](https://arxiv.org/abs/2403.07815)

该仓库同时包含原始/处理后来源数据、Chronos 训练语料、增强数据、合成数据和评测数据。不能把整个仓库大小解释为独立真实 observations。

### 4.5.2 配置类型

| 类型 | 代表配置 | 适用性 | 说明 |
| --- | --- | --- | --- |
| Chronos 增强训练语料 | `training_corpus_tsmixup_10m` | **C** | 1000 万条 TSMixup 增强序列；不是独立真实来源 |
| Chronos 合成训练语料 | `training_corpus_kernel_synth_1m` | **C** | 100 万条 KernelSynth 序列；单独用于合成消融 |
| 原始来源训练段 | Chronos 所用公共数据的训练部分 | **A/B** | 必须恢复原始数据集名称并与 Monash/UTSD 去重 |
| in-domain 评测配置 | Chronos 训练来源中的保留测试段 | **D** | 不得并入训练 |
| zero-shot 评测配置 | 训练时未见数据的测试配置 | **D** | 应保持评测隔离 |
| extra 构建配置 | 原始数据下载和转换脚本对应的来源 | **B** | 适合追踪 provenance，不应与主仓库重复计数 |

### 4.5.3 经典来源配置家族

Chronos 评测和训练来源涉及多种 Monash/经典预测集合，通常包括下列家族：

```text
M1 / M3 / M4
Tourism
NN5
Electricity
Traffic
Solar
Weather
Australian electricity demand
Kaggle web traffic
Taxi / rideshare
Hospital
FRED / economic
Bitcoin / finance
Pedestrian
Wind farms
```

准确清单应以所使用 revision 的官方 YAML 和 HF 配置为准。

### 4.5.4 推荐策略

Chronos 仓库包含多类内容，整理时应按实际来源拆分，而不要只以仓库名称作为来源单位：

```text
chronos_real_source
chronos_tsmixup
chronos_kernel_synth
chronos_eval_only
```

具体建议：

- 真实来源：按 `canonical_source_id` 与 Monash、UTSD 做差集；
- TSMixup：单独保留，用于增强数据消融；
- KernelSynth：单独保留，用于合成结构覆盖实验；
- in-domain 和 zero-shot 测试段：设置 `evaluation_exclusion=true`；
- 无法确定来源的训练 shard：不进入“独立真实观测”统计。

### 4.5.5 枚举配置

```python
from datasets import get_dataset_config_names

configs = get_dataset_config_names("autogluon/chronos_datasets")
for name in sorted(configs):
    print(name)
```

同时扫描官方评测 YAML：

```bash
git clone https://github.com/amazon-science/chronos-forecasting.git
find chronos-forecasting -type f \( -name "*.yaml" -o -name "*.yml" \) -print
```

---

## 4.6 UTSD

### 4.6.1 权威源头

- 数据仓库：[thuml/UTSD](https://huggingface.co/datasets/thuml/UTSD)
- 相关代码：[thuml/OpenLTM](https://github.com/thuml/OpenLTM)
- 仓库提供 `source.csv`，应作为来源映射的重要依据。

### 4.6.2 官方规模配置

| 配置/目录 | 近似规模 | 适用性 | 建议 |
| --- | ---: | --- | --- |
| `UTSD-1G` | 约 10 亿点 | **A** | 最适合快速复现和小规模对照 |
| `UTSD-2G` | 约 20 亿点 | **A/B** | 用于中等规模扩展 |
| `UTSD-4G` | 约 40 亿点 | **B** | 用于 scaling 研究 |
| `UTSD-12G` | 约 120 亿点 | **B** | 最大版本，处理前先检查与小版本的包含关系 |

这些版本可能是层级或抽样版本，而不是四套完全独立的来源。使用时应记录：

```text
utsd_version
source_dataset
is_nested_version
```

不能把 1G、2G、4G、12G 直接相加为 19G 独立数据。

### 4.6.3 来源领域

UTSD 官方描述覆盖：

```text
Energy
Environment
Health
IoT
Nature
Traffic
Web
```

具体子数据集应由 `source.csv` 解析，而不是仅根据 Arrow 文件名推断。

### 4.6.4 推荐策略

使用 UTSD 时：

- 当前主语料固定一个明确版本；
- 若已有数据来自多个版本，先核验是否存在层级包含；
- `source.csv` 中每一行映射到统一 `canonical_source_id`；
- 后续实验使用更大版本时，作为 scaling axis，而不是作为新来源。

### 4.6.5 下载

```bash
hf download thuml/UTSD \
  --repo-type dataset \
  --include "UTSD-1G/**" "source.csv" \
  --local-dir ./data/utsd
```

---

## 4.7 Time Series PILE / MOMENT

### 4.7.1 权威源头

- 数据仓库：[AutonLab/Timeseries-PILE](https://huggingface.co/datasets/AutonLab/Timeseries-PILE)
- 官方模型与处理代码：[moment-timeseries-foundation-model/moment](https://github.com/moment-timeseries-foundation-model/moment)
- 论文：[MOMENT](https://arxiv.org/abs/2402.03885)

Time Series PILE 覆盖预测、分类、异常检测等任务，约 13 个领域。它的价值是任务多样性，而不是作为纯预测预训练语料的最大独立数据源。

### 4.7.2 配置方式

PILE 的数据组织应从两个维度登记：

| 维度 | 类别 |
| --- | --- |
| 任务 | Forecasting、Classification、Anomaly Detection、Imputation/Pretraining windows |
| 领域 | Healthcare、Energy、Traffic、Weather/Climate、Finance、Human activity、Sensors、Industrial、Web、Nature 等 |

其内部会包含或转换经典数据源，例如：

```text
Monash forecasting datasets
UCR/UEA classification datasets
anomaly-detection benchmark datasets
long-horizon forecasting benchmarks
```

这些名称代表来源家族，而不是全部配置名。

### 4.7.3 推荐策略

Time Series PILE 是任务和来源混合的集合。用于预测任务前，应检查其任务组成、来源和切分，避免将不适用的数据直接混入训练集：

1. 与 Monash 和其他经典 benchmark 重叠明显；
2. 分类和异常检测数据的样本定义与 forecasting trajectory 不同；
3. 某些窗口化版本可能重复暴露同一原始轨迹。

更合理的用法：

| 用途 | 建议 |
| --- | --- |
| 纯 forecasting 主实验 | 只取未重复的预测来源，等级 **C** |
| 跨任务结构分析 | 保留分类、异常检测和预测来源，等级 **A/C** |
| 特征空间覆盖检查 | 可用于检查所选结构特征是否跨任务稳定 |
| 训练规模扩展 | 优先级低于 RMISC、LOTSA 差集和 Time-300B 特定领域 |

### 4.7.4 枚举配置

```python
from datasets import get_dataset_config_names

configs = get_dataset_config_names("AutonLab/Timeseries-PILE")
print("\n".join(sorted(configs)))
```

若仓库不是标准多配置脚本，则枚举根目录：

```python
from huggingface_hub import HfApi

api = HfApi()
for item in api.list_repo_tree(
    "AutonLab/Timeseries-PILE",
    repo_type="dataset",
    recursive=False,
):
    print(item.path)
```

---

## 4.8 Monash Forecasting Repository

### 4.8.1 权威源头

- 官方网站：[Monash Forecasting Repository](https://forecastingdata.org/)
- Hugging Face：[Monash-University/monash_tsf](https://huggingface.co/datasets/Monash-University/monash_tsf)
- 论文：[Monash Time Series Forecasting Archive](https://arxiv.org/abs/2105.06643)

Monash 是预测领域最稳定、最容易复现的来源之一。官方仓库包含约 30 个数据集和 58 个频率/缺失值变体。

### 4.8.2 Hugging Face 主要配置

以下按来源家族整理。实际配置名以 HF `get_dataset_config_names` 输出为准。

#### 能源与环境

```text
australian_electricity_demand
electricity_hourly
electricity_weekly
solar_10_minutes
solar_weekly
wind_farms_minutely
weather
saugeenday
```

#### 交通与移动

```text
traffic_hourly
traffic_weekly
pedestrian_counts
rideshare_without_missing
rideshare_with_missing
vehicle_trips_without_missing
vehicle_trips_with_missing
```

#### 零售、需求与 Web

```text
car_parts
dominick
kaggle_web_traffic_daily
```

#### 金融、经济与公共统计

```text
bitcoin
fred_md
hospital
covid_deaths
```

#### 智能电表

```text
london_smart_meters_without_missing
london_smart_meters_with_missing
```

#### 竞赛和经典预测集合

```text
cif_2016

m1_yearly
m1_quarterly
m1_monthly

m3_yearly
m3_quarterly
m3_monthly
m3_other

m4_yearly
m4_quarterly
m4_monthly
m4_weekly
m4_daily
m4_hourly

nn5_daily
nn5_weekly

tourism_yearly
tourism_quarterly
tourism_monthly
```

#### 其他高频或专项集合

```text
kdd_cup_2018
```

部分版本可能在官方站和 HF 中采用略有差异的命名，尤其是带缺失值/不带缺失值的变体。

### 4.8.3 稳定性分级

| 子集类型 | 等级 | 说明 |
| --- | --- | --- |
| Electricity、Traffic、Weather、Solar、Wind | **A** | 经典、稳定、容易追踪 |
| M1/M3/M4、Tourism、NN5、CIF | **A** | 权威竞赛数据，但与其他聚合语料重复率高 |
| London smart meters、rideshare、vehicle trips | **A/B** | 结构丰富，需明确 missing variant |
| Bitcoin、FRED-MD | **A/B** | 非平稳性强，适合扩展结构空间 |
| KDD Cup 2018、Kaggle Web Traffic | **B** | 规模或处理成本较高 |
| `*_with_missing` 与 `*_without_missing` 同时使用 | **C** | 属于同源变体，不应作为独立来源重复计数 |

### 4.8.4 数据集使用提示

Monash 是常用的预测基准集合。使用时应优先核对数据集版本、来源登记和标准切分：

- 明确当前使用了哪些配置；
- 缺失值版本是否重复纳入；
- 与 Chronos、LOTSA、GIFT-Eval、PILE 的同源副本建立排除表；
- 竞赛数据的训练/测试段是否保持隔离。

### 4.8.5 获取完整配置名

```python
from datasets import get_dataset_config_names

configs = get_dataset_config_names(
    "Monash-University/monash_tsf",
    trust_remote_code=True,
)

print(len(configs))
print("\n".join(sorted(configs)))
```

---

## 4.9 Quito Corpus 与 QuitoBench

### 4.9.1 权威源头

- 训练语料：[hq-bench/quito-corpus](https://huggingface.co/datasets/hq-bench/quito-corpus)
- 评测集：[hq-bench/quitobench](https://huggingface.co/datasets/hq-bench/quitobench)
- 官方代码：[alipay/quito](https://github.com/alipay/quito)
- 论文：[Quito](https://arxiv.org/abs/2603.26017)

Quito Corpus 是单一真实业务来源的多变量长序列语料。QuitoBench 从同一项目中构造结构化测试集合，二者用途必须分开。

### 4.9.2 配置

| 仓库 | 配置 | 数据粒度 | 适用性 |
| --- | --- | --- | --- |
| `hq-bench/quito-corpus` | `hour` | 小时级五变量序列 | **A** |
| `hq-bench/quito-corpus` | `min` | 十分钟级五变量序列 | **A** |
| `hq-bench/quitobench` | `hour` | 小时级结构化评测 | **D** |
| `hq-bench/quitobench` | `min` | 十分钟级结构化评测 | **D** |

### 4.9.3 推荐策略

优先级：

```text
quito-corpus/hour
-> quito-corpus/min
```

理由：

- `hour` 规模更易管理，适合先做特征一致性和多变量处理验证；
- `min` 提供更高频、更长的局部结构，适合后续扩展；
- QuitoBench 属于评测基准；应按其发布说明用于评测，不要与 Quito Corpus 的训练数据角色混淆。

建议设置：

```text
canonical_source_id = "quito_business_system"
wrapper_corpus = "quito-corpus"
frequency_config = "hour" | "min"
evaluation_exclusion = false   # corpus
evaluation_exclusion = true    # quitobench
```

### 4.9.4 加载

```python
from datasets import load_dataset

quito_hour = load_dataset(
    "hq-bench/quito-corpus",
    "hour",
    split="train",
)

quito_min = load_dataset(
    "hq-bench/quito-corpus",
    "min",
    split="train",
)
```

---

# 5. 数据集发现与选择流程

## 5.1 明确用途和边界

获取数据前先确定任务类型、目标领域、时间频率、变量结构、所需时间跨度、数据许可和评测边界。预测、分类、异常检测、插补和表示学习对标签、切分和数据形态的要求不同；不要只按仓库规模选源。

## 5.2 分阶段核验

| 阶段 | 核验内容 | 记录项 |
| --- | --- | --- |
| 来源确认 | 原始发布方、聚合仓库、配置和版本之间的关系 | 来源标识、仓库地址、revision、配置名 |
| 小样本检查 | 字段结构、时间戳、频率、缺失值、变量数和序列长度 | schema、抽样范围、已知异常 |
| 用途检查 | 任务标签、训练/验证/测试切分、许可和潜在泄漏 | 允许用途、切分规则、排除项 |
| 成本评估 | 下载体积、处理时间、存储需求和依赖 | 估算值、机器环境、获取命令 |
| 固定版本 | 保存可复现清单并记录转换步骤 | revision、文件清单、转换链和日期 |

先检查少量配置或样本，再决定是否获取完整数据。对于大型仓库，优先按来源目录、年份、领域或明确配置筛选，并核验仓库当前实际目录后再使用下载模式。

## 5.3 区分数据角色

原始观测、重采样数据、增强数据、合成数据和评测数据应分别标记。聚合仓库的名称不等于原始来源；同一原始数据可能通过多个仓库和不同处理版本发布。汇总数据规模时，应说明计量单位、处理阶段和切分范围。

# 6. 数据源注册与去重

## 6.1 必需字段

```text
canonical_source_id
source_family
upstream_dataset
wrapper_corpus
repository_id
repository_revision
config_name
original_split
domain
subdomain
frequency
num_variates
is_multivariate
is_synthetic
is_augmented
license
transform_chain
evaluation_exclusion
original_filename
source_url
```

建议补充：

```text
time_range_start
time_range_end
geographic_scope
entity_type
sampling_policy
missing_value_policy
resampling_policy
parent_multivariate_id
```

## 6.2 去重层次

```text
L0：仓库与配置登记
L1：canonical source 去重
L2：原始文件和轨迹 ID 去重
L3：时间戳和值完全匹配
L4：重采样、缩放和平移后的高相似轨迹
L5：滑动窗口级重复暴露控制
```

### L0：仓库重复

同一数据可能同时出现在：

```text
Monash
Chronos
LOTSA
GIFT-Eval Pretrain
Time Series PILE
UTSD
Time-300B
```

`wrapper_corpus` 不等于 `canonical_source_id`。

### L1：来源重复

例如 M4 Yearly 在不同聚合库中格式不同，但仍是同一原始来源。

### L2/L3：精确重复

建议生成两类 hash：

```text
raw_file_hash
canonical_series_hash
```

规范化 hash 前应统一：

- 时间排序；
- NaN 表示；
- 浮点字节序；
- 时间戳时区；
- 明确是否包含静态或动态协变量。

### L4：派生重复

需要识别：

- 重采样；
- 固定比例缩放；
- 常数平移；
- 子区间裁剪；
- 不同窗口长度；
- 同一多变量轨迹拆出的变量视图。

### L5：窗口曝光

即使原始轨迹没有重复，同一长序列也可能产生大量高度重叠窗口。数据处理报告应分别列出：

```text
unique source series
unique canonical windows
total sampled windows
total observations exposed
```

---

# 7. 自动生成配置清单

建议不要手工维护大型仓库的全部目录，而是生成 revision-locked manifest。

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from huggingface_hub import HfApi


@dataclass
class RepoEntry:
    repo_id: str
    revision: str
    path: str
    entry_type: str


def list_top_level(
    repo_id: str,
    revision: str = "main",
) -> list[RepoEntry]:
    api = HfApi()
    entries = api.list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        recursive=False,
    )

    result: list[RepoEntry] = []
    for item in entries:
        result.append(
            RepoEntry(
                repo_id=repo_id,
                revision=revision,
                path=item.path,
                entry_type=item.__class__.__name__,
            )
        )
    return result


repos = [
    ("Salesforce/lotsa_data", "main"),
    ("Salesforce/GiftEvalPretrain", "main"),
    ("Maple728/Time-300B", "main"),
    ("nju-zhangsq/RMISC", "recommended_corpus"),
    ("autogluon/chronos_datasets", "main"),
    ("thuml/UTSD", "main"),
    ("AutonLab/Timeseries-PILE", "main"),
    ("Monash-University/monash_tsf", "main"),
    ("hq-bench/quito-corpus", "main"),
]

rows: list[RepoEntry] = []
for repo_id, revision in repos:
    try:
        rows.extend(list_top_level(repo_id, revision))
    except Exception as exc:
        rows.append(
            RepoEntry(
                repo_id=repo_id,
                revision=revision,
                path=f"ERROR: {exc}",
                entry_type="Error",
            )
        )

df = pd.DataFrame(asdict(row) for row in rows)
Path("manifests").mkdir(exist_ok=True)
df.to_csv("manifests/official_repo_inventory.csv", index=False)
```

对于标准 Hugging Face 多配置数据集：

```python
from datasets import get_dataset_config_names

repos = [
    "Salesforce/GiftEvalPretrain",
    "autogluon/chronos_datasets",
    "AutonLab/Timeseries-PILE",
    "Monash-University/monash_tsf",
    "hq-bench/quito-corpus",
    "hq-bench/quitobench",
]

for repo_id in repos:
    try:
        names = get_dataset_config_names(
            repo_id,
            trust_remote_code=True,
        )
        print(repo_id, len(names))
        for name in sorted(names):
            print("  ", name)
    except Exception as exc:
        print(repo_id, "ERROR", exc)
```

清单生成后，应人工补充：

```text
canonical_source_id
domain
license
is_synthetic
is_augmented
evaluation_exclusion
recommendation_level
```

---

# 8. 下载建议

## 8.1 安装

```bash
pip install -U "huggingface_hub[cli]" datasets pyarrow pandas
```

## 8.2 小规模优先下载

```bash
hf download hq-bench/quito-corpus \
  --repo-type dataset \
  --local-dir ./data/quito

hf download nju-zhangsq/RMISC \
  --repo-type dataset \
  --revision smaller_version \
  --local-dir ./data/rmisc_smaller
```

## 8.3 推荐版本

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nju-zhangsq/RMISC",
    repo_type="dataset",
    revision="recommended_corpus",
    local_dir="./data/rmisc_recommended",
)
```

## 8.4 按目录选择

```python
snapshot_download(
    repo_id="Salesforce/lotsa_data",
    repo_type="dataset",
    allow_patterns=[
        "TARGET_CONFIG_A/**",
        "TARGET_CONFIG_B/**",
        "README.md",
    ],
    local_dir="./data/lotsa_subset",
)
```

不要在未查看仓库当前目录前照搬示例领域名作为 `allow_patterns`。应先运行第 7 节的枚举脚本。

## 8.5 流式检查

```python
from datasets import load_dataset

dataset = load_dataset(
    "autogluon/chronos_datasets",
    "training_corpus_tsmixup_10m",
    split="train",
    streaming=True,
)

for sample in dataset.take(5):
    print(sample.keys())
```

流式读取适合：

- 检查 schema；
- 估计长度分布；
- 读取来源字段；
- 小规模抽样计算目标分析特征；
- 判断是否值得完整下载。

---

# 9. 使用建议

选择时序数据来源时，应优先确认原始发布方、版本、许可、配置结构、变量组织、采样频率、时间覆盖和任务切分。对聚合仓库，按其原始来源清单理解数据组成，不要把仓库名直接当成来源标识。

下载前先做小规模 schema 和数据质量检查，再估算完整获取与处理成本。保留固定版本清单和转换记录，并区分原始数据、派生数据、合成数据及评测数据。若报告数据规模，应同时说明统计单位与处理口径；只有经过相应核验后，才报告独立来源、独立序列或去重规模。
