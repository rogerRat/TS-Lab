# TSFM 公开语料、配置清单与 TS3 数据集补充建议

> 更新日期：2026 年 7 月 20 日。  
> 本文优先采用项目论文、官方 GitHub、官方 Hugging Face 数据仓库和原始数据发布页。大型聚合仓库的目录可能继续更新，因此本文同时提供静态分类和自动枚举方法。

## 1. 目标与口径

本文面向 TS3 的语料扩展，回答四个问题：

1. 每个公开 TSFM 语料的权威获取源头是什么；
2. 仓库包含哪些配置、子数据集、领域分组或版本；
3. 哪些配置适合优先作为稳定数据源纳入 TS3；
4. 哪些配置更适合留作后续大规模实验，而不是立即并入主语料。

需要区分以下概念：

- **配置（configuration）**：Hugging Face `load_dataset(repo_id, config_name)` 中可选择的配置。
- **子数据集（sub-dataset）**：聚合语料内部具有独立来源、独立许可证或独立语义的数据集。
- **目录分组（folder/domain shard）**：仓库按领域、年份或来源组织的文件夹；它不一定等价于 Hugging Face 配置。
- **派生语料**：由真实数据切窗、重采样、插值、混合或增强产生的数据。
- **合成语料**：由核函数、状态空间模型、规则系统或其他生成机制产生的数据。
- **评测配置**：用于验证或测试的配置，不应默认并入预训练语料。

不同项目使用 `observations`、`time points`、`timesteps`、`timestamps`、`tokens` 等不同口径。本文不把这些数字视为严格可比。

## 2. 推荐等级

本文使用四级标记：

| 等级 | 含义 | TS3 中的建议 |
| --- | --- | --- |
| **A：优先稳定源** | 来源清晰、真实数据、获取稳定、结构增量明显、许可证可追踪 | 可进入当前主语料候选池 |
| **B：大规模扩展源** | 数据量大或覆盖广，但下载、去重、授权或质量核验成本较高 | 留作第二阶段扩展实验 |
| **C：条件性使用** | 仅在特定评测、领域偏斜、合成数据或多变量研究中有明确价值 | 单独建实验分支，不默认并入 |
| **D：不进入训练主池** | 评测数据、测试切分、泄漏风险、来源不清或纯派生重复 | 只作评测、审计或消融 |

等级针对 TS3 当前目标，而不是对数据集质量作一般性评价。

---

## 3. 总览

| 语料 | 官方主入口 | 配置形态 | 当前建议 |
| --- | --- | --- | --- |
| LOTSA / Moirai | [HF: Salesforce/lotsa_data](https://huggingface.co/datasets/Salesforce/lotsa_data)；[GitHub: uni2ts](https://github.com/SalesforceAIResearch/uni2ts) | 大量来源级目录/配置 | **B**，先做源级差集 |
| GIFT-Eval Pretrain | [HF: Salesforce/GiftEvalPretrain](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain)；[GitHub: gift-eval](https://github.com/SalesforceAIResearch/gift-eval) | 88 个来源数据集，仓库中可出现更多按年份或处理方式拆分的配置 | **C/B**，采用 GIFT-Eval 时升为 B |
| Time-300B | [HF: Maple728/Time-300B](https://huggingface.co/datasets/Maple728/Time-300B)；[GitHub: Time-MoE](https://github.com/Time-MoE/Time-MoE) | 领域目录、来源目录和大文件分片 | **B/C**，适合偏斜和 scaling 实验 |
| RMISC | [HF: nju-zhangsq/RMISC](https://huggingface.co/datasets/nju-zhangsq/RMISC)；[GitHub](https://github.com/zhangsq-nju/RMISC) | 分支版本 + 约 200 个来源级子数据集 | 推荐分支 **A**，全量 **B** |
| Chronos datasets | [HF: autogluon/chronos_datasets](https://huggingface.co/datasets/autogluon/chronos_datasets)；[extra](https://huggingface.co/datasets/autogluon/chronos_datasets_extra)；[GitHub](https://github.com/amazon-science/chronos-forecasting) | 原始来源、增强语料、合成语料和评测配置并存 | 分类型处理 |
| UTSD | [HF: thuml/UTSD](https://huggingface.co/datasets/thuml/UTSD)；[GitHub: OpenLTM](https://github.com/thuml/OpenLTM) | 1G、2G、4G、12G 四个规模版本 | 已纳入；保持版本记录 |
| Time Series PILE | [HF: AutonLab/Timeseries-PILE](https://huggingface.co/datasets/AutonLab/Timeseries-PILE)；[MOMENT GitHub](https://github.com/moment-timeseries-foundation-model/moment) | 任务和来源混合的聚合语料 | **C**，更适合跨任务分析 |
| Monash | [HF: Monash-University/monash_tsf](https://huggingface.co/datasets/Monash-University/monash_tsf)；[官方站](https://forecastingdata.org/) | 频率、来源和缺失值版本构成的配置 | 已纳入；作为稳定基准源 |
| Quito Corpus / QuitoBench | [Corpus](https://huggingface.co/datasets/hq-bench/quito-corpus)；[Bench](https://huggingface.co/datasets/hq-bench/quitobench)；[GitHub](https://github.com/alipay/quito) | `hour`、`min` 两个配置 | Corpus **A**；Bench **D** |

---

# 4. 各语料配置与选择建议

## 4.1 LOTSA / Moirai

### 4.1.1 权威源头

- 数据仓库：[Salesforce/lotsa_data](https://huggingface.co/datasets/Salesforce/lotsa_data)
- 官方代码：[SalesforceAIResearch/uni2ts](https://github.com/SalesforceAIResearch/uni2ts)
- 论文：[Unified Training of Universal Time Series Forecasting Transformers](https://arxiv.org/abs/2402.02592)

LOTSA 是 Moirai 的公开预训练归档，论文报告超过 270 亿 observations，覆盖九个领域。Hugging Face 仓库以来源级目录存放 Arrow 数据，目录名通常比论文中的领域标签更细。

### 4.1.2 配置与子数据集分组

LOTSA 的根目录数量较多，而且可能随仓库维护发生变化。对 TS3 更有意义的不是复制一份静态目录截图，而是把配置按原始来源和领域登记。当前可识别的主要类别如下。

| 领域 | 代表性目录或来源家族 | TS3 等级 | 说明 |
| --- | --- | --- | --- |
| 能源 | `australian_electricity_demand`、电力负荷、电价、太阳能、风电、建筑能耗类 | **A/B** | 优先保留现有语料中未覆盖的国家、频率和测量对象 |
| 交通 | `PEMS03`、其他 PEMS、`BEIJING_SUBWAY_30MIN`、`HZMETRO`、出租车和道路流量类 | **B** | 与 UTSD、Monash、Chronos 重叠概率高，必须做源级排除 |
| 气象与气候 | ERA5 年份目录、CMIP6 年份目录、WeatherBench/气象站类 | **B/C** | 数据量极大；适合后续 scaling，不适合当前全量处理 |
| 零售与销售 | 商店销售、产品需求、竞赛销售类 | **A/B** | 若能确认未被 Monash/Chronos 收录，可补充非物理过程 |
| 金融与经济 | 汇率、宏观经济、股票、银行和竞赛数据 | **A/B** | 高频金融数据需单独处理非平稳性和交易日历 |
| Web 与云服务 | Web traffic、服务器、请求量、云指标类 | **A** | 对 TS3 的结构覆盖通常比继续增加气象数据更有增量 |
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

| 配置家族 | 常见命名形式或代表来源 | TS3 等级 | 处理建议 |
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
| 多变量环境与工业 | 多传感器、环境、工业控制类 | **A/B** | 对 TS3 多变量扩展有价值 |

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

当 TS3 采用 GIFT-Eval 作为正式下游评测时，GIFT-Eval Pretrain 的主要优势是官方设计的非泄漏关系，此时可将其提升为 **B 级核心候选**。

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

| 领域 | 内容 | TS3 等级 | 建议 |
| --- | --- | --- | --- |
| `Nature` / 自然环境 | 气象、气候、地球与环境过程 | **B/C** | 占比极高，适合研究偏斜；不宜直接按原比例进入主池 |
| `Energy` | 负荷、发电、价格、设备与建筑能耗 | **A/B** | 优先选取未覆盖国家、设备和频率 |
| `Transport` | 道路、公共交通、轨迹和流量 | **B** | 与 UTSD/LOTSA 重叠核验后使用 |
| `Web` | 网站访问、云服务、请求量和系统指标 | **A** | 建议优先，通常能补充不同于物理传感器的结构 |
| `Finance` / `Economics` | 市场、汇率、宏观与业务指标 | **A/B** | 建议按来源分层抽样 |
| `Health` | 医疗、人口和生理数据 | **A/C** | 许可明确时优先 |
| `Sales` / `Retail` | 商品、门店和需求 | **A/B** | 对 TS3 结构多样性有价值 |
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

Time-300B 的 Nature 占比很高。TS3 可利用这一点设计两个独立实验：

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

| 分支或版本 | 含义 | TS3 等级 | 建议 |
| --- | --- | --- | --- |
| `recommended_corpus` | 项目方推荐用于训练和复现实验的语料组合 | **A** | TS3 首选 |
| `smaller_version` | 更小、便于验证管线的版本 | **A** | 先用于格式、速度和特征抽取测试 |
| `main` | 完整来源组织或主版本 | **B** | 仅按子数据集选择性下载 |
| `zipped_version` | 压缩归档版本 | **B/C** | 适合批量归档，不利于流式和选择性读取 |
| Parquet 转换分支 | Hugging Face 自动或项目提供的 Parquet 版本 | **A/B** | 优先用于结构扫描，但需记录转换 revision |

### 4.4.3 子数据集领域

RMISC 的约 200 个子数据集不应被压缩成几个 Hugging Face 配置名。应按原始数据源登记。官方描述覆盖的主要领域包括：

| 领域 | 典型内容 | TS3 等级 |
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

当前 TS3 应按以下顺序：

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

| 类型 | 代表配置 | TS3 等级 | 说明 |
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

TS3 已经纳入部分 Chronos，因此下一步不应继续以“Chronos 仓库”为来源单位，而应拆成：

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

| 配置/目录 | 近似规模 | TS3 等级 | 建议 |
| --- | ---: | --- | --- |
| `UTSD-1G` | 约 10 亿点 | **A** | 最适合快速复现和小规模对照 |
| `UTSD-2G` | 约 20 亿点 | **A/B** | 用于中等规模扩展 |
| `UTSD-4G` | 约 40 亿点 | **B** | 用于 scaling 研究 |
| `UTSD-12G` | 约 120 亿点 | **B** | 最大版本，处理前先检查与小版本的包含关系 |

这些版本大概率是层级或抽样版本，而不是四套完全独立来源。TS3 必须记录：

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

TS3 已经纳入 UTSD，因此：

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

**不建议**把 Time Series PILE 全量直接并入 TS3 当前预测主语料，原因是：

1. 与 Monash 和其他经典 benchmark 重叠明显；
2. 分类和异常检测数据的样本定义与 forecasting trajectory 不同；
3. 某些窗口化版本可能重复暴露同一原始轨迹。

更合理的用法：

| 用途 | 建议 |
| --- | --- |
| 纯 forecasting 主实验 | 只取未重复的预测来源，等级 **C** |
| 跨任务结构分析 | 保留分类、异常检测和预测来源，等级 **A/C** |
| 特征空间覆盖检查 | 很适合验证 TS3 六维结构是否跨任务稳定 |
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

### 4.8.4 TS3 建议

Monash 已纳入 TS3，应把它视为稳定来源注册表，而不是继续扩容对象。重点工作是：

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

| 仓库 | 配置 | 数据粒度 | TS3 等级 |
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
- QuitoBench 只用于验证 TS3 特征是否对应 trend、seasonality、forecastability 等结构 regime，不进入训练。

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

# 5. TS3 推荐新增路线

## 5.1 当前阶段：稳定增量

建议当前只增加能够明显改变来源或结构分布的数据。

| 顺序 | 数据 | 配置 | 原因 |
| ---: | --- | --- | --- |
| 1 | Quito Corpus | `hour` | 下载和处理成本较低；单一真实来源；可与 QuitoBench 对照 |
| 2 | Quito Corpus | `min` | 增加高频局部结构和长序列 |
| 3 | RMISC | `smaller_version` | 快速验证多变量入库和特征管线 |
| 4 | RMISC | `recommended_corpus` | 增加真实多变量、工业和 IoT 来源 |
| 5 | LOTSA | Web、工业、健康等未重复子集 | 补充与现有物理传感器不同的过程 |

## 5.2 第二阶段：中等规模扩展

```text
LOTSA 源级差集
-> RMISC main 中未覆盖来源
-> Time-300B Web / Industry / Health / Sales
-> GIFT-Eval Pretrain 中未重复且非评测的数据
```

该阶段应固定新增数据预算，例如：

```text
+1B
+3B
+10B observations
```

而不是无上限抓取。

## 5.3 第三阶段：大规模与偏斜实验

```text
Time-300B Nature
LOTSA ERA5 / CMIP6
GIFT-Eval Pretrain ERA5 / CMIP6
RMISC full
```

这些数据更适合回答：

- 结构调度能否在极端领域偏斜下优于随机采样；
- 增加同质数据规模是否仍然提高结构覆盖；
- 固定训练预算下，结构覆盖与原始 observations 的贡献如何分离。

## 5.4 独立实验分支

以下数据不进入“真实去重主语料”，而进入独立分支：

```text
Chronos TSMixup
Chronos KernelSynth
Time-300B Synthetic
其他合成或增强语料
```

建议目录：

```text
ts3_dataset/
├── real/
├── augmented/
├── synthetic/
├── evaluation/
└── manifests/
```

---

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

即使原始轨迹没有重复，同一长序列也可能产生大量高度重叠窗口。TS3 应分别报告：

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
- 小规模抽样计算 TS3 特征；
- 判断是否值得完整下载。

---

# 9. 最终建议

TS3 当前已有约 5.73B 去重数据，来自 Monash、UTSD 和部分 Chronos。下一阶段的目标不应是追求最大的公开数字，而应是增加下列三类增量：

1. **新来源**：当前语料没有覆盖的业务、工业、Web、健康和多变量过程；
2. **新变量结构**：保留真实多变量组，而不是只增加更多单变量切片；
3. **新结构分布**：能够填补当前 temporal structure space 稀疏区域的数据。

当前推荐顺序：

```text
Quito Corpus / hour
-> Quito Corpus / min
-> RMISC smaller_version
-> RMISC recommended_corpus
-> LOTSA 中未重复的 Web / Industry / Health 子集
-> Time-300B 特定领域
-> GIFT-Eval Pretrain 的源级差集
```

后续大规模实验顺序：

```text
LOTSA 大型气象与交通分片
-> Time-300B Nature
-> GIFT-Eval Pretrain ERA5 / CMIP6
-> RMISC full
```

独立合成/增强分支：

```text
Chronos TSMixup
Chronos KernelSynth
Time-300B Synthetic
```

最终应避免使用如下表述：

```text
“TS3 包含 X 亿独立真实 observations”
```

除非已经完成源级、轨迹级和派生级审计。更稳妥的报告方式是同时给出：

```text
raw observations
post-source-dedup observations
post-series-dedup observations
canonical windows
number of canonical sources
number of occupied structure grids
```

这会使 TS3 的数据规模、结构覆盖和去重主张可以被独立复核。
