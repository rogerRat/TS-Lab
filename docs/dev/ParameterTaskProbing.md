# 参数探测与共享任务探测

本页描述参数与任务探测协议。当前运行快照在 `/root/TS3-probe`，执行状态见
[本轮执行记录](../experiments/probe_execution_20260914.md)；主仓库文档不意味着源码已在主分支合并。
使用当前三模型：Chronos **T5 Base**、TimesFM **2.5**、TimeMoE **200M**。
入口复用 `full_pool_exploration` 的原生训练、重采样与开发验证，
不会启动历史 `main_table.py`，也不读取最终考试数组。

## 官方参数参考（2026-09-14 核查）

| 模型 | 首试 LR | 有效 batch / micro batch | AdamW decay / betas | 学习率调度 |
| --- | --- | --- | --- | --- |
| Chronos T5 Base | 1e-3 | 32 / 32 | 0 / (0.9, 0.999) | linear → 0 |
| TimesFM 2.5 | 1e-4 | 32 / 32 | 0.01 / (0.9, 0.999) | cosine → 0 |
| TimeMoE 200M | 1e-4 | 64 / 16 | 0.1 / (0.9, 0.95) | cosine → 初始 LR 的 0.5 |

三组 eps=1e-8、clip norm=1、warmup=0。TimeMoE 用四个 micro batch 累积为一次更新；
其 bias/norm 参数不做 weight decay，保持 Trainer 风格的参数分组。
不同模型不强制同一 batch/LR/优化器设置。同一模型所有任务探测使用其固定设置。

来源和适用边界：

- [Chronos 原始 T5 微调示例](https://github.com/amazon-science/chronos-forecasting/blob/4dbf163c2734c089cdf7da2b86fde48862ff9c6f/scripts/README.md)
  演示 Small 以 1e-3 微调 1000 steps；
  [Base 配置](https://github.com/amazon-science/chronos-forecasting/blob/4dbf163c2734c089cdf7da2b86fde48862ff9c6f/scripts/training/configs/chronos-t5-base.yaml)
  使用相同 LR、C512/H64、batch32 和 linear。此处加载已有 Base，不随机初始化。
  [上游脚本](https://github.com/amazon-science/chronos-forecasting/blob/4dbf163c2734c089cdf7da2b86fde48862ff9c6f/scripts/training/train.py)
  没有显式传 weight_decay，实际继承 Trainer 默认 0；不要与论文预训练的 0.01 混淆。
- [TimesFM 2.5 官方示例](https://github.com/google-research/timesfm/blob/8cb0628371af142e16b8c232cc9fbf667ffb12f9/timesfm-forecasting/examples/finetuning/finetune_lora.py)
  是 **LoRA** 零售数据示例：rank4、alpha8、dropout0.05、all-linear，10 epochs，5000 预采样窗口。
  本项目保持现有**全参数**实验方法，只把其 LR/AdamW/cosine/clip 设置作为起点。
  没有找到可直接等同本项目全参数任务的统一官方推荐配方；产物明确写 `reference_scope`，
  不将参数迁移实验称作官方 LoRA 复现。TimesFM 3.0 的推理接口不参与本次训练探测。
- [TimeMoE 官方 CLI](https://github.com/Time-MoE/Time-MoE/blob/915bfda4c78a544d62a2bec6ab22948423059236/main.py)
  默认 1 epoch、LR1e-4、minimum LR5e-5、有效 batch64、micro16、decay0.1、beta2=0.95。
  [README](https://github.com/Time-MoE/Time-MoE/blob/915bfda4c78a544d62a2bec6ab22948423059236/README.md)
  对小数据建议 stride1。本项目使用合法滑窗随机抽样、官方原生多 horizon 损失。

这些是有出处的起点，不是已证实最优超参数。模型代码/权重版本继续使用仓库已有固定版本；
最新上游参考的 commit 另存 `official_references.json`，不会自动升级模型依赖。
精度以执行版本的 `precision.POLICY` 为准；本次远端为 `bf16-model-gradient-adamw-v2`，
模型、梯度与AdamW动量均为BF16，无FP32主权重。优化器使用 `foreach=False`。

## 参数探测

入口：`python -m rags.experiments.probing`，配置 `configs/rags/probing.json`。

1. 固定 `parameter_tasks` 和 seeds；三个模型分别探测，不跨模型共享 LR 或 Epoch。
2. 每个模型按 `learning_rates` 顺序运行，第一项必须是上述官方参考值；
   可独立编辑后面的学习率候选，覆盖更高/更低 LR。示例含三个额外候选。
3. 每个 LR 从同一基础模型训练，先观察 `coarse_epochs: [1, 2]`。
   同一 LR 曲线保持连续优化器；每个任务都有 epoch0 基础分数。
4. 按“每任务相对改善，再对任务、种子等权平均”选择较好的 LR。
   在该LR上最多追加 **4 次查找**（`max_search_steps`），扩展和二分共用预算。
   若E2优于E1，查E3；继续改善就查E4，最多一直到E6。每次扩展消耗一次预算。
   若新点不优于前一点，进入相邻区间二分；例如E3差于E2，下一点为E2.5。
   E1优于E2时直接查E1.5，再根据中点和两端的改善值保留较有希望的一半。
   持平按没有继续改善处理。预算按每模型最佳粗LR计，不是每个LR各4次。
5. 中点可能是 1.25、1.125、1.3125 等，runtime 已允许任意正有限 Epoch。
   每次从基础权重重跑，绝不把 E2 的权重当作 E1.5 权重。相同 seed 使用相同抽样前缀。

Epoch 仍定义为 N 次有放回曝光，N=`sum(floor(prevalidation_length/(C+H)))`，
实际步数 `ceil(E*N/effective_batch)`。每次记录 requested/actual Epoch 与真实曝光。
微小池中，中点若与端点对应同一个 optimizer step，会停止细分并说明原因。
这是有限局部搜索，不保证非单峰曲线的全局最优；到预算上界仍改善时报告预算耗尽，
不将边界最优称作收敛，不自行增加次数。

所有候选的**学习率衰减总长度事先固定为粗搜索最大Epoch加追加预算**，本次为6。
E1/E2/E3和所有中点都是同一条六轮衰减轨迹的停止点，避免E2学习率已经耗尽后再扩展。
追加点从基础权重按同seed/采样前缀重跑到指定Epoch，保留失败和旧检查点。
冻结产物保留 `schedule_epochs`。`search_trace`逐次记录extend/bisect与预算。
`epoch_search_policy.json`区分新旧协议，旧的两轮衰减结果不能混入新目录或继续充作新结果。

## 任务探测

三份模型参数结果齐全后才进入任务阶段。默认 `task_probe_model: timemoe-200m`；
也可设 `best_relative_gain`，使用参数探测阶段平均相对改善最大的模型。
若该模型在参数阶段没有正收益，停止任务冻结并提示扩展参数探测，避免用无效参数强行挑任务。

对每个候选任务，使用选定模型的相同 LR、Epoch、优化器、batch、seeds 和窗口设置。
仅改变 `datasets: [task]`，在该任务的安全历史前缀上后训练，再评估其开发验证段。
每次都恢复基础模型；不同任务不串行累积权重。

相对改善为 `(MASE_untrained - MASE_trained) / MASE_untrained`，数值0.1表示10%。
先对各 seed 计算配对改善，再平均。零/无效 baseline 不排名，退化任务不补入列表。
按改善降序取正收益前 `top_k` 个；若正收益任务不足，只保留实际数量并记录 requested_top_k。
若没有正收益，保存排名但不生成冻结清单。不同任务 N 不同，步数可能不同，但 Epoch 参数固定。

`shared_selection.json` 只有一份全局 `datasets`，三模型各有自己的训练设置。
不接受配置中的 per-model datasets，也不生成 per-model 任务排名/任务清单。
这个文件是开发选择结果，尚不是原正式六任务矩阵的配置：原正式入口仍有其任务矩阵校验、
epsilon/temperature 证据及 source-only 训练要求，不会被探测脚本悄悄覆盖。

## 数据与运行

先编辑 `configs/rags/probe_inputs.json`，填入实验机实际数据盘路径。
本次候选扩展为8个，参数任务固定为Finland_Traffic/15T、CPHL/H和EWELD_Load/15T。
候选另含NEM需求、NEM价格、Australia_Solar/H、Quito hour/min。详见
[候选范围与数据核查](ProbeScopeV2.md)。不纳入ETT；FEV旧任务退出本次挑选池。
这几个ID是数据集/频率/指定变量组的集合任务，内部仍有多个实体和验证窗口，
不是“最小单序列单窗口”。统一H64是本项目开发探测口径，不冒称TIME原生全部预测时长。

```bash
python -m rags.experiments.prepare_probe_inputs \
  --config configs/rags/probe_inputs.json --out /DATA_DISK/ts3/outputs/probe-inputs
```

Quito 按固定UTC cutoff截断；TIME 按原生 test_length 预留整个考试尾部；
FEV 读取现有 `__reference.npz` 及其 manifest。导出只包含 pre-exam reference。
再从 reference 尾部预留完整验证窗口（包括 context），更早的前缀用于训练，
三部分在时间上分离。不读取 `__exam.npz`，不以最终考试结果调参/选任务。
非有限或不足长度的前缀明确计数、记录原因并跳过；季节长度记录在 manifest 并传入开发 MASE。
TIME任务可用 `variate_names`精确指定预测变量，用`require_native_univariate`要求原始target维度为1。
manifest记录原始维度，拆通道后不会被标作原生单变量。无插值、无跨缺口拼接。

编辑 `configs/rags/probing.json` 的 inputs、模型环境 python 路径及可选 model_path。
候选任务、参数任务、LR列表、种子数和 top_k 均在配置中明确声明。
默认三个 seed 增加稳健性；如先只用 seed0，后续复核要使用新输出目录，不能混进旧结果。

```bash
# 仅打印计划；默认行为，不加载模型或下载数据。
python -m rags.experiments.probing --config configs/rags/probing.json \
  --out /DATA_DISK/ts3/outputs/probe-v1 --stage plan

# 在实验机，依次使用各自环境完成三模型参数探测，再运行单模型任务探测。
python -m rags.experiments.probing --config configs/rags/probing.json \
  --out /DATA_DISK/ts3/outputs/probe-v1 --stage all
```

也可在对应模型环境单独运行 `--stage parameters --student chronos-base`（其余两模型同理），
完成后以选定任务探测模型的环境运行 `--stage tasks`。`all` 默认串行，退出码失败即停止；
相同配置/目录重入会复用已完成 trial 并从未完成 trial 的 AdamW/RNG 状态续跑。
改变配置或官方参考设置必须使用新目录。默认保留断点，不删除模型权重。

主要结果：`parameters/<model>/result.json`、各 trial 的 `original.json`、`curve.json`、
`sampling.json`、`validation_boundaries.json`；`task_probe/result.json` 和 `shared_selection.json`。
任务域后训练的可改善性不等同于 source-only 数据选择收益，也不证明基准无预训练重叠；
冻结后的最终模型比较仍需在保留的考试段进行。

## 测试范围

CPU合成模型/合成曲线覆盖：官方LR先试、三模型独立选择、1→2→1.5→1.25、
四次上限、相对改善排名、单模型固定参数、共享任务清单、错误/缺失分数拒绝、
数据时间隔离、分数Epoch执行、micro batch累积、调度器与AdamW/RNG精确恢复。
正式权重和GPU效果不属于前置代码测试通过的含义。

初次前置开发验收（历史）：**64 passed，2 skipped**（两项需要 CUDA），并通过 `git diff --check`
及变更 Python 文件编译检查。覆盖的测试文件：`test_probing.py`、`test_probe_runtime.py`、
`test_online_calibration.py`、`test_training_resume.py`、`test_full_pool_exploration.py`、
`test_full_pool_models.py`、`test_full_pool_worker.py`、`test_resampling.py`、`test_evaluate_quito.py`。
本地测试需从 `tests/` 运行并设置 `--rootdir=. --confcutdir=.`，避免 pytest 扫描
当前仓库指向不可访问历史存储的 `archive` 链接。

2026-09-14 v2相关远端套件38 passed；新增总候选20项上限后，probing套件18 passed。
两组有重叠，不将数字相加。E1/E2后的追加搜索与候选范围均已获用户批准。
