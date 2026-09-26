# 三学生主表：exp 运行说明（2026-09-11）

> 历史归档：保留当时的设计、操作或实验证据，不作为当前执行指令。当前入口见 `docs/README.md`。

依据 `001-iclr17/rags-md/_exp-0907.md` 拍板 9，以及 `_models-integration-0911.md` 的最终替换决定。
dev 仅开发/测试；下列正式运行命令只在用户的 exp 容器执行。

0912 最新协议见 [点预算与整单位交付](PointBudgetDelivery0912.md)：RAGS 整段、
trajectory_level/LESS 整轨迹、TSDS 整重复副本交付；训练时在交付区域全部合法起点采窗。
主环境需安装 `faiss-cpu==1.12.0`。全部旧固定窗协议 run 不兼容，更新后开新 run。

0912 执行优化与验收见 [SelectionExecution](SelectionExecution.md)：外部方法仅接入公共
批处理/窗口/预测设施，RAGS 另优化 CRB 调度；默认启用，不改变正式实验预算与训练 batch。
未经用户要求不要中断在途实验。本次用户已要求 Push 后通知 tmux zcode，
由其停止并清理当前错误批、拉取新代码后新建 run；只处理已确认的该批目录与进程。

## 当前实验

| 学生 | 环境 | 损失 / 预测读出 |
|---|---|---|
| Chronos-T5-small | ts3 | 原生分桶 CE / 历史自回归 bin CDF 中位数 |
| TimesFM 2.5 | ts3-timesfm | Transformers 原生 loss / `mean_predictions` |
| Time-MoE-200M | ts3-timemoe | 官方多 horizon Huber + router 辅助损失 / 自回归连续值 |

六任务依次进行。每任务七个选择臂各选一次（Chronos-small 512/64 探针），三学生共享这些
`selection/*.npz`，加各自 original 参照，得到八臂。学生之间可以并行，任务之间有完成屏障。
所有学生 FP32 全参数、AdamW lr=1e-5 / wd=.01 / batch=32，无裁剪、调度或 warmup；
四个 exposure epochs，累计 `ceil(epoch*N/32)` 步保存一次，保存全部检查点并逐个评测，最终轮作为主结果。

只用 MASE 比较模型和方法。尺度使用各预测时点之前的完整历史，与 FEV `y_past` 一致，
不含该时点之后的考卷值；先序列内平均，再数据集内平均，跨任务等权。零尺度另列 MAE，不制造极大 MASE。
旧版整条序列尺度包含了考试未来值，因此旧结果不能直接混入这版主表。

Time-MoE 使用选中的每个 576 点窗口作为一条独立训练序列：官方 zero-normalization，
575 个移位输入/标签对，所有原生 horizon heads 和 router loss 保留，不拼接、不读取窗口外的点。
这是原生自回归训练适配，并非仅对最后 64 点计算损失；评测仍只提供 512 点历史并预测 64 点。
训练标准差采用官方 zero_scaler 的 population std，预测采用官方示例的 history sample std。

直接导入 [Time-MoE 官方仓库](https://github.com/Time-MoE/Time-MoE/tree/67460eb4af69eeed5fa86e591846cb0ae0744c26)，
不复制或重写模型/损失。固定源码 `67460eb4` + 原版 Transformers 4.40.1；
尝试新版源码 `915bfda4` + Transformers 4.57.6 时，真实反传触发了 `loss_function`
属性冲突（错误进入 `ForCausalLMLoss`），因此不使用这条新版组合，也不注入补丁改变官方模型。
[权重](https://huggingface.co/Maple728/TimeMoE-200M)固定 `794591bf`。
完整源码/模型版本在 `requ/model_integrations.lock`。

Sundial、TimesFM 3.0 适配器保留，但不在学生列表中。前者无已采用的官方微调配方，后者权重为非商业许可；
此次不接 Moirai，不做长上下文选择消融，也不自动重标训练超参数。

## exp 安装与启动

从仓库根目录执行：

```bash
git fetch origin
git switch rags
git pull --ff-only origin rags
bash _install.sh
conda activate ts3
```

安装器准备 `ts3`、`ts3-timesfm`、`ts3-timemoe`、`ts3-sundial`、`time`
和固定上游源码，不下载正式数据。完整选项见 [安装说明](../../../requ/INSTALL.md)。
模型权重可先下载到指定目录；TimesFM / Time-MoE 不设本地路径时会自动加载固定 HF revision。
Chronos 探针需本地模型目录（默认 `cache/pretrained_models/amazon/chronos-t5-small`）。

在 `configs/rags/run.yaml` 中填实际路径；例如下列字段替换为 exp 的真实目录：

```yaml
source_load: /data/source/load
reference_root: /data/reference/fev
model_path: /models/chronos-t5-small
student_model_paths:
  - timesfm-2.5=/models/timesfm-2.5
  - timemoe-200m=/models/TimeMoE-200M
student_parallelism: 3
```

`student_model_paths` 可省略，使用已固定的 HF 模型。
`student_environments` 可覆盖环境，例如 `timemoe-200m=/path/to/python`；通常无需设置。
先确认资源，再启动主表：

```bash
python -m scripts.exp.deploy --config configs/rags/run.yaml --out-root /large-disk/rags
python -m scripts.exp.rags run --config configs/rags/run.yaml --steps main --out-root /large-disk/rags
```

不指定 `--steps main` 时按 YAML 继续机制、消融和曲线；这些后续实验仍使用冻结的 Chronos 探针/学生，
不是自动把所有附录扩成三个模型。正式主表配置不适合在 dev 上启动。

**存储量已更新：** 三学生共 1,512 份 FP32 检查点约 **1,371.8 GiB（1.34 TiB）**，
不含数据、选择缓存、附录与预训练模型。Time-MoE-200M 是激活参数口径，总参数 453,196,800；
TimesFM 总参数 231,289,280。预检按总参数估计，不能再用单 Chronos 的约 86 GiB 估算。
没有自动删检查点、压精度或关机。

## 输出与续跑

- `main/selection_metrics.jsonl`：统一记录各臂本次选择耗时、峰值显存及从头计算/中间缓存/选集读取状态，
  每次调用追加，支持跨 run 比较；口径见 [选择资源记录](../../dev/SelectionMetrics.md)。
- `RUN.md`：启动代码版本、环境、配置、依赖锁哈希（只在新批记录一次）。
- `config.yaml`：本批配置；`main/tasks/`：每任务共享考试数据及选集路径/选择成本。
- `main/students/<模型>/cells/`、`checkpoints/`：按模型独立的结果和四轮权重。
- 各模型 `checkpoint_curve.{csv,json,html}`：MASE—epoch；`cost_curve.html`：模型训练时间—MASE。
- `main/summary.json`、`student_comparison.html`：逐模型结果、三种子波动、按任务 bootstrap 区间、
  相对 Random/original 的变化与平均名次。选择成本只在共享任务记录一次，训练与评测时间按模型单列。

中断后用原目录续跑：

```bash
python -m scripts.exp.rags run --resume /large-disk/rags/<run-id> --steps main
```

复用已完成的选择和结果格；中途未完成的格从预训练状态重跑，不恢复半途优化器。
若改实验设置则使用新批目录，勿混用旧缓存。没有每格/每轮/每文件重复哈希或协议比对。
也可用 `python -m scripts.exp.model_train --model timemoe-200m --task-manifest <task.json> --output <student-dir>`
在对应环境单独跑一个学生。

## 开发验证

`tests/test_multistudent.py` 覆盖七臂一次选择→三学生八臂结果、续跑、原始模型恢复、末轮读数、
共享成本、模型分面汇总和尺度未来隔离。`tests/test_native_student_models.py` 在两个独立环境
使用官方小型随机模型：512/64、真实原生 loss 反传、`from_pretrained` 读入、四检查点保存/评测、
新进程运行生产 CLI。Time-MoE 输入/标签/mask 与官方 dataset 逐值对照。
测试产物：`outputs/native-students-0911/{timesfm,timemoe}/`。这是开发集成验证，不是预训练大模型正式性能实验。

全量核心回归：212 passed、5 skipped、6 failed；六项失败与修改前一致（两项 FEV 数据、两项 TIME checkout 缺失，
两项既有临时配置路径约束）。两个模型环境各自的官方模型/生产 CLI 测试通过；三环境 `pip check` 和安装验证通过。
