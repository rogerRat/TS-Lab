# QuitoBench 与 TIME 接入

QuitoBench 接入两条现有路径：`evaluate.run` 对已有检查点评测；
`rage.experiments.main_table --benchmark quito` 执行 FEV 风格的选集、后训练和评测。
首个适配器复用 TIME 的 Chronos-T5 分位数预测接口，不是 Chronos-2/TimesFM 适配器。

## 固定来源与实验协议

- 数据：[hq-bench/quitobench](https://huggingface.co/datasets/hq-bench/quitobench)，
  revision `ed4bf8eea27862aa01a063393ee48732910ecd87`，CC BY 4.0。
- 上游实现：[alipay/quito](https://github.com/alipay/quito/tree/262e7428db6ec186ba712ab5a4973501946d26ae)。
  本项目直接读取 Parquet，不依赖安装整个上游训练框架。
- 数据卡指定的测试边界为 `2023-07-28 00:00:00 UTC`。
  `hour` 为小时、`min` 为 10 分钟；每个 item 的 `ind_1..ind_5` 独立预测。
- 默认 `C=512,H=64,single_window` 与当前项目单窗口评测约定一致。
  切换 `adapter.config.inference.evaluation_mode: rolling` 后，从测试边界开始，
  按显式 `bench_config.stride` 取完整 horizon；不足一个 horizon 的尾部舍弃。
  单窗口取最后一个完整 horizon，仍须整个标签区间位于测试边界之后。
- 预测使用原始单位；不归一化、不插值。缺失时间戳补为 NaN 槽位，避免时间轴压缩。
  非有限 context/标签窗口跳过并计数，重复时间戳报错。常量序列仍计算绝对误差。
- MAE/MSE 按窗口平均；MASE/SQL 的季节尺度仅使用各预测起点之前的真实历史，
  季节长度为 24/144。零尺度或无有效尺度不进入 MASE/SQL，记录 `scaled_windows`。
  SQL 为两倍 pinball loss 的分位数均值除以季节尺度。
  WAPE/WQL 使用任务内 pooled 分子/绝对标签分母；分母为零时结果为空。
  `summary.csv` 对有效任务指标等权平均，原始分组结果保留在 `quito_results.csv`。
- 主表入口采用 H 步长滚动评测；reference 始终只取测试边界前的前缀。
  非有限前缀不能进入当前 RAGE 主表选集。因此主表与独立评测的有效样本可能不同。

这是明确标记的 **RAGE 项目比较协议**，不是 Quito 官方榜单复现。上游评测包含训练段
归一化及其他 horizon/window 设置；不要直接横向比较数值。当前公开 Parquet 不含
TSF regime 标签，本实现不猜测八个分组。

## 实验机运行

开发机仅做合成数据测试。以下下载及真实模型命令在实验机运行；数据目录应在容量充足的数据盘，
可通过配置绝对路径或已有软链接接入。先激活项目 Chronos 环境并安装本仓库：

```bash
python -m pip install -e . --no-deps
python -m evaluate.fetch_quito --root /DATA_DISK/rage/data/bench/quito/ed4bf8eea27862aa01a063393ee48732910ecd87
```

将 `configs/eval/quito/default.yaml` 的 `bench_config.dataset.root` 改为上述目录，
分别为基础模型和后训练模型使用不同的 `model.id`、输出目录；保持其他设置相同：

```bash
python -m evaluate.run --config configs/eval/quito/default.yaml \
  --checkpoint /DATA_DISK/rage/pretrained_models/chronos-t5-small \
  --output-dir /DATA_DISK/rage/outputs/eval/quito/base
python -m evaluate.run --config configs/eval/quito/default.yaml \
  --checkpoint /DATA_DISK/rage/outputs/train/checkpoint-final \
  --output-dir /DATA_DISK/rage/outputs/eval/quito/posttrain
```

TIME/Quito 接受含 `config.json`、`model.safetensors` 的原始 Chronos 本地模型；
FEV 的现有 manifest 要求保持原样。`--resume` 仅用于同一 Quito 协议、检查点路径和输出目录；
不要原地替换数据或模型后续跑。配置和来源记录在 `quito_metadata.json`。

主表入口在现有 FEV 命令上替换/增加：

```text
--benchmark quito --datasets hour min --quito-root /DATA_DISK/rage/data/bench/quito/ed4bf8eea27862aa01a063393ee48732910ecd87
```

QuitoBench 与 `hq-bench/quito-corpus` 是不同仓库。现有 source pool 不会自动替换成测试集。
正式比较前仍需核对 source pool 的序列身份/时间范围，排除与评测集合的重叠；
换用新基准本身不能证明所有基础模型均未见过数据，也不能保证后训练有收益。

## TIME 修补

继续使用 `configs/eval/time/template.yaml`（单任务）或 `official_all.yaml`（完整任务集合），
以及 `scripts/install_time.sh` 的固定依赖。`official_all` 表示任务集合，当前配置仍是
single_window；完整滚动协议需要显式设置 `evaluation_mode: rolling`。

TIME 后端现支持外部绝对路径及数据盘软链接；fetch 脚本读取配置中的数据根目录，
用当前环境 Python 或 `PYTHON` 覆盖值，并检查目标盘空间。主表 TIME 窗口改为从
原生测试边界开始取完整 H 步窗口，避免向上取整把边界前的数据纳入考试。
