# FEV 评测机制

本文说明项目当前集成的 FEV（Forecasting Evaluation）评测如何组织数据、生成评测窗口，以及 `id`、`target` 和 `num_windows` 之间的关系。

## 1. Task 与数据集

一个 FEV evaluation task 对应一个数据集配置（`dataset_config`）。例如：

```yaml
dataset_path: autogluon/fev_datasets
dataset_config: boomlet_1282
target: [target_0, target_1, target_2]
horizon: 60
num_windows: 20
```

task 还会声明预测 horizon、窗口数量、评估指标以及要评估的 target。不同 task 可以使用不同的数据集配置、target 集合和 `num_windows`。一个 evaluation run 包含多个 task；每个 task 独立加载自己的数据集并计算自己的评测结果。

## 2. `id` 与 `target`

FEV 数据集通常包含以下类型的字段：

```text
id          序列实体或序列组标识
timestamp   该序列的时间戳数组
target_*    一个或多个目标序列数组
```

严格来说，一条具体的单变量时间序列由 `(id, target)` 共同确定：

```text
(id=A, target_0) -> 一条具体时间序列
(id=A, target_1) -> 另一条具体时间序列
```

同一个 `id` 下的多个 target 共享时间轴，因此原始数据可以是多变量时间序列。task 中的 `target` 列表指定本次评测使用哪些目标序列；未被列出的 target 不参与该 task。

当前 Chronos adapter 会将选中的多变量输入转换为 univariate 输入，再分别进行预测，见 `adapters/chronos/eval/fev.py` 中对 `fev.convert_input_data(..., as_univariate=True)` 的调用。

## 3. `num_windows` 的含义

`num_windows` 是 task 级别的评测窗口数，不是为每个 target 单独动态计算的窗口数。

例如：

```yaml
target: [target_0, target_1, target_2]
num_windows: 20
```

表示该 task 生成一组 20 个 forecast cutoff/window。每个被选中的 `(id, target)` 序列都应在这组共同窗口上完成预测：

```text
(id, target_0) -> 20 个窗口
(id, target_1) -> 20 个窗口
(id, target_2) -> 20 个窗口
```

这些窗口由 task 统一确定；每个 target 不会依据自己的长度生成另一组窗口。

如果一个 task 有 `I` 个 `id`、`T` 个 target、`W` 个窗口，则理想的 forecast 单元数量为：

```text
I × T × W
```

每个 forecast 单元预测 `horizon` 个未来时间点。

## 4. 最小评测单元

FEV 的最小预测评测单元可以表示为：

```text
(id, target, window/cutoff)
```

它包含 cutoff 之前的历史 context、cutoff 之后长度为 `horizon` 的真实未来值、模型预测，以及对预测和真实值计算的评估指标。

Chronos adapter 的执行顺序是：

```text
task.iter_windows()
    -> 生成 task 的评测窗口
    -> 将窗口中的选定 target 转为 univariate 序列
    -> 对每条序列预测 horizon 个值
    -> 汇总每个窗口的预测结果
```

模型推理参数中的 `num_samples` 是同一个 forecast 单元生成的随机预测路径数量，用于估计分位数；它不是额外的 FEV 评测窗口数。

## 5. 轨迹长度与 FEV 评测权重

在当前 FEV task 机制下，评测窗口数由 `num_windows` 和 task 的时间参数决定。只要某条 `(id, target)` 序列满足 task 对 context、horizon 和 cutoff 的数据要求，它就应当提供相同数量的 `num_windows` 个评测样本。

FEV 不采用“长 trajectory 根据长度生成更多评测窗口、短 trajectory 根据长度生成更少评测窗口”的规则。在当前评测设计中，trajectory length 不作为评测样本数量的权重，也不使用训练阶段的 source `base_mass`、OT/KDE 分布或采样结果。

长度的作用仅限于数据可用性：如果序列短到无法支持 task 要求的窗口，task/window 构造或评测校验可能失败；它不会作为正常路径下的连续权重，让长序列贡献更多窗口。

因此，针对 FEV 可以采用以下结论：

> 每个被 task 声明的具体 `(id, target)` 序列，都应在同一组 `num_windows` 上参与评测；在满足数据长度要求的前提下，trajectory length 与其贡献的评测样本数量和评测权重无关。

这与训练阶段按 trajectory length 构造 source base mass 的逻辑相互独立，不应混用。
