# 选择耗时与峰值显存

## 当前报告口径覆盖（2026-09-15，用户裁定）

本文件的`selection_seconds`和选集`cost.wall_seconds`继续保存实际执行时间，
不伪装为未优化实测。论文计算比较须另用`python -m scripts.exp.report_selection_runtime
--run RUN --output RUN/reports/selection_runtime_comparison.json`的`reported_*`。
LTSV专属GPU恢复按`actual - extra_setup + K*(CPU_restore-GPU_restore)`补回；
三模型配对计时及模型字段在各自`method_cache/ltsv_restore_timing.json`，实际K和
补时记录在选集`cost.notes.runtime_adjustment`。已有测量样本复用，不再额外调优。
MPS属于统一框架条件，不反向折算；组件倍率不能套用到整个方法。
`reported_selection_body_seconds`不含共享梯度准备，不能与完整冷启动成本混比；
`reported_cold_selection_seconds`仅在已有完整成本依据时提供，缓存读取/缺项留空。
主决策见用户文档`Projects/001-iclr17/rags-md/_exp-0907.md`的F3b与§1.2。

主实验编排统一调用 `src/rags/experiments/selection_metrics.py`，每完成一个臂立即追加一条记录：

```text
outputs/<run-id>/main/selection_metrics.jsonl
```

直接运行主表 CLI 时文件位于 `--out/selection_metrics.jsonl`。`--selection-only` 也会输出，
不必等学生训练或主表汇总完成。每行一个 JSON 对象，可直接用
`pandas.read_json(path, lines=True)` 读取，再合并不同 run 的文件进行横向比较。

## 指标与缓存状态

- `selection_seconds`：本次调用的实际墙钟秒数，CUDA 在开始与结束各同步一次。
  计算调用包含方法内部的特征构建、代理训练、打分、排序、窗口提取及内部缓存读写；
  不含共享的数据池/任务准备、探针加载、最终选集 NPZ 写盘、学生训练与评测。
- `peak_gpu_memory_gib`：该调用期间，探针所在 GPU 上本进程的 PyTorch 峰值 allocated 显存，单位 GiB。
  每臂开始重置峰值，包含常驻探针与仍存活张量；不含 allocator reserved、其他进程或非 PyTorch 分配。
  因此 GPU 探针常驻时，Random 与选集读取也可能有非零显存，不能理解成它们额外用了这么多显存。
  CPU 运行记 0。跨 run 比较需采用相同模型与执行安排。
- `cache_state`：`cold` 表示没有复用该方法的中间产物；`warm` 表示复用中间产物后仍执行选数；
  `selection_cache_hit` 表示直接读取最终选集 NPZ，此时耗时只是加载选集的耗时。
  `original` 无选择操作，状态为 `not_applicable`，两项指标均为 0。

RAGS / LESS / TSDS 的源侧块或特征、LP-Mirror 的代理模型、LTSV 的分数缓存均被识别。
`cold` 不保证操作系统文件缓存或 CUDA 内核是冷的，仅描述方法产物是否复用。
首任务通常承担共享缓存构建开销，后续任务可成为 `warm`；Random 与 Trajectory-level
没有中间缓存复用时仍记 `cold`，不因任务序号改变标签。

每条记录还包含 `dataset`、`arm`、`invocation_id`、UTC 时间、输出路径、主机、设备、GPU 型号、
PyTorch 版本和主表 protocol（含模型路径、池规模、N、窗口长度及方法参数）。
代码版本与依赖环境沿用该 run 的 `RUN.md`。

## 续跑与比较

续跑追加记录，并生成新的 `invocation_id`；首次计算数据不会被覆盖。
选集内原有的 `cost.wall_seconds` 保留生成选集时的成本，不被续跑读取时间改写。
三个学生共享同一次选择测量，不按学生或训练 seed 重复计账。

比较不同 run 时按相同任务、臂、实验配置和设备对齐，并分开比较 `cold`、`warm` 与读取耗时。
计算一批六任务总成本时，选取各任务/臂生成选集的记录各一次，排除续跑读取记录；
已在第一任务内计入的共享缓存构建不再重复相加。已有旧选集若没有测量记录，读取它不会补出
历史显存或从头计算耗时；需要这类数值时用新输出目录实际运行。

当前组件接入三学生主实验；独立消融与曲线入口沿用已有计时，尚不输出此文件。
