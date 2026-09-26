# 数据选择执行优化（2026-09-12）

> 历史归档：保留当时的设计、操作或实验证据，不作为当前执行指令。当前入口见 `docs/README.md`。

本文记录已提交的 `c063727` 执行优化。随后用户批准了
[LESS/TSDS 轨迹级重设计](TrajectoryBaselines0912.md)，改变了这两臂的样本单位与计算量；
下文 LESS/TSDS 的耗时是旧窗口级设计的历史测量，不是当前轨迹级实现的性能。
当前 `baselines` 执行奇偶基准只对仍可比的 LP-Mirror/LTSV 运行，避免把方法变更当成等价提速。

本批只优化实现执行，不缩减候选池、参考集或计算预算，不改变优化器更新次数、
损失、评分、排序/抽样规则与精度。外部基线的 `algorithms.py` 完全不改。
各方法使用同一套公共执行设施，不保证不同计算负载得到相同的加速倍数。

## 改了什么

- 所有臂共享窗口索引和参考采样实现：先选原来的全局行号，再提取窗口，避免先展开全部窗口。
  随机种子、行号与返回顺序不变；连续存储加批量 gather，仍严格遵守轨迹边界。
- 公共 Chronos 引擎按可用显存选择逐样本梯度/推理批次。梯度 OOM 时减半并记住上限，
  不再每次先 OOM 再整批串行。训练优化器的 batch 不受此设置影响。
- 公共 Count-Sketch 在较大 CUDA 批次使用融合核，仍是原来的符号、哈希和 FP32 求和。
  小批次保留 Torch 路径；梯度前向关闭无用 KV cache。
- 公共预测后端使用官方 T5 forward 与官方静态 KV cache，将固定长度 greedy 解码捕获为
  CUDA graph。每次 replay 都重新计算 encoder 和全部 decoder 步骤，读取当前权重；
  **没有跨 Adam 更新复用预测、encoder 输出或 KV 内容**。固定历史可只 tokenize 一次。
  LTSV 及 Chronos 检查点评测共用该设施，LTSV 的逐块 Adam/裁剪/参考 MSE/得分公式不变。
- 主表全部臂的选择调用统一限制 CPU Torch/BLAS 线程，避免小算子过度开线程；调用后恢复。
- RAGS 专用：CRB 风险优先级改为两个堆，保持原始风险比较及左端点 tie-break；
  多轨迹共享梯度批次，复用轨迹映射；源端分组处理并及时释放 anchors。
  同一主表进程的六任务直接复用源 CRB 内存结果，参考端也跨轨迹批处理。
  匹配后先按源块合并最大余弦，再展开窗口（原有语义不变）。

没有增加逐文件 hash、正式任务自动重启或某个外部基线专属的近似/预算削减。
现有缓存仍按原规则工作；没有新增块级断点，未完成的臂重启仍可能重算。

## 默认值与测量

`configs/rags/run.yaml` 的主表参数：

```yaml
selection_gradient_batch: 0
selection_inference_batch: 0
selection_memory_fraction: 0.7
selection_cpu_threads: 4
selection_generation: cuda_graph
```

两个 batch 的 0 表示自动；memory fraction 是估算批次的可用显存预算比例，不是显存硬配额。
梯度批次最大 128、推理最大 1024；最多用满显存不一定最快。可显式设置批次做设备侧小测。
`selection_generation: eager` 可回到官方 Python `generate`。
非 T5、CPU 或不支持的生成约束自动走官方生成。CUDA graph 依赖锁定的主环境
Torch 2.7 / Transformers 4.57；不要求 Time-MoE 的独立环境升级依赖。
这些 CLI 参数目前由主表接收；其他诊断入口仍用引擎默认值。

图捕获的首次开销计入选择耗时。选集缓存读取不是“重新计算更快”：对比时保持
`cold` / `warm` / `selection_cache_hit` 分开，继续使用每个 run 的
`main/selection_metrics.jsonl`，详见 [测量边界](../../dev/SelectionMetrics.md)。
自动批次、GPU 型号、并行负载和 FP32 求和次序会影响实际结果的末位，
数学语义一致不等于任意设备下逐 bit 一致。接近并列的离散排序仍应在正式实验中留意。

## 可复现的小规模验收

入口 `python -m scripts.exp.bench_selection_execution --suite <套件> --out <TMP输出.json>`。
套件为 `cpu`、`rags`、`projection`、`greedy`、`baselines`；后两者可加 `--full-size`。
对照通过只读 `git show cde56ec:<文件>` 加载上一交付版本，需保留该 Git 历史。
这些是合成数据、随机初始化模型，不下载正式权重，也不替代论文实验。

RTX 2080 Ti / 11 GiB 的开发测试：

- CPU CRB 四种配置：约 5.8–10.7 倍，anchors、块方向及计数一致。
- RAGS 完整选择（合成确定性梯度场）：约 2.2 倍，选集、分数、计数一致；
  这不是带 Chronos 反传的正式整臂加速倍数。
- 46M 参数、512/64 的 greedy：稳定调用约 0.819 → 0.266 秒，约 3.08 倍；
  首次图捕获约 2.75 秒，输入/权重改变后 token 仍与官方生成一致。
- Count-Sketch 大批次微测约 1.15–1.41 倍；小批次未加速，因此不启用融合路径。
- 四个外部基线用相同模型和窗口完整重算，分别校验选集 ID、窗口和逻辑工作量；
  基准另记录首次调用与两次无选集缓存重算的中位耗时，避免把初始化误报为稳定加速。

46M 参数、1,154 个池窗口的小规模完整选择稳定吞吐：

| 臂 | 旧版秒数 | 新版秒数 | 加速 |
|---|---:|---:|---:|
| LESS | 7.963 | 2.804 | 2.84× |
| TSDS | 4.663 | 4.685 | 约持平 |
| LP-Mirror | 0.0159 | 0.0150 | 约持平 |
| LTSV | 9.546 | 2.792 | 3.42× |

四臂选择 ID/窗口/逻辑计数均一致。真实模型投影梯度相对 L2 差为 `4.67e-7`。
LESS 的收益包含 11 GiB 设备上的 OOM 退化修复；96 GiB 设备未必有同样收益。
LTSV 新版首次完整调用 5.672 秒，包含捕获成本；LP-Mirror 首次调用的巨大差距
主要是初始化次序，不能声称稳定加速约 7 倍。TSDS 仍受全池 encoder 计算量约束，
本次没有为它单独改算法、缩池或近似特征。

原始开发输出在 dev 的 `outputs/execution-0912/`（TMP，不提交）。
正式 exp 没有由本次开发停止、拉取或重启；在途任务继续使用已加载的版本。
旧版与新版性能数据须标明代码版本；不要为了应用优化中断已经运行数小时的臂。
