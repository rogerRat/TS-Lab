# 全参考窗口梯度与任务异质性分析（2026-09-21）

用户授权：Codex 写代码、测试；正式计算与迭代执行交给 WSL 的 tmux moni 内 Claude。
当前三模型为 chronos-base、timesfm-2.5、timerxl-84m。任务从当前 V5、V5-uni、TimerXL run 的 protocol.json 取并集，不凭旧任务列表硬编码或抽样替代全量。

## 科学范围与冻结约束

全量 = 所有当前参考片段的安全历史中所有 C+H 合法窗口，stride=1，包含首尾起点。每窗原生独立 loss/BP；沿用正式评分模型 checkpoint、精度、参数范围、Count-Sketch 投影和 BF16 存储。这里不是完整参数梯度落盘，也不是 CRB 插值梯度。结果必须注明投影和量化限制。

safe_references 对照 manifest 的 full_reference_history、formal_exam_excluded_from_reference、source_training_only、C/H、实体/片段ID、长度、有限性及每个 stop<=exam_boundary 逐字段校验。计算入口不读取考试数组。历史文件若同路径换数据必须变更显式 revision；不使用 hash。

现有训练/选集/考试结果保持冻结。新梯度使用独立分析目录，不写旧 grad_cache，也不重新选集或重训。这意味着首次分析会重新计算原先约4%的参考锚点；后续分析在新缓存内只算缺失窗口。同模型的 V5/uni 共用该缓存，不同模型严禁混用。

## 第一轮假设登记（全部探索性）

| ID | 猜想 | 资源与特征 | 配对结果/反证 |
|---|---|---|---|
| H1 | 轨迹内方向异质性越高，块级越有利 | 全窗口方向方差、范数保留方差、抵消率 | 同聚合器 block vs traj；负相关/留一变号均须报告 |
| H2 | 异质性或历史漂移越大，RAGS 相对 TSDS 越有利 | 轨迹内/间分解、前后四分位梯度方向/位移 | 全部 RAGS vs TSDS；同步比较 LESS、LP、rand |
| H3 | 剧烈/不平滑梯度影响 CRB 代表质量 | lag 1、H、C+H 的方向变化、相对步幅、二阶差分；完整真实梯度重放 CRB | 非锚点重构相对 RMSE、方向误差、锚点均值偏移与任务收益 |
| H4 | 更好的源-参考梯度对齐对应更好的迁移 | 全参考精确期望 × 各方法真实源分布的2048个抽样原生梯度；均值原始梯度余弦 | 对齐绝对值、相对 rand 改变量及方法间差值关联收益 |
| H5 | 少数大范数窗口主导轨迹代表 | 范数CV、前1%范数质量、方向抵消 | 各 RAGS 聚合器的收益及 block/traj 差异 |
| H6 | 选集过度集中影响迁移 | 逐窗口概率 ESS/全池窗口数，抽样源轨迹最大份额 | 相对 rand/TSDS；TimerXL同run内 block vs uni |
| H7 | 数据长度、可优化空间或原始信号解释差异 | 窗口数/轨迹数、orig/rand MASE及rand相对orig改善、lag1自相关、标准化步幅/均值漂移/零值率 | 同上，另给控制log参考窗口数后的描述性秩相关 |

梯度期望主定义为每个参考有限片段等权、片段内窗口等权，匹配现有 V5 评分。零梯度的单位向量定义为0并单列零比例；需要方向的余弦若范数为0则为null。另保存原生实体等权、实体内按有限窗口数加权的对齐敏感性结果，避免一个实体有多个缺失分段时混淆实体与片段。

统计单位是任务：同任务/模型/方法的全部配置seed齐全才形成一个配对收益，100*(1-mean(MASE_arm)/mean(MASE_baseline))。不把重叠窗口、seed或同任务的不同模型当独立任务。各模型、run分别检验，四个聚合器全部报告；不会逐任务挑最优方法。默认9999次双侧任务标签置换，1999次任务bootstrap，逐任务留一，所有有效检验统一BH校正。置信区间是未多重校正的边际区间；相关非因果。九任务下不显著不等于无效应。

跨run公共臂不混合；V5-uni与V5中的block/uni跨run对比不自动构造（需要另核对配方、seed与共同基线）。TimerXL的15臂同run可以直接做block-vs-uni。

## 执行交接

执行者 Claude 先核实 TimerXL 队列收尾和资源，再在 tmux 接续。维持用户的每30分钟巡检频率，不另建更频繁巡检，不关机。可并行不同模型，但同模型分析目录必须单写者，两个run同模型顺序执行。

在 `/root/TS3` 的 rags 分支执行。三个 run 当前为：

```bash
MAIN=/root/autodl-tmp/TS3/outputs/main-rank-volume-20260918
UNI=/root/autodl-tmp/TS3/outputs/main-rank-volume-uni-20260920
TIMER=/root/autodl-tmp/TS3/outputs/main-rank-volume-timerxl-20260920
ANALYSIS=/root/autodl-tmp/TS3/outputs/reference-census-20260921
/root/miniconda3/envs/ts3/bin/python -m rags.experiments.reference_census plan --runs "$MAIN" "$UNI" "$TIMER" --out "$ANALYSIS/inventory.json"
```

先阅读 inventory：逐模型任务数、每条任务合法窗口数、BF16梯度载荷字节下界及已有时延推算。时间估计为历史混合负载的BP时间，正式运行后用新缓存的实际吞吐更新，不把它当承诺。空间还需索引、loss、对齐抽样、内存与余量；worker检查1.25倍未计算参考载荷+1GiB，操作者额外检查多个模型同时运行的总预算。磁盘不足不得悄悄采样或写系统盘，应报告。

已完成CPU资源盘点：27个独立模型×任务，每模型4,978,074个合法参考窗口，共14,934,222个；4096维BF16参考载荷113.94GiB，另需索引/源对齐/运行日志。最初df显示约145GiB空闲，余量不宽裕，启动前必须重新核对全批总预算，不能将三个worker各自磁盘检查当作并发配额。Azure单任务3,538,158窗，占每模型71.1%。历史梯度均时粗估Chronos62.9h、TimesFM73.5h、TimerXL8.0h；它们不是并行墙钟承诺，也不包括全部分析IO。优先用首批实际吞吐校正，保持全量范围，任何压缩方案必须无损并保持字段/精度/窗口身份。

阶段A，全部参考梯度与几何（各模型用自己的已安装环境；示例）：

```bash
/root/miniconda3/envs/ts3/bin/python -m rags.experiments.reference_census compute --run "$MAIN" --student chronos-base --out "$ANALYSIS" --stage references
/root/miniconda3/envs/ts3-timesfm/bin/python -m rags.experiments.reference_census compute --run "$MAIN" --student timesfm-2.5 --out "$ANALYSIS" --stage references
/root/miniconda3/envs/ts3-timerxl/bin/python -m rags.experiments.reference_census compute --run "$TIMER" --student timerxl-84m --out "$ANALYSIS" --stage references
```

阶段B，已有正式选集的分布对齐：对上面三命令把 `--stage references` 改为 `--stage alignment`；再对 chronos-base / timesfm-2.5 各运行一次 `--run "$UNI" --stage alignment`。默认2048源抽样/方法，所有参考窗精确纳入期望，不抽参考窗。新source键按窗口身份去重BP，重复抽样在估计权重中保留。现有评分/选集禁止额外BP的约束不变；这里的新增BP仅用于独立机制分析。

阶段C，CPU统计与报告：

```bash
OPENBLAS_NUM_THREADS=4 /root/miniconda3/envs/ts3/bin/python -m rags.experiments.reference_associations --runs "$MAIN" "$UNI" "$TIMER" --census "$ANALYSIS" --out "$ANALYSIS/report"
```

模型路径/环境以当前 protocol 与实际环境为准，不能换成其他权重。execution沿用run设置，支持 `--gradient-batch` 仅调整执行批大小；CUDA OOM时只减小batch重试。模型支持的批量VJP保持独立每窗梯度；记录gradient_timing、实际batch计数、峰值显存与耗时。

compute先在首个安全参考片段的前三个窗口执行原生单窗/批量对照，保存 probe_validation.json（loss与梯度相对误差、5%阈值）。不一致或非有限即停止，不进入全量计算；该小检查是数值/独立性冒烟，不代表完整模型证明。成功记录按全部身份字段复用，三个窗口的少量重复BP单列为冒烟，不冒充缓存覆盖。

## 资源格式、恢复及后续探索

- `ANALYSIS/<student>/grad_cache/<readable-model-revision>/`：沿用window-gradients-bf16-v1，NPZ保存dataset、fragment_id、1-based start、原生loss、BF16投影梯度。独占写锁，临时文件原子发布，断点只补缺失窗口。保留全部，不自动清理。
- `<task>__geometry.json`：身份字段、coverage(expected=actual)、全部轨迹级度量和任务级期望。分析以有界窗口块流式读取，内存O(chunk*D+n)，不创建N×N矩阵。CRB只读已有真实梯度重放，不新BP。
- `<task>__alignment__<run>.json`：当前对齐主记录；另存带seed、n后缀的每次抽样记录，保留采样window IDs和MC标准误。更换抽样seed/样本数不可删除前次结果。共同随机数不保证严格配对方差抵消，单列SE不能当方法差值SE。
- `state.json`：当前任务片段完成数、计算数量、资源路径；最终stage=references不是整个机制实验完成。geometry与缓存覆盖不一致会停止，不把缺失资产当完成。
- `report/{associations.json,effects.csv,tests.csv,REPORT.md}`：全部逐seed配对收益、全部检验、缺失清单；缺数据会列明，不静默称全量显著。

正式产物只放TMP，不提交Git。代码、测试、方案放核心仓库。新增代码目前不复用旧约4%锚点缓存，避免在实验收尾期间改写共享资源；新独立全量缓存将成为后续资源。

迭代：读第一轮完整结果→登记新猜想与判定规则→列已具备/缺失资源→补算→再次验证，并累计纳入多重检验家族。优先复核留一敏感的结论、异号模型和反例任务；对齐先用另一个seed复测并扩大源样本（例如8192），保留前次文件；再考虑季节lag、源域组成/覆盖、层或投影敏感性及安全历史独立验证。此轮不自动改冻结科学协议、不依据正式考试调选参数。涉及新训练干预先形成明确实验对照与资源清单再交接。

每轮将假设、命令、结果与反例、未完成项追加到 docs/Progress.md，并写到该批次 hypothesis-ledger.jsonl 或 Markdown 流水账。当前尚无正式机制实验结果，不能据此声称任一假设成立。
