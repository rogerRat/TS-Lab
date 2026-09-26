# 2026-09-15 十任务正式主实验接续

## 当前覆盖：轨迹内 P10（2026-09-15）

用户修改方法后，正式配置明确使用 `crb.epsilon_policy=trajectory_p10`。
每条源/参考轨迹只取均布与误差阶段的不同真实梯度模长，以 float64 计算
`quantile(.10, method="linear")`，第三阶段前冻结；旧分裂、LESS及其他轨迹
缓存中的额外窗口不计入。P10=0 时零代表仍剔除，非有限值直接报错。
固定阈值模式只为历史配置兼容保留，当前正式配置不再引用外部 epsilon.json。

中断时三模型均在首任务 assets 阶段，7 个 Original 结果保留，无选择概率、
训练结果或训练断点需要作废。复用原路径同身份逐窗口缓存；重放 CRB 的
前两阶段并按新阈值执行第三阶段，只计算缺失梯度。迁移脚本为
`scripts/exp/migrate_trajectory_p10_0915.py`，先检查无活跃 worker 和派生产物，
再备份 protocol/readiness/results 至 run/migrations/trajectory-p10-20260915-v1。
同目录 migration.json 记录新旧协议、模型身份与保留梯度数。

阈值审计位于各模型 grad_cache 下 `crb-thresholds-<task>.json`，包含
模型、完整 CRB 配置、源/参考逐轨迹身份、阶段1/2锚点数、P10及冻结阶段。
在该任务 CRB 准备完成后原子发布；梯度 NPZ 仍按原窗口身份增量写入。
下文开发 P10 标定流程仅为此次修正前的历史执行记录。

部署验证：相关回归65 passed、1 skipped（CUDA隐藏的并行测试），
覆盖阈值冻结、跨轨迹批处理一致性、零/非有限值、已有缓存额外窗口隔离、
无外部标定文件的RAGS选择/训练/完成格恢复、分裂预算及成本。
保留梯度：Chronos275456、TimesFM162816、TimeMoE52224，共490496窗口。
队列第5项 attempt3 已恢复 running；worker PID510377/510379/510380
分别为Chronos/TimesFM/TimeMoE。三份日志均报告trajectory_p10及上述缓存命中数。
当前仍在梯度准备，完整正式矩阵尚未完成。本轮改动未提交/Push。

用户已授权：当前诊断计算结束后接续主实验；正式任务迁移为人工确定的十任务。
依据本地更新后的实验稿和一致性审计顶部用户裁定，旧 .tex 不作为执行依据。

配置：`configs/rags/main_shared_ten_20260915.json`；生产工作区 `/root/TS3`。
输出：`/root/autodl-tmp/TS3/outputs/main-shared-ten-20260915`。
诊断工作区 `/root/TS3-probe` 继续原计算，不以其旧训练代码启动主实验。

冻结内容：三模型各自预训练探针，BF16 模型/梯度/AdamW，4096 维投影、seed42，
同身份开发 P10；CRB4/2/2%；三种子0/1/2；共同源池N16284、合法起点9052777。
主表六训练臂 rags/random/less/tsds/lp_mirror/ltsv，另有未训练 original；
共540训练cell及30原始模型cell。RAGS-Traj/Uniform单独接口不扩入十任务主矩阵。

模型训练细节完全对应 shared_tasks_10.json，TimesFM停在2.625，调度仍按6Epoch；
TimeMoE有效batch64/micro16。每cell完成评测后按已有用户授权清理该cell权重，
结果/恢复所需未完成cell状态保留。模型串行执行以限制缓存/检查点并发存储。

正式 Ref/考试从原生 TIME/Quito 重新构造：原生尾部或固定cutoff为考试，H64滚动，
不是开发八窗；完整可用考前历史作Ref。边界与原生值核验见
`inputs/native_input_audit.json` 及 `shared_formal_inputs_20260915.md`。
源池沿用已核对source_load、seed0、W576、N16284的历史共同源池，仅复制原生池及manifest。
不复用旧模型梯度、选择概率或旧P10。插值按需计算，不保存逐窗重建数组。

队列位于 `/root/autodl-tmp/TS3/outputs/task-queue/queue.sqlite`：

1. 等待现有 scan PID275956 成功完成260041窗口并退出。
2. Chronos Base BF16 开发P10标定。
3. TimesFM2.5 BF16 开发P10标定。
4. 等待 `readiness.json` 的 ready=true。
5. `rags.experiments.formal_main_entry` 检查审核配置未变、模型身份/P10、配方及预算后运行主表。

TimeMoE BF16 P10 已完成12539窗口，值0.44411512371641215，使用诊断下该标定资产。
队列失败自动阻断，不擅自回退Small/FP32/旧六任务或改变基线估值粒度。

**概率温度已冻结。** 用户确认三模型的 RAGS/LESS/LP-Mirror/LTSV 统一 θ=1；
Random/TSDS 保留原生机制，不强加温度。三个同身份 BF16 P10 均已完成，
正式配置及 P10 身份预检通过后，解除 readiness 等待，队列自动启动主实验。

实施验收：CRB实际分裂预算、数据原生边界、原始梯度线性表示、两个变体、梯度成本、
三模型训练配方/2.625停止点/累计与恢复、正式worker与队列配置相关联合测试65passed。
未声称540训练cell已运行，未进行三模型真实GPU训练冒烟；当前尚未提交/Push本轮改动。

## GPU 利用率修正

用户要求提高昂贵容器利用率。连续8秒采样，单Chronos eager原始评测GPU36–40%、约4GiB，CPU约1核。对同128个真实开发窗口，batch32约52.6窗/s、128约150.3窗/s，但BF16自回归输出不完全一致；同batch32的CUDA Graph热运行也较快但输出不完全一致，因此两者未部署。证据在运行目录 eval_batch_benchmark.json 和 eval_graph_benchmark.json，均是开发窗口预测，未读测试收益选设置。

采用不改变数值执行路径的跨模型并发：execution.json设置student_parallelism=3、original_first=false，各模型各任务在训练前仍先完成自身原始评测。有效训练batch、microbatch、梯度与原始评测batch、eager均保持。已保存的前五项Chronos原始评测复用，受控中断尚未完成的Quito原始评测；队列第5项重试，继续同一正式run。编排/worker12测试通过。

## CUDA MPS 执行优化（2026-09-15 22:15）

三模型和各自独立梯度进程直接共享 GPU 时，`nvidia-smi` 虽显示100%利用率，
但实测仅约212/600W、显存带宽利用约11%，TimeMoE长期吞吐仅3.54窗/秒。
保持全部科学配置、batch、worker数、模型精度和逐窗口原生梯度路径不变，
仅启用默认路径的 NVIDIA CUDA MPS，使多进程通过同一MPS server提交工作。

停作业后的30秒三模型同步稳态A/B如下（窗/秒）：

| 模型 | 无MPS | MPS | 倍率 |
| --- | ---: | ---: | ---: |
| Chronos Base | 18.50 | 33.21 | 1.80x |
| TimesFM 2.5 | 10.83 | 22.59 | 2.09x |
| TimeMoE 200M | 3.67 | 19.61 | 5.35x |

MPS下再以各模型单窗结果为基准通过数值门槛：最大loss差不超过
`2.4e-10`，最小梯度方向余弦不低于`0.99999999998`，最大相对梯度误差
不超过`5.7e-6`，无OOM。恢复后的正式缓存前两分钟实测约为Chronos
34.2、TimesFM 23.1、TimeMoE 18.9窗/秒；GPU约43/98GiB、功耗约
383--400W。A/B及数值记录位于
`/root/autodl-tmp/TS3/outputs/perf-execution-20260915/`。

attempt4于22:03受控SIGINT后缓存完整收尾，attempt5于22:15从原分片无损恢复。
当前默认MPS进程由`nvidia-cuda-mps-control -d`启动；监控必须把control/server
存活作为attempt5健康条件。MPS不是科学协议变更，不修改`protocol.json`或
`execution.json`；若MPS消失，先报告并协调受控恢复，不静默重试队列。

22:26--22:27补充跨执行核对：`scripts.exp.verify_cached_gradient_execution`
直接按旧分片记录的`fragment_id/physical_start_1based`重建源窗口，在MPS下
复算并与启用前的正式BF16缓存比较，每模型8个分散样本。三者loss均完全一致；
最大相对梯度差依次为Chronos 0.001979、TimesFM 0.001772、TimeMoE 0.001702，
最小非零方向余弦均超过0.999998。差异包含未量化FP32投影与BF16存储的量化差，
通过原有cosine>=0.9999、relative<=0.01门槛；这是有限样本核对，非全缓存证明。
TimesFM样本含1个两边同为零的梯度，单独判定相等（零向量角度无定义）。
首版校验器因此误报的报告保留，修正后的报告为
`cache-agreement-timesfm-zero-aware.json`，另两份为`cache-agreement-chronos.json`
和`cache-agreement-timemoe.json`，均在上述perf目录。只读取正式分片，每次最多
额外一个审计进程，正式worker未再次停止。

计时口径：目前`gradient_timing.json`的同一记录混含MPS启用前后时间，
不能用累计均值推算当前ETA，也不能把跨22:03--22:15停机的30分钟差值称为
MPS稳态吞吐。当前速度使用同attempt、启用后的计时/分片差值；计时累计还可
包含中断时未发布的部分分片，实际可复用窗口数以原子发布分片及恢复日志为准。

## 优化边界与Baseline耗时口径（用户裁定，2026-09-15 22:49）

优化点到为止，停止收益不明确的性能探索。保留已验证的框架层MPS加速；
各方法都在相同框架执行条件下比较，框架共享/并发耗时须如实标注。
RAGS自身可积极优化，但不能改变方法语义。其他Baseline若采用方法专属优化，
必须标明“本框架优化实现”，并同时给出未做该专属优化的实现耗时；若采用估算，
须说明同配置测量依据、计算方式与不确定性，不能只报告优化耗时。
本框架的时序适配实现与上游原生代码不是同一个执行环境，不能把适配版时间
未经验证地称为上游原生时间。

用户随后澄清：明确有效的LTSV加速应继续保留，只还原Baseline专属优化耗时；
统一框架MPS不反向折算。22:49的临时撤回已被本条覆盖，现已重新接入GPU恢复，
且必须存在各student的`method_cache/ltsv_restore_timing.json`同模型配对记录才启用。
选择产物`cost.notes.runtime_adjustment`保存实际GPU恢复块数K、额外准备时间和
补时：`reported = actual - extra_setup + K*(CPU_restore-GPU_restore)`。
用`scripts.exp.report_selection_runtime`生成`reports/selection_runtime_comparison.json`，
比较使用`reported_*`，实际优化耗时另列，不能把约20倍的恢复倍率乘到整方法。
尚未完成LTSV正式调用，当前报告仍是缺项清单；不会伪造完整方法耗时。
用户指定的主文档`Projects/001-iclr17/rags-md/_exp-0907.md`已同步此决策。
Adam状态复用的独立合成测试也未部署：C/F/M中位数fresh/reset约
26.95/28.58、31.82/23.40、57.05/49.90ms，波动明显且Chronos无收益；
不再追加探测来追逐这类小收益。原始报告留在独立perf目录。

## LTSV 权重恢复预检依据（专属加速按上述口径补时）

首任务进入训练前，worker仍会先完成六个方法的静态选择。LTSV按既定
W576/stride288覆盖共同源池，精确需要33058个独立块更新/任务，每块后对64个
安全Ref窗口做原生预测，再恢复预训练权重。不要把assets完成理解为马上产出训练cell。

原实现每块都把整套BF16参数从CPU复制回GPU。现在仅在副本大小不超过
当前空闲显存5%时，把相同位模式的不可变扁平恢复副本留在GPU；内存不足或
分配OOM保留CPU路径。没有改变Adam、逐块更新、loss、预测batch、模型精度、
采样/RNG或评分公式。副本随本次选择返回释放，不保存磁盘权重副本；
另保存小型配对计时JSON和选择成本补时元数据，供报告按用户口径折算。

`scripts.exp.bench_ltsv_restore`每模型取2个真实源块、首任务64个安全Ref窗口，
交替比较CPU/GPU恢复，未读考试数据。调用实际生产恢复helper，且像生产一样在
参考预测期间保留Adam状态。三模型的loss/参考MSE/块价值均完全一致，最终参数
逐位恢复一致；这是有限样本证据。五次纯恢复耗时中位数如下：

| 模型 | CPU恢复 | GPU恢复 | 副本显存 | 仅恢复开销外推节约/任务 |
| --- | ---: | ---: | ---: | ---: |
| Chronos | 43.73 ms | 2.25 ms | 0.375 GiB | 22.9分钟 |
| TimesFM | 48.57 ms | 2.05 ms | 0.431 GiB | 25.6分钟 |
| TimeMoE | 99.62 ms | 5.54 ms | 0.844 GiB | 51.8分钟 |

并发MPS下预检进程自身峰值allocated显存依次5.022/3.360/7.278GiB，
不是整卡占用或完整正式LTSV峰值。上述外推不是全流程加速率或完工ETA；
Chronos单块的64窗预测仍约0.88秒，是更大的剩余开销。
权威报告在独立perf目录`ltsv-restore-*-production-helper.json`；早期同名无后缀
报告保留，但其诊断提前释放Adam，不用于生产峰值判断。正式结果目录只读。

LTSV runner由full_pool在首次执行该方法时惰性导入，本轮未停止attempt5。
tmux exp已ACK：保持原进程，实际阶段到达时核对
`[LTSV] immutable restore snapshot: cuda...`，不能提前声称正式LTSV已用新路径。

验证：`test_native_ltsv.py`及`test_time_series_baselines.py`使用pytest筛选
`-k "not end_to_end"`合计14 passed、2 deselected；覆盖
GPU/CPU恢复位一致、完整小型LTSV分数一致、5%边界及OOM回退、失败恢复、
既有评分/损失语义。两个旧缓存端到端用例未执行（会走历史hash命名路径）；
本次没有改写或迁移历史基线缓存规则。脚本编译与git diff --check通过，未提交/Push。


## LTSV预算缩减与MPS恢复（2026-09-16）

用户允许为运行时间牺牲部分性能并要求保留原版倍率证据。211 run已接入
LTSV-Budget：stride288→576，每任务33058→17361块；Ref64、eager、一步Adam保留，
加GPU恢复、按显存约束的4/2/2 worker与原子分片续跑。详见
[LTSV预算、牺牲与计时证据](ltsv_budget_20260916.md)。
三模型小样本并发外推倍率约4.04/3.47/3.16；不是正式完整运行倍率或下游精度保证。
原始配对计时与正式配置在该文档指向的JSON中；CUDA Graph和Ref16因收益不抵变化未部署。
诊断终止后MPS fatal teardown导致attempt7故障，证据已保全；恢复队列常驻worker后
attempt8已从缓存续算。未为LTSV配置部署再次中断作业，未提交/Push。

## T=0.25 温度补充臂接入（2026-09-16 11:36，用户授权，zcode 实施）

用户决策：仅 RAGS/LESS 增加 T=0.25 选择/训练/评测对照，其余臂不加温度、LTSV 不重算。
在主 run main-shared-ten-20260916-211 原地扩展（受控 SIGINT attempt9 → 安装 → retry，空窗约 3 分钟）：

- **代码**（备份 *.pre-t025-0916）：full_pool.py 增加 TEMPERATURE_ARMS/TEMPERATURE_ARM_BASE 与 dispatch（rags_t025→_rags、less_t025→less_select，输出重命名）；full_pool_worker.py 资产准备按基础方法名映射、缓存选样新增温度一致性校验；full_pool_run.py shared-ten 协议下正式矩阵要求 8 臂。
- **配置**：main_shared_ten_20260916_211.json、run/readiness.json、run/protocol.json 同步修订（备份 *.pre-t025-0916），仅 arms +[-rags_t025,-less_t025] 与各学生 temperature 增 0.25 两处差异；冻结等价/预检等价/validate_config 三重预演通过。
- **复用验证**：timesfm 任务1 pool CRB 46s 全缓存重放；LESS 原生分数→T=0.25 概率重建与 T=1 存档逐位一致校验（top10 质量 0.195→0.347，均归一）；chronos/timemoe LTSV 断点恢复 0 new + N cached blocks 零重算。
- **测试**：staging 61 passed（含 full_pool_run/worker/resampling/regions/training_resume 等 8 文件）；合成端到端冒烟（dispatch/命名/温度元数据/归一/npz 回读）全通过。
- **首证**（11:33-11:35）：timesfm selections/time_nem_demand_5t__{rags,less}_t025.npz（temperature=0.25，概率和=1，与 T=1 分布不同）；state=training rags_t025 seed0；attempt10 三 worker 存活，无孤儿、无双写，T=1 cells 26 与全部选样完好。

## 范围澄清：T=0.25 改为独立补充 run（2026-09-16 12:10，用户澄清，zcode 执行）

前节的原地扩展方案被用户范围澄清取代：T=0.25 仅 RAGS/LESS 两臂，采用独立温度命名空间的
补充 run，排在主 run 之后，避免污染/中断主 run；LP-Mirror 与 LTSV 完全排除（非"有缓存才做"）。
- **主 run 回滚**：11:44 受控 SIGINT attempt10（曾按前授权原地扩展 8 臂）→ t025 产物迁出
  （2 选样+1 cell 暂存 /root/t025-artifacts 后预入补充 run）→ config/readiness/protocol 从
  .pre-t025-0916 备份恢复 → attempt12 以六臂冻结协议恢复，LTSV 断点零重算续跑，零 t025 文件。
  过渡中 attempt11 因校验器仍要求 8 臂而 preflight 失败（未写任何产物），修正校验器（主 run
  六臂或补充 run 恰两温度臂均合法）后 attempt12 正常，无数据影响。
- **补充 run**：main-shared-ten-20260916-211-t025，arms=[rags_t025, less_t025]，
  config main_shared_ten_20260916_211_t025.json（其余字段与主 run 逐字相同，仅 arms+temperature
  两处）；grad_cache 符号链接共享主 run 缓存（梯度/CRB/LESS 分数全复用）；execution.json 继承；
  启动包装 /root/t025-run-launch.sh 先从主 run 复制 original/random cells 再进入
  formal_main_entry（Random/TSDS/LP-Mirror/LTSV 均不重算）；timesfm 任务1 两个 t025 选样与
  rags_t025 seed0 cell 已预置。已接入队列为 job6（job5 成功后自动执行）。

## T=0.25 即时配套编排（2026-09-16 13:35，用户指令，zcode 实施）

补充 run 不再排队等待主 run：任何 T=1 rags/less 结果落地后立即补对应 t025。实施：
worker 增加 --tasks 子集参数（.pre-tasks-0916 备份，13 测试通过）；队列 job6 skip；
tmux t025filler 常驻 /root/t025-filler.py 按 (模型,任务) 粒度监听补跑（串行+显存门槛
10/22/38G+锁重试+baseline cells 复用）；supp run 补写 protocol.json（entry 同构）。
首证：timesfm nem t025 于 13:31 开始补（rags_t025 seed1 训练中）。

## T=0.25 设为默认 + RAGS-Gateless 定名（2026-09-16 16:25，用户决策，zcode 实施）

用户覆盖此前"取消 t025"：后续排程中 RAGS/LESS 一律 T=0.25（主 run 协议改混合矩阵
[rags_t025,random,less_t025,tsds,lp_mirror,ltsv]，第三次受控重启 attempt13），
T=1 存量保留为对照不覆盖；lp_mirror/ltsv 维持 1.0。无硬门控消融定名 RAGS-Gateless
（slug rags_gateless，弃用临时名 rags_nogate，无旧名产物），T=0.25、仅任务1 nem，
独立 run main-shared-ten-20260916-211-gateless + 常驻 runner（显存/锁/互斥三层门槛）。
16:18 主 run 三模型 nem t025 选样落盘并开训；16:21 gateless timesfm 首发。

## RAGS-Traj-gls 正式基线接入（2026-09-16 16:55，用户决策，zcode 实施）

新增独立臂 rags_traj_gls（展示名 RAGS-Traj-gls，T=0.25）：严格沿用既有 RAGS-Traj 定义
（RAGS CRB 线性锚点重构、按合法窗口贡献的轨迹聚合、逐参考方向、π_j 行混合、轨迹内均匀采样），
仅关闭 rags_marginal 硬门控（gating=False，全部有限余弦参与行 softmax）。审计确认 _rags_traj
路径本即 RAGS 重构非 LESS 式。主 run 协议第 4 次修订（attempt14）：arms += rags_traj_gls（第 7 臂），
temperature += 0.25；存量命名空间（rags_traj/rags_gateless/rags_t025/less_t025/T=1 对照）零覆盖。
端到端合成测试 + 18 项回归通过；16:52-55 三模型 nem 选样落盘，chronos 开训 seed0。

## CRB 分裂语义与后续矩阵清理待办（2026-09-16 18:36，用户决策，tmux codex↔zcode 已同步）

用户要求修改当前 RAGS 的 CRB：分裂预算必须全部用完；每次从当前区间中寻找最小的一批可分裂段并执行二分，持续到预算耗尽；分裂过程中不得因方向、检测结果或预算不足丢弃任何轨迹段，所有保留段最终都要进入后续匹配。该项目前是待实现的代码/测试变更，不能把本条记录误写成已部署或已重算的实验结果。实现时需同步核对 `src/rags/crb.py` 的分裂、检测与计数语义，以及 blocks/trajectory 聚合、序列/多片段等价性、回归测试和实验报告字段；旧的 discarded/banned 计数应明确为零语义或改成与新协议一致的诊断字段。

用户同时决定：后续实验矩阵删除 `rags_gateless` 与 `rags_traj_gls`，不再安排这两个方法。已完成产物保留作历史溯源，不删除、不覆盖，也不改变当前正在运行的主实验配置。

协调约束：tmux codex 通信端点为 `codex:0.0`（pane `%0`）；zcode 实验主管已确认主 run `job5 attempt14` 当前保持运行、不停止、不修改。当前存量主协议仍是七臂
`rags_t025/random/less_t025/tsds/lp_mirror/ltsv/rags_traj_gls`；NEM 已收官，Azure 仍在推进。CRB 代码、未来矩阵及其计算/报告内容应在该作业完成或主管明确迁移窗口后修改，并先完成测试与配置校验。

## 缓存后参考轨迹异质性诊断（2026-09-16，用户新增需求，tmux codex 已通知 zcode）

新增一个不参与选样和匹配的解释性诊断：只要某次方法已经把窗口梯度写入共享缓存，就按参考轨迹汇总已有缓存窗口，判断该模型下轨迹是否异质、非平滑或方向不一致。诊断不额外触发梯度计算；若缓存只覆盖部分合法窗口，必须报告覆盖率，不能把稀疏缓存伪装成完整轨迹结论。

每条轨迹至少报告：`coherence_ratio=||mean(g_t)||/mean(||g_t||)`（轨迹聚合保留方向的程度）、`cancellation_ratio=1-coherence_ratio`（均值抵消）、窗口方向相对轨迹均值方向的离散、相邻缓存窗口的方向跳变和相对梯度跳变、梯度幅值变异，以及合法窗口/缓存窗口数量。必要时附带 CRB 块方向相对轨迹方向的对齐统计。高抵消或高跳变轨迹是轨迹级聚合可能损失局部结构、而块级 RAGS 可能更有优势的可解释证据；诊断本身不预设哪种方法更优，也不作为新的实验臂。

诊断需落盘到梯度资产/缓存旁并在选择产物中留引用，保证 RAGS 原生、LESS 及其他复用同一梯度缓存的方法使用同一模型、同一参考轨迹口径；报告时按轨迹逐条保留，并同时给出汇总统计。当前 job5 attempt14 不停、不改；该功能先实现和测试，待后续任务使用。

## 参考轨迹诊断落地确认（2026-09-16，zcode 主管回报）

zcode 已将该诊断按只读独立流程落地：梯度缓存采用 append-only 分片与 `window_index` 行表，诊断不取写锁、不进入 worker、不修改 run 的 `results.json`。产物统一写入 `<run>/diagnostics/<model>/<task>__reference_traj_diag.json`；后续任务在资产完成后由巡检脚本补算，报告层再并入矩阵比较。

当前指标口径为逐轨迹方向离散度（`1-cos` 到轨迹均值方向）、相邻窗口余弦/跳变、均值抵消 `rho=||sum(g)||/sum(||g||)` 和梯度范数 CV。首批 NEM 参考侧（5 条轨迹/模型）：TimesFM `dispersion=0.893`、`adjacent_cos=0.287`、`rho=0.197`；Chronos `0.804/0.634/0.205`；TimeMoE `0.582/0.807/0.440`。这只能作为解释性证据：TimesFM/Chronos 的参考轨迹方向更分散、均值抵消更强，提示轨迹级聚合可能损失局部结构；不预设方法优劣。源池侧扩展及逐轨迹与 LESS 分数的关联分析列为后续工作。

以上诊断对当前 `job5 attempt14` 零影响；此前本会话尝试的另一套 cache/worker 接入已回收，不与主管正式口径并行。

## 诊断与主实验最新复核（2026-09-16，zcode 主管回报）

二次清理复核通过：四个相关 `src` 文件编译通过，zcode 补丁集（7 臂 form、gateless slugs、方法映射、gating 参数、`--tasks`）保持完整；独立只读诊断目录现有 6 份产物（NEM×3，以及 Azure×TimesFM/Chronos、Darwin×TimesFM）。

`job5 attempt14` 继续健康运行且无孤儿：Chronos 正在 Azure 选样，TimesFM 正在 Darwin 选样，TimeMoE 正在 Azure 资产阶段。后续仍由 zcode 在任务资产完成后补算诊断，并由巡检每拍更新 `docs/MAIN.md` 及预注册预测 P1/P2；worker 与 protocol 路径保持不变。

## LESS 主实现切换为数值 softmax（2026-09-16 20:05，用户指令，zcode 实施）

用户裁定 rank-softmax 为项目自定部署层且无记录依据：LESS 主实现改为 RAGS 式数值 softmax
（exp(s/T)×长度，s=sum-of-cosines，符号门控保留），新 slug lessv_t025 进主协议第 7 臂
（替换 less_t025，第 5 次受控重启 attempt15）；旧 less/less_t025 产物原样保留、口径降级为
LESS-RankOnly（非主实现）。TSDS 不受影响（native OT mass）；LP-Mirror/LTSV 同为 rank 实现，
已通知用户待裁定。插队兑现：attempt15 起三模型 nem lessv_t025 即时回填（选样 mapping=
positive_alignment_value 已核验，20:0x 起训练）。事故记录：staging baselines 符号链接导致
runner.py 一次未备份的误写，已用精确串逆恢复原样后重新规范部署（备份 .pre-lessv-0916）。
