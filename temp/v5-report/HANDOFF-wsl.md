# V5 实验巡检与总结：从本机交接到 WSL tmux moni

## 用户授权与责任

2026-09-18 用户明确要求：本机停止巡检；由 SSH `wsl` 上 `tmux moni` 中的 Codex 进程接管，模型 **gpt-5.6-luna**、推理强度 **high**。你是此实验的新巡检和总结负责人。不创建其他任务、不委派子代理。原本本机 heartbeat `v5` 已 PAUSED，不应重新启动。

用户已授权正式运行、必要的开发/测试和故障恢复，要求尽可能充分利用 GPU。持续推进现有 V5，正常进度安静记录；关键阶段/整轮完成、失败、需要用户决策、全矩阵完成才总结。只使用当前用户授权，不发送 Slack/邮件。不要关机、释放容器、重启 V4、启动新实验矩阵或重复训练已完成 cell。

## 连接与运行身份

- WSL 已有 FlyingBird VPN，用户再次确认 GPT 必须通过其 **127.0.0.1:7892** 代理。monitor.py 仅对 Codex 及其子进程设置 HTTP_PROXY/HTTPS_PROXY/ALL_PROXY（以及小写形式），NO_PROXY 为 localhost/127.0.0.1/::1；不更改全局网络配置。不带代理的 GPT 请求会超时，已验证代理恢复模型响应；不可移除。实验 SSH 通路不依赖 HTTP 代理。
- 本机 `wsl` 上能直接运行 `ssh -p 26856 root@connect.bjb1.seetacloud.com`；已验证无交互连接。
- 实验源码 `/root/TS3`，分支 `rage`；首先读取 `/root/TS3/AGENTS.md`，然后 `docs/MAIN.md`、`docs/method.md` 和 `docs/Progress.md`。
- 冻结科学代码提交 **6d3ead6**：`Freeze V5 ranking, source-volume prior and recursive kappa CRB`。Git 自身提交 ID 可引用，运行时数据/模型/缓存不允许 hash；只比较显式字段。
- 正式 run **/root/autodl-tmp/TS3/outputs/main-rank-volume-20260918**。
- 实验 tmux **main-rank-volume**。Supervisor Python `/root/TS3/temp/v5-launch/supervisor.py`，最后已确认 PID **148026**（实时核对，不把旧 PID 当永久事实）。实验在容器独立运行，WSL 巡检只监督它。
- supervisor.json/state.json/launch.json/protocol.json/execution.json/execution_control.json 在 run 根；各轮 `rounds/seedK/`，模型状态 `students/<model>/state.json`，结果 `students/<model>/cells/*.json`。
- 日志 `rounds/seedK/chronos-base.log`、`timesfm-2.5.log`，父 `supervisor.log`。
- 临时脚本只放仓库 `temp/`；数据/权重/cache/checkpoint/大日志在数据盘。绝不把它们提交 Git。

## 严格科学与执行约束

- 2 模型（chronos-base、timesfm-2.5）、9 下游任务、13 方法、seed 0/1/2；666 个独立 cell =18 orig+648训练评估。每轮216训练cell。orig只测一次，后续seed引用，不重复计数。
- 任务顺序：time_nem_demand_5t、time_azure_i_cpu_5t、time_darwin_water_15t、time_finland_15t、time_eweld_15t、time_cphl_h、time_nem_price_5t、time_solar_h、time_port_activity_d。
- 方法：rand、orig、less-main；rage-block 与 rage-traj 各 linearSum/zeroClipSum/maxSum/expSum；tsds、lp-mirror。不得自动选定或删除某个RAGE主方法。
- 优先级：**seed → 下游任务 → 该任务全部方法 cell**。当前seed两模型×九任务×12训练方法全部完成训练评测后才进入下一seed。轮内两模型并行，每个模型先做完当前任务全部方法再推进；两模型独立按相同任务顺序推进，当前没有跨模型任务屏障，整轮seed屏障覆盖两模型。
- N16284、C512/H64/D1；CRB cover/error/split 2%/1%/1%；训练 batch/microbatch32/32、优化器等全部不变。Chronos epochs2、LR.001、wd0、linear schedule_epochs6；TimesFM epochs2.625、LR3e-5、wd.01、cosine schedule_epochs6；AdamW betas .9/.999 eps1e-8 clip1。
- 源侧按合法窗口起点数量先验；参考轨迹1/N、参考块(1/N)×本块起点/所属轨迹起点。参考权重只在得分阶段。Max=max(alpha*cos)、Linear=sum(alpha*cos)、ReLU=sum(alpha*relu(cos))、Exp=sum(alpha*exp(cos/1.0))。
- 代表取原始梯度含范数的平均/插值，仅匹配时归一化。LESS均匀rounded-linspace锚点，Traj使用CRB锚点。块取端点梯度几何中点插值，绝无额外BP。
- CRB递归最小kappa、并列最右、实际floor中点、估计几何中点、严格耗尽预算。完全无P10/资格筛选/调度RNG。
- LESS/RAGE只取对象分数排序及精确并列，冻结 F_T(x)=(1-exp(-x/T))/(1-exp(-1/T))，**T=1/(2ln4)**。每对象等宽排名区间积分质量；并列组区间质量组内等分。再乘源窗口量并归一化一次。前半80%约束在源先验之前；奇数/边界并列可偏离。没有最终Softmax。LP-Mirror原生T.05、TSDS原生分布不改。
- BP只能在真实锚点资产准备；评分/插值不能新增BP；缺资产报错。梯度projection4096/seed42、nativeBF16缓存，字段校验和单写者锁、原子写。

## 执行优化及已否决项

- GPU RTX PRO6000 Blackwell Server Edition约96GiB。初始配置student_parallelism2；Chronos gradient_batch8、gradient_workers4，**独立原生单窗口BP**，禁止batch均值或未经验证的vmap；TimesFM已验证原生batch16、worker1，超过16不允许。两者inference_batch64、cpu_threads4、generation eager、memory_fraction .40。执行参数见父/子run execution.json。
- 自定义supervisor通过seed_rounds.run_rounds(...,runner=runner)载入父execution.json再run_suite，所以恢复要沿用 `temp/v5-launch/supervisor.py`，不能随手改用默认入口丢掉执行优化。禁止绕过seed屏障。
- Chronos持久梯度pool资产准备期间有4个spawn进程，资产完成后释放是正常现象。
- 真实BF16 Chronos CUDA Graph比eager快约1.36倍，但2048 tokens有68个不同，**已否决**；不得启用。报告temp/v5-launch/graph-check.json。
- reference_scores使用bounded FP64 CUDA pairtiles，驻留源矩阵，CPU归约，不能构建全参考×源矩阵。Azure约14万参考锚点，块评分每聚合约10分钟是计算，不是必然卡住。不同阶段GPU负载会变。不能为堆高GPU百分比改变科学配方/训练batch/精度。
- MPS已有server，不随意结束。任务queue等其他进程不归本任务管理。
- 原始梯度cache来自 `/root/autodl-tmp/TS3/outputs/main-shared-ten-20260916-211/grad_cache`，父grad_cache symlink；仅通过字段一致性复用raw逐窗口梯度，V5 CRB/表示/分数/概率重建，V4旧选集/完成cell不复用。
- 用户要求保留历史产物/checkpoint；不进行额外清理。冻结protocol已有 `checkpoint_retention=completed_cell_cleanup`，worker会在metrics落盘后自动清理本轮已完成训练权重。这是既有冻结行为，本交接没有授权额外删除或悄悄改变保留策略。磁盘不足时明确报告用户，不改写系统盘。

## 验证范围及交接瞬间状态

- 冻结版本全仓库 **586 passed,2 skipped**；Chronos/TimesFM真实预训练模型开发smoke各12/12训练方法，共24/24，真实BP/保存加载/训练一步/eval。开发夹具不是效果结论。
- V4用户授权停止，旧产物完整保留，STOPPED_FOR_DEVELOPMENT.json记录；不要恢复。
- 最后完整巡检北京时间 **2026-09-18 22:19:48**：83/666独立cell，65训练+18orig，seed0；Chronos已完成NEM/Azure/Darwin，正在Finland Block-ReLU；TimesFM完成NEM/Azure，正在Darwin Block-Linear。无failed或Traceback/OOM。GPU当时99%，滚动约2分钟均值90.96%，显存约27GiB，数据盘可用209GiB。首次接管必须重新核对实时状态。
- NEM seed0：Chronos最佳Block-Linear相对rand+13.53%、Block-ReLU+11.05%；TimesFM最佳Block-Exp+4.16%。只是单任务/seed结果，不能做全局因果结论。训练seed和下游评估与V4矩阵未变，方法变动。
- TSDS原生FAISS在Darwin/Finland少样本聚类会打印建议更大样本的WARNING，已在native流程正常继续；不得为消除警告改变方法。
- 科学代码仍6d3ead6。当前dirty docs/MAIN.md、docs/Progress.md、AGENTS.md是授权运行/报告约束记录；无科学源码修改。另有早前未追踪docs/experiments/rage_implementation_review_20260917.md，勿纳入/删除。没有Push。

## 每次巡检动作（本WSL单一Codex任务持续负责）

1. SSH核对supervisor/state/当前轮worker状态、tail模型和父日志、实际进程、GPU、磁盘。以实时结果为准；不要凭日志暂时静止判故障，结合stage转换、cell计数、进程活性和时间。
2. 在 `/root/TS3` 执行：
   `/root/miniconda3/envs/ts3/bin/python temp/v5-report/main_report.py`
   `/root/miniconda3/envs/ts3/bin/python temp/v5-report/patrol_record.py`
   前者更新MAIN与results.json、核验18个任务模型表/234方法行和末尾汇总表排序，后者写Progress实测状态。脚本源在temp，不影响科学代码。
3. **所有结果表严格按相对rand从更好到更差降序**，100×(1−方法均值/rand均值)，orig也同排序；缺结果/缺rand末尾；样本sd ddof1、单seed—、同seed配对胜rand。沿用V4任务分节、模型表、末尾汇总，不能混入历史结果。脚本保留用户强制约束，未来不得覆盖丢失。
4. run GPU telemetry每5s采样gpu.csv/telemetry.json，滚动24samples均值；仅报告实测stage负载，不能说全流程始终100%。telemetry停止在supervisor complete/failed，恢复时要检查是否需恢复采样。
5. 故障先定位原因，必要修复验证仅执行层，保留完成cell与checkpoint并从父supervisor续跑；不要重复worker。错误/修复/更改/开始结束必须写Progress。只在原授权边界内操作，不改科学版本。SSH transient重试。恢复前核对tmux与worker.lock，避免双启动。
6. 分任务/模型/seed总结，整轮完成和全矩阵完成给明确结论；当前全部3seeds尚未完成，不宣称稳定优劣。需要科研解释时区分rank概率和源长度加权最终概率；方向代理与真实AdamW更新差异；未计算全真实窗口gradient。
7. 每次巡检结束在WSL当前目录写 `STATUS.json`（至少checked_at、complete、needs_attention、cells、seed、简要摘要）；`complete=true`仅当666独立cell和全部seeds均完成且报告已核验。遇到需要用户决定的不可恢复失败，needs_attention=true，并在最新总结明确操作/原因，不继续危险动作。
8. 输出最后摘要，正常进度简短记录；wrapper每30分钟自动续跑**同一Codex session**，不用你sleep等待下次，不创建新任务/自动化。最后全矩阵完整完成后wrapper停止；不要自动seed3或新batch。

## 接管确认

首次立即完成一轮实时巡检和MAIN刷新；向实验 `docs/Progress.md` 追加用户责任交接、WSL tmux moni、模型gpt-5.6-luna/high、本机heartbeat停用、当前真实状态。写本目录 `ACK.json`：accepted=true、model、reasoning_effort、run、cells、checked_at、session_id（如能获取）。你的首次最后回复需明确“已接管”，并记录真实cell数、状态和下一步，方便原负责人确认后退出。

## 本 WSL 巡检包装器

`python3 -u monitor.py 2>&1 | tee -a monitor.log` 在 `moni:0.0` 运行。包装器持有 monitor.lock，顺序启动 Codex exec / exec resume，固定 Luna/high/自动审批，不并行启动第二巡检者。会话ID文件 session_id.txt；当前会话 `01a0b4e7-1228-75c1-a564-b4a3784688e3`。第一次无代理启动仅发生网络超时，原进程已结束后，经7892代理续跑同一会话；当前wrapper PID18227、Codex PID18229（实时核对）。不要根据旧启动记录杀掉其他进程。每次正常巡检结束10分钟后续跑，短暂失败60秒后重试，连续3次失败停止并记录；需要用户决定时停在needs_attention，完整结束停在complete。latest-summary.md、STATUS.json、ACK.json为交付面向用户的摘要，cycle-*.jsonl和monitor.log用于追溯。不要运行本机或其他巡检自动化。
