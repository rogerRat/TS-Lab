All agents should never edit this file unless explicitly reqested。

20260916：

## 昨日执行优化与今日正式实验阶段性结果

- 2026-09-15 完成了一轮以压缩实验时间为目标的执行优化：Chronos T5 Base、TimesFM 2.5、Time-MoE-200M 的模型、梯度和评分路径统一采用 BF16；三模型的数值复核均通过，记录中的梯度方向余弦接近 1，最大相对误差不超过约 5.7×10^-6。输出证据：/root/autodl-tmp/TS3/outputs/perf-execution-20260915/numeric-mps-*.json。
- 启用 CUDA MPS 并发后，在同一 30 秒吞吐测试口径下，Chronos 为 18.50→33.21 windows/s（约 1.80×），Time-MoE 为 3.67→19.61 windows/s（约 5.34×），TimesFM 为 10.83→22.59 windows/s（约 2.09×）。这些是执行基准，不等同于正式主表收益。另对 LTSV 做了预算压缩，块数由 33,058 降至 17,361，预估串行加速约 2.14–2.52×、并行加速约 3.16–4.04×；预算压缩后的选择质量仍需单独复核。
- 今日主批为 main-shared-ten-20260916-211：512/64 窗口、共同池等价预算 16,284；主 CRB 配置为 cover/error/split=4%/2%/2%，温度仍为 T=1。当前检查时主批尚未完成，不能把当前 cell 汇总当最终十任务结论。
- Time-MoE 在已完成的 time_nem_demand_5t 阶段性结果中，原模型 seed0 的 MASE 为 1.065605；训练一轮后，RAGS/LESS/TSDS/LP-Mirror/LTSV/Random 分别为 1.253448/1.272419/1.379230/1.208463/1.195594/1.191778。即当前这个任务上所有已完成的微调臂都劣于原模型，且半轮检查点已经显示同方向风险；这支持“Time-MoE 微调未带来收益、选择后恶化”的阻碍判断，但因主批和原模型独立重复仍未全部完成，暂不外推为所有任务的最终结论。
- 为验证温度假设，启动补充批 main-shared-ten-20260916-211-t025，仅比较 rags_t025 与 less_t025，将温度改为 T=0.25，同时把 CRB 预算从 4%/2%/2% 降为 2%/1%/1%。在目前已完成的 TimesFM time_nem_demand_5t 三种子结果中，RAGS 为 0.716362，LESS 为 0.690957；MASE 越低越好，LESS 领先约 0.025405（3.55%）。主批 T=1 的同任务当前均值为 RAGS 0.711761、LESS 0.707079，LESS 也略优于 RAGS。因 T 与预算同时变化，当前证据只能说明 T=0.25/低预算组合没有修复 RAGS，不能单独归因于温度。
- 运行基础设施还出现一次独立风险：2026-09-16 00:38 的 CUDA MPS 服务记录了 fatal GPU error，并触发多个客户端 teardown 和设备 reset；证据在 /root/autodl-tmp/TS3/outputs/perf-ltsv-20260916/incident-003846/mps-server-excerpt.log。该事件作为执行稳定性问题记录，不作为算法效果结论。

当前两个阻碍：

1. Time-MoE 全参数微调在当前训练配方下失效，且 RAGS/LESS 等选择臂相对原模型进一步恶化；需要先区分优化器/学习率、BF16 梯度噪声、MoE 路由与辅助损失、原生自回归评测之间的原因。
2. LESS 在已完成对照中超过 RAGS，且 T=0.25 与低预算联动后仍未扭转；RAGS 的温度映射、梯度分数与下游 MASE 的对应关系需要重新校准。

下一步改进计划：

- 做严格的二维解耦：固定 4%/2%/2% 比较 T=1、0.5、0.25；固定 T=1 再比较 8% 与 4% 预算，保持任务、种子、选集缓存和训练步数一致，不能继续用当前 t025 批同时改变两个变量。
- 对 Time-MoE 先用原模型、Random、RAGS、LESS 的半轮/一轮曲线诊断；同时记录 native loss、router auxiliary loss、梯度范数、专家路由统计和 MASE，优先测试更小学习率、冻结或单独约束 router/共享干路，再决定是否保留全参数微调。
- 对 RAGS 与 LESS 做同预算同温度的选集审计：比较分数熵、实际唯一窗口/重复率、top-decile overlap、任务级梯度余弦与训练后 MASE，先判断是 RAGS 评分失配、温度过尖/过平，还是选择正确但 Time-MoE 微调不稳定；在此之前不冻结新温度，也不宣称 RAGS 已被总体否定。
- CUDA MPS 在正式续跑前增加单独的稳定性复核；若再次触发 fatal GPU error，降低并发或切换无 MPS 基线，确保算法比较不被执行故障污染。

### 20260916 追加：默认温度与 RAGS 门控重构

- 默认温度已调整为 **T=0.25**：当前 shared-ten 主配置的默认实验臂由 `rags/less` 切换为 `rags_t025/less_t025`，并保留旧配置备份 `main_shared_ten_20260916_211.json.pre-t025default-0916` 与 `main_shared_ten_20260916_211.json.pre-t025-0916`。当前默认批同时沿用 cover/error/split=2%/1%/1% 的低预算，因此温度变化与预算变化仍须在后续对照中严格拆开。
- 正在测试 RAGS 的无硬门控消融 `rags_nogate`。原 RAGS 会将参考—源窗口余弦小于等于 0 的方向从该参考行的 softmax 支持集中剔除，并在没有正对齐方向时判为不可行；新逻辑通过 `rags_marginal(..., gating=False)` 保留所有有限余弦，让正、零、负对齐共同进入 `exp(cosine / T)` 的行内归一化，不再直接过滤负对齐样本。审计同时记录正/非正 pair 数、非正质量和实际映射方式。
- 这不是对 RAGS 主算法的静默替换，而是独立的 `rags_nogate` 对照臂；当前正在重新跑实验，尚无下游 MASE 结果，不能提前判断去掉门控是否有效。正式比较需固定模型、任务、种子、预算和 T=0.25，只比较门控前后选集重叠、非正质量占比、唯一/重复曝光及训练后 MASE。

### 20260916 追加：RAGS-Traj-gls 新实验臂

- 新增实验臂 `rags_traj_gls`。当前实现将其映射到 RAGS-Traj 的轨迹聚合路径，并同时关闭硬门控（`gating=False`），因此它不是普通窗口级 RAGS 的别名：先在保留的 RAGS 支持上按轨迹聚合原始梯度，再在轨迹级余弦上让所有有限对齐值参与温度 softmax。
- 它复用 RAGS-Traj 的共享轨迹梯度资产与支持范围，区别于普通 `rags_t025` 的 CRB block 评分；gls 版本额外不丢弃非正对齐轨迹。审计仍应保留正/非正 pair 数、非正质量、近零轨迹以及实际轨迹/窗口覆盖。
- 代码的 `TEMPERATURE_DEFAULT_FORM` 已将 `rags_traj_gls` 纳入默认七臂形态：`rags_t025/random/less_t025/tsds/lp_mirror/ltsv/rags_traj_gls`。当前已检查的 main JSON 仍是旧的六臂 t025 配置，尚未发现该新臂的实验产物；因此需要新 run/新选集命名空间，不能把它与既有 `rags` 或 `rags_t025` 缓存混用。
- 当前仅完成接入/注册，尚无 RAGS-Traj-gls 的下游结果。后续要与 `rags_traj`（有门控）、`rags_gateless`（窗口级无门控）、`rags_t025`/LESS 在同一任务、种子、T=.25、CRB 预算下比较选集重叠、非正质量、训练曝光及 MASE。
> Historical progress note. The legacy TS3-E, alpha, grid-index, and protocol
> descriptions below are not current operating instructions; use the SOP and
> project documentation for current behavior.

0912:

- 最新拍板 15 已落实：N 按点预算 N×576 记账；RAGS 按召回块最大余弦整段交付，保留 CRB 起点域定义，交付完整窗口包络；trajectory_level 重写为均匀预算的真实轨迹平均梯度、整条交付，旧 576 等分段首估值移为 `uniform_segments` 消融。LESS 整轨迹交付；TSDS 保留多项抽样真实副本，打散副本后整条累计至预算，重复实际进入训练采样，不是仅排序权重。唯一点数、重复点数、末单位溢出率分别记录。
- 共享 `RegionWindows` 只打包原始数据段，每批从区域内全部合法起点均匀有放回采样，不预先展开海量滑窗；三学生仍统一 512/64、FP32 原生损失、4 epoch，步数按预算 N 而非合法起点总数。主表、学生 CLI、曲线整单位预算前缀、段级消融和覆盖统计已接通，固定块/窗口返回消融修复多轨迹全局索引偏移。移除旧 TSDS 固定窗副本转换助手；算法内核、LP-Mirror/LTSV 选择方法均未改动。
- 验证：针对性 **68 passed**；最终完整回归 **252 passed、5 skipped、6 个既有失败**。既有失败仍是 FEV/TIME 本地资源缺失和临时配置路径限制，无新增失败。Chronos 官方随机小模型 512/64 完成区域选择、原生训练与四轮检查点；TimesFM/Time-MoE 隔离环境分别 **1 passed / 1 skipped**，包含原生模型加载、反传、区域训练与新进程 CLI。RAGS 冷热源块复用的交付区域/数据/分数一致，六阶段 dry-run、compileall 和 diff 检查通过。dev 未跑正式实验。日志位于 TMP 的 `outputs/point-budget-0912-*.log`。
- 当前交接以 `docs/dev/PointBudgetDelivery0912.md` 为准，旧固定窗文档已标为历史。选集 schema `point-budget-regions-v1`、task manifest v2；旧批必须新建 run，不能沿用旧选集、学生 cell 或检查点。用户要求完成 Push 后通知 tmux `zcode` 拉取并重启，清理范围仅当前确认的错误实验批次，不涉及数据、权重或其他结果。

- 后续按最新实验计划拍板 11/12 完成 LESS/TSDS 轨迹化：两侧共用均匀播散抽窗、原始特征求均值、RAGS 三段配置预算配平。LESS 使用两侧裸 SGD 梯度、单状态、全参数投影、逐参考轨迹余弦聚合与不相交轨迹切窗；不将 Adam/LoRA/warm-up 列为待补。TSDS 候选及 KDE 检索恢复上游 FAISS IVF（nlist=sqrt(T)//2、nprobe=10），保留 OT 内核、多项抽样重复轨迹；按用户随后确认的规则，各副本切窗后从完整副本池均匀无放回抽 N 项，重复身份实际进入训练，不去重、不补入未抽中轨迹。参数仍为本实验冻结值 64/16/4/.1/5，明确不是上游 config 默认值。
- 源侧轨迹 FP32 均值缓存跨任务复用，旧 stride-16 梯度缓存/全池逐窗嵌入缓存退役；主表与曲线接入同一密度参数，旧 LESS/TSDS 选集和学生 cell 需新 run，启动仅一次轻量版本/预算检查，不删除旧结果，不反复 hash。共享线程控制覆盖 FAISS OpenMP；主环境新增 faiss-cpu 1.12.0。LP-Mirror/LTSV 的方法代码、四个外部 algorithms.py 均未改动。三学生统一 512/64（最新拍板 10-S1）已与工作器一致，长上下文不在本批实施。
- 本批最终全量回归 **244 passed、5 skipped、6 个既有失败**（FEV/TIME 数据缺失及既有临时配置路径限制），无新增失败。新增真实 IVF/KDE 上游对照、轨迹采样/均值/预算与冷热缓存、重复副本虚拟索引对实际展开逐项一致、重复选集落盘、三个学生路由完整接收重复行与四轮计数、随机初始化官方 T5+Chronos tokenizer 512/64 的 LESS/TSDS 冷热完整选择及权重不变。六阶段 dry-run、compileall、diff 检查和 pip check 通过。测试不等于正式实验；未运行 exp 或下载正式数据/权重。
- 交接见 `docs/dev/TrajectoryBaselines0912.md`；`BaselineAudit`、三学生文档、执行优化历史说明同步更新。上一批 2.84× LESS/约持平 TSDS 是旧窗口级方法的历史测量，不能作为这次轨迹化后的性能结论。新旧方法不再纳入同一个执行等价基准；保留 LP-Mirror/LTSV 的旧新对照。

- 选择阶段执行优化验收完成，按用户要求集中为一个稳定版本提交。外部基线仅接入共享窗口/参考提取、显存自适应梯度与推理批次、Count-Sketch 融合、CPU 线程控制、官方 T5 forward 的 CUDA graph 预测；四个 `algorithms.py` 未修改，候选数、参考样本、Adam 更新和评分逻辑不变。RAGS 专用优化为双堆风险调度、多轨迹批处理、固定映射复用、源 anchors 及时释放与跨任务内存复用。正式精度、训练 batch 和实验预算不变，没有增加反复 hash 或自动重启。
- dev RTX 2080 Ti 上以随机初始化的 46M 模型、512/64、1,154 池窗口对照 `cde56ec`：无选集缓存的稳定重算 LESS 7.963→2.804 秒（2.84×）、LTSV 9.546→2.792 秒（3.42×），TSDS/LP-Mirror 基本持平；四臂选集 ID、窗口与逻辑工作量完全一致，投影梯度相对 L2 差 `4.67e-7`。RAGS 合成确定性梯度场整链路约 2.2×、纯 CRB 调度 5.8–10.7×；均验证选集/分数或 anchors/块/计数一致。不能把这些微测直接外推到正式池或 PRO6000，LP-Mirror 首次初始化差异不计为稳定收益。
- 最终核心全量 225 passed、5 skipped、6 个既有失败，无新增失败；缺失 FEV/TIME 数据及既有临时配置路径问题未掩盖或改成跳过。新增真实 T5 CUDA 图与官方 greedy token/score 对照、输入与权重更新、FP32 四检查点训练/逐轮评测/权重恢复测试均通过；Time-MoE、TimesFM 两个隔离环境各 1 passed/1 skipped（CPU 小模型及新进程 CLI）。六阶段 dry-run、编译和 diff 检查通过。
- 执行参数、性能边界、缓存与测量口径见 `docs/dev/SelectionExecution.md`，三学生部署说明同步更新，默认开启共享加速。输出留在 TMP 的 `outputs/execution-0912/`，不提交模型、数据或实验产物。未操作正式 exp 作业；交付前两次只读 SSH 连接被远端关闭，因此最新正式进度未再次确认。现有续跑仍以完整臂缓存为单位，不自动中断在途臂来应用更新。
- 主实验新增轻量 `SelectionMetrics` 组件：编排统一测量全部七个选择臂，original 单列零成本，按调用追加到 `main/selection_metrics.jsonl`。记录实际墙钟秒数、PyTorch 峰值 allocated 显存、设备与 protocol，区分 `cold`、中间产物复用 `warm`、最终选集读取 `selection_cache_hit`；三学生共享一次记录，续跑不覆盖首次值，不改写选集中的历史生成耗时。RAGS/TSDS/LTSV 补充真实缓存状态，LESS/LP-Mirror 复用已有状态；支持 selection-only。显存包含常驻探针，测量边界与跨 run 比较规则见 `docs/dev/SelectionMetrics.md`。针对性 26 passed（包含真实 CUDA 临时张量峰值与逐臂重置、三学生编排及续跑）；补充 LTSV 实际落盘/缓存读取一致性后，该测试文件 9 passed。dev 仅小规模测试，未启动正式实验。

0911:

- 第六批交付完成：代码提交 **`5ef83df` 已 Push 到 `origin/rags`**，exp 应拉取该分支；对应开发、测试与正式执行边界已同步到本地参考文档 `_prog-0907.md`、`_exp-0907.md` §7 和 `_models-integration-0911.md` §2。正式批仍由用户在 exp 启动，本次没有正式实验结果；下方“用户要求完成后 Commit 和 Push”现已完成。参考文档位于本地非 Git 目录，不属于 TS3 仓库的 Push 内容。
- 第六批：按更新后的实验文档拍板 9 接入 Time-MoE-200M，替换实验中的 Sundial；主表为 Chronos-small / TimesFM 2.5 / Time-MoE 三学生，共用七个选择臂的一次选择，各有 original 参照。学生分环境并行、任务逐个完成；FP32 全参数、4 epoch、每 epoch 存权重并全部评测，只用 MASE 比较。选择成本共享记一次，训练/评测成本分模型记录；增加分模型主表/相对 Random 曲线及 `--resume` 复用原批配置。用户要求科研代码从简：已去掉新加的选集/考卷/manifest 重复哈希和协议防御检查，仅在新批 RUN.md 记录一次依赖锁哈希。
- Time-MoE 直接使用官方源码 `67460eb4` + Transformers 4.40.1，权重固定 `794591bf`。新版源码 + 4.57.6 在真实反传中发生 HuberLoss/ForCausalLMLoss 属性冲突，故采用官方原训练组合，不修改官方模型。原生 576 点序列训练对应 575 个移位标签，保留多 horizon Huber 和 router loss；预测只输入 512 点历史。TimesFM 强制 FP32，取消按 GPU 自动启用 BF16。Sundial / TimesFM 3.0 仅保留历史适配器、不进入主表。
- 评测修正：MASE 分母改为每个预测时点之前的完整历史（FEV `y_past`），不再包含考试未来；零尺度单列 MAE，SQL 不进入主表及检查点曲线。旧结果需重新评测，不能直接混合。
- 第六批验证：核心全量 212 passed、5 skipped、6 failed，六个失败均为原有资源/路径约束问题，无新增失败。两个隔离环境分别通过官方小模型 512/64 FP32 训练、原生预测、四检查点全评测、参数恢复及实际新进程 `model_train` CLI；七臂一次选择→三学生 24 格的小规模主表及续跑测试通过。三环境安装验证和 pip check 通过，六阶段 dry-run 与静态检查通过。未运行正式数据实验。
- exp 交接说明见 `docs/dev/MultiStudent0911.md`。三学生主表全部保留权重约需 1,371.8 GiB（1,512 份），不含数据和缓存；Time-MoE-200M 总参数为 453,196,800，不能按名称中的激活参数估算存储。dev 无需补齐正式资源，不下载正式权重/数据，不自动关机。用户要求完成后 Commit 和 Push，正式批由用户在 exp 启动。
- 用户明确当前 `dev` 容器仅做代码开发和测试，不进行正式实验。正式数据、Chronos-small 权重与检查点存储是未来实验环境的准备项，不是当前开发阻塞；当前仅运行单元/集成测试、小模型冒烟和 dry-run，不启动正式批、不为正式批下载大规模数据或扩容。
- 第五批实验协议与部署准备：依据当前实际路径 `001-iclr17/rags-md/_exp-0907.md` 的 0911 决策 8/D2（用户所写 `001-iclr27` 路径不存在），六任务主表统一为 4 exposure epochs、batch=32、lr=1e-5、N=4096，累计 128/256/384/512 步保存并评测全部权重；主结果取最后一轮，不按测试集挑最好一轮。数据量曲线同步使用相同 epoch 规则，新增检查点/成本/数据量曲线输出与重叠覆盖统计；未将文档中的待定项自动纳入实验。
- 修复评测未来信息泄漏：旧 evaluator 向解码器传入真实未来标签；现在仅由历史自回归生成，再读取 greedy prefix 下的条件 bin CDF。此结果不是采样边际分位数，不能与旧指标混用，正式实验解释前须重新做真实数据 original/random sanity。协议及局限详见 `docs/dev/ExperimentDeployment0911.md`。
- 第五批全量回归：204 passed、3 skipped、6 failed；六项与前批一致（FEV 数据缺失两项、TIME checkout 缺失两项、既有临时配置路径约束两项），无新增失败。Chronos 官方 dummy 模型实际保存/评测四个检查点并验证未来标签隔离与权重恢复；这是接口冒烟，不是预训练 Chronos-small 性能验证。六阶段 CLI dry-run 完成，正式训练尚未启动。
- 部署预检已写入 `outputs/deployment-0911.json`：当前容器缺少源 shard、六任务 reference bundle 和 Chronos-small 权重；仅主表 504 份 FP32 检查点约需 86.4 GiB，预检时数据盘余量约 41.2 GiB。这些仅供未来正式实验环境规划，不要求在 dev 补齐；未下载大规模数据、未启用自动关机。历史输出和模型未删除。
- 第四批外部基线分包：每个方法单独位于 `baselines/{less,tsds,lp_mirror,ltsv}/`，分别拥有 `algorithms.py`、`runner.py`、`contract.py` 和 `select` 接口。顶层只保留统一派发、注册表、兼容导出和公共工具。26 项针对性测试通过；与清理提交逐函数 AST 比对，15 个执行函数及 6 个数值函数逻辑完全未变，四个数值包均可不加载 Torch 独立导入。
- 第四批最终全量回归：191 passed、3 skipped、6 failed；失败项与第三批相同，无新增失败。最新实验文档接入作为下一批工作，尚未启动正式训练。
- 第三批清理：删除旧 OT/KDE、candidate uniform/vote/ratio、特征网格、transfer cache 与旧训练调度器；同时删除对应描述符/目录构建、旧分析编排、113 份旧方法配置及 28 个专属测试文件。保留 RAGS、四个外部基线及数据/模型/梯度/评测基础设施；TSDS 自身的 OT/KDE 与 RAGS 的 2×2 消融矩阵不属于被删除的旧方法。`scripts.assets` 改为纯 fetch/load/dedup 入口，`scripts.run` 指向 RAGS。详见 `docs/dev/LegacyRemoval.md`。
- 第三批第一轮全量验证：188 passed、3 skipped、6 failed，6 项均来自清理前失败集合。移除旧方法测试不计作修复；删除内容可从 `582b515` 恢复，数据、模型和历史实验产物未删除。另新增“旧方法不得重新出现”和数据 CLI 路由检查。
- 第二批目录整理：外部基线移至 `baselines/{algorithms,contracts,runners}.py`；11 个自有实现目录移至 `src/`，保留原 Python 包名；`scripts/` 保留编排与 CLI，并新增 `python -m scripts.exp.rags {run,main_table,curves}`。新增 `pyproject.toml` editable 安装与 pytest 路径配置，安装器同步登记源码包，迁移后的路径计算统一复用 `configs.paths`。数据、模型与实验产物未移动；新布局说明见 `docs/dev/CodeLayout.md`。
- 第二批迁移前基准：第一批扩大回归为 86 passed、2 skipped；完整测试为 358 passed、3 skipped、19 failed（完整迁移前实测，不将其当作全绿）。19 项涉及缺失 FEV/TIME 数据或脚本、日志目录、既有配置路径与采样/训练缓存约束。迁移后将逐项对照失败集合。
- 第二批验证完成：全量为 362 passed、3 skipped、19 failed；与迁移前逐项比较，新增失败 0、消除失败 0，新增 4 项布局/路径/CLI/仓库外导入测试全部通过。三个环境 editable 安装完成，三个 RAGS CLI 帮助、静态编译和安装脚本语法检查通过。另以 Chronos 官方仓库的 dummy 模型完成 LP-Mirror/LTSV 真实 tokenizer/生成接口冒烟，选集合法且模型参数恢复；这不是正式预训练模型的性能实验。
- 外部基线修复（第一批）：按“复刻方法，不复刻实验设置”执行。LP-Mirror 恢复完整源集 Adam 训练和完整目标集 SGD 适配，去掉固定 50 步有放回抽样，改为安全轨迹统计标准化；LTSV 修正先点平均再窗平均，使用论文 Adam 与历史输入自回归预测 MSE，保留 Chronos 原生训练损失作为模型接口适配。两种时序基线保留原生 top-k（允许窗口重叠），N 不再代表相同的新数据覆盖量。旧结果缓存使用新版本名称，避免沿用修复前结果。
- 第一批针对性测试：`test_time_series_baselines.py`、`test_baseline_algorithms.py`、`test_baseline_contracts.py` 共 17 passed。新增覆盖 LP-Mirror 独立训练/评分对照、完整遍历、LTSV 非均匀覆盖公式、无未来输入预测 MSE、长块、原生 top-k 与异常后的模型恢复。此结果是小规模算法/集成验证，不是论文规模实验或真实预训练模型性能验证。
- 在 `rags` 分支完成项目名称统一：Python 包与配置目录改为 `rags/` 和 `configs/rags/`，模块导入、公开类名、实验 arm、运行标识、脚本及文档同步使用 `rags`/`RAGS`。Git 跟踪的文件路径和文本中已无旧拼写。
- 将原先独立维护的 RAGS 测试全部并入 `tests/`，测试文件统一采用 `test_rags*.py` 命名，删除分离的测试目录。
- RAGS 相关回归测试（排除需要外部 TIME checkout 的用例）结果为 95 passed、2 skipped；包静态编译、`rags`/`RAGSConfig`/`RAGSPipeline` 导入和 Git diff 格式检查均通过。
- TIME bench 中 2 个用例因缺少固定 commit 的 `third_parties/TIME` checkout 未能执行。完整测试结果为 343 passed、2 skipped、17 failed；其余失败来自现有数据、目录和脚本缺失或既有路径约束，未显示由本次名称迁移引入的回归。
- 将诊断归档从单层日期缩写目录迁移到 `diag/0908/`，并同步修正诊断脚本、日志中的硬编码输出路径；后续诊断可继续按 `diag/MMDD/` 分目录保存。

0829:

补充更正（短/长 catalog 命名核对）：首次成功写出的 1024 结果曾同时扫描到 20 个长名称目录和 20 个旧短名称目录，manifest 因此记录为 40 个 catalog、26 个 eligible；这不是 FEV 原始下载产生的两份数据，而是旧流程按 `fev-autogluon%2Ffev_datasets%3A<dataset_config>` 创建的别名与当前 bundle-derived `fev-%2F...` 长名称并存。当前脚本已统一使用长名称解析，禁止短名称回退，并删除了 1024 根目录下 20 个短 catalog 目录；后续结果必须以 20-task catalog 集合为准。

本次进一步将 cache builder、reference diagnostics、PCA population 和 FEV asset diagnostics 接入同一严格长名称 API；遇到短目录会直接失败，不再执行历史 profile-nested glob fallback。当前仓库中的 42 个失效临时训练 YAML（仅引用已不存在的旧短 catalog）已删除，避免继续作为可执行配置被误用；archive 中的历史记录保留不变。

本日完成/启动的 reason 分析与代码整合：将 `scripts/exp/exp_filter_grad.py` 并入 `scripts/exp/main_reason.py`，新增 data-only 入口，可直接针对 assets 做参考集异质性、候选支持宽度、参考窗口梯度一致性，以及 cu-r1 candidate 相对 random source 的 reference-gradient 对齐分析；不依赖主实验训练或评估产物。

本次数据侧分析使用 profile-10b 的 FEV catalog，训练窗口协议为 context=512、prediction=64、stride=1；代码沿用现有 reference task 最小完整窗口 gate（window_length=576），不满足的 reference task 自动排除并写入分析 manifest。当前已确认完整的是 analysis-segment length/stride=1024/1024：20 个唯一任务中 13 个通过 gate、7 个被 gate；该目录同时存在两套重复 catalog header，因此日志曾出现 14 条 warning，但对应的仍是这 7 个唯一任务。被排除者包括 uk_covid_nation_1D、us_consumption_1Y、world_co2_emissions、M_DENSE_1D、hospital_admissions_1W、rossmann_1D/1W 等短任务。未将无完整 asset 的任务强行纳入计算。

1024/1024 data-only reason run 已启动，run_id=`reason-fev20-segl1024s1024-20260829`，结果目录为 `outputs/exp/reason-fev20-segl1024s1024-20260829`。模型侧没有训练 seed；为复现原 `exp_filter_grad` 的多 checkpoint 协议，filter-gradient 使用 warmup checkpoint seeds=0/3/8/42，source/candidate 每任务最多采样 1000 segments、projection=4096、batch=32；主 reason 的参考梯度一致性使用每 catalog 64 windows、projection=1024、batch=4，device=auto。

后续对 assets 的独立分析按三套版本分别保存：`length/stride=256/256`、`512/512`、`1024/1024`，每套使用独立 run_id，结果不混合。256/256 与 512/512 目前由其他 agent 生成，优先使用已完整生成的 1024/1024 版本；本记录时 256/256 尚在生成，512/512 尚未确认完整。
日志同时已优化：main.py 与 main_reason.py 现在统一输出 reference preflight 的 discovered/eligible/gated/window_length 汇总，并列出唯一 gated task；reason 分析会先去除重复 catalog header。
补充记录：今日停止了无效旧主实验 chronos46m-main-v2-s1（含训练与 watchdog），并删除对应无效结果目录 /root/autodl-tmp/TS3/outputs/exp/run-chronos46m-main-v2-s1（约 88 MB）。

主实验协议已修改为支持在配置中使用百分比 checkpoint 步数：整数表示绝对 optimizer step，(0,1] 小数表示 training_steps 的比例。默认 schedule/evaluation 均为 [0.1, 0.2, 0.5, 0.8]；对于 10K steps，即在 1K、2K、5K、8K 保存 checkpoint，并在最后对四个 checkpoint 全部评估。该修改及测试已提交，commit=b0a5c4e；比例解析测试 21 passed，模板解析验证得到 [1000, 2000, 5000, 8000]。完整 pytest 受远端缺少 autogluon、nbformat 第三方测试依赖影响，未能在该环境全量收集。

新的 AnalysisSegment 消融主实验全部固定 seed=0，ChronosSmall 46M from-scratch、batch size=128、10K optimizer steps，目标为全部 FEV task（当前 runner 会自动移除不能提供至少一个完整最小参考窗口的任务，并记录跳过原因），使用现有四方法 rand、cu-r0.1、cu-r1、cu-r10。三组 run_id 明确编码变量：
- chronos46m-anaseg-l256s256-s0，catalog=profile-10b-segl256s256
- chronos46m-anaseg-l512s512-s0，catalog=profile-10b-segl512s512
- chronos46m-anaseg-l1024s1024-s0，catalog=profile-10b-segl1024s1024，直接复用已处理数据

已生成三份独立 run 配置及串行启动脚本/看门狗；单 GPU 按 l256s256、l512s512、l1024s1024 顺序运行，避免并发争抢显存。数据处理截至本补充记录：l256s256 已完成 load/dedup，正在对 840209 条 source trajectory 做 feature extraction；l512s512 尚未开始；l1024s1024 已存在并复用。临时处理脚本与日志均放在 temp/，脚本结束时会恢复生产配置为 l1024s1024。


0819:

完成 9m-kmeans-static-cu 实验（group A：task-05/15/09，四方法 rand/cu-k1/cu-k4/cu-k8，单 seed、20K、六 checkpoint），检验用 KMeans 分解参考集分簇各自筛选支持集，能否解决"ref 异质性高→支持集大→退化"。结论：失败。task-05（ETT_15T，1862 seg，高异质性，典型退化任务）cu-k1 仍比 rand 差约 18% SQL，分解后更糟：k4 +73%、k8 +30%；task-15（solar，97 seg）cu-k1 −28%，但收益随 k 单调衰减至与 rand 无差别；task-09（hospital，24 seg，最低异质）cu-k1 −16%，k4 −20%，为唯一 k4 胜 k1 的任务。机制判断：每簇各自跑 OT 后支持集并集≥单一支持，k 越大并集越接近全候选域，等于放大而非收缩曝光；簇间均匀分配与 7-23 稀疏 grid 均匀曝光问题同构。修复方向应是显式控制支持并集大小（全局配额、收紧 threshold/neighbor_count），而非分簇取并集。局限：单 seed、单一 budget、checkpoint 波动大（task-15 cu-k1 16K 最优后 20K 反弹）。
B组（03/04/10）k1均改善，分簇无增益，总体结论不变。

绝对 M 支持集宽度实验（9m-support-size-cu，policy=candidate-uniform-size：静态 OT 检索按质量排名取 top-M、集内均匀采样，已去掉 OT 加权与聚类）在 task-05 完成四配置、task-15 完成 M32 后中止，结果归档于 archive/run-9m-support-size-cu-abort-20260819（ckpt 与 cache 已清理）。task-05 SQL@10K 随宽度单调改善：M32=5.60（塌缩，分位曲线全平）、M128=2.52、M512=2.01、M4096=1.52，仍不及 rand@20K=1.34；task-15 M32=2.14，比 rand@8K=2.07 略差。未见"支持集太宽导致退化"，反而"过窄即塌缩"；cu-k1 在 task-05 的历史退化更可能来自支持集收缩与 OT 质量集中，而非宽度本身。放弃继续绝对 M 扫描的原因：各 ref 任务的 cu 支持集规模天然相差两个数量级（6K–327K），同一硬 M 在不同任务对应完全不同的相对覆盖（M4096 在 task-09 为 68%、在 task-05 仅 1.3%），硬数值无法跨任务比较，也难以在单任务内探测完整宽度响应。改为相对 M：以 vanilla cu 支持集为基准取比例 r，r≤1 自支持集内随机抽样、r>1 自 OT 邻域候选池随机补齐，仅选 task-05/15 两任务细致分析（新实验 9m-support-ratio-cu）。
相对 M 实验（9m-support-ratio-cu，policy=candidate-uniform-ratio，r=0.1/0.5/1/2/5+rand，ChronosTiny from scratch 10K 步仅评 10K ckpt；两 run 目录附 rand-equivalent-ratio.md：task-05 中 M14.9≈rand、task-15 中 M437≈rand）。task-05 SQL@10K：rand=1.42，M0.1=1.76、M0.5=1.51、M1=1.24、M2=1.21（最优，−15%）、M5=1.32，倒 U 形，静态 cu 族首次明确超过 rand；而同一采样器 0816 cu-k256（2.14）、0819 cu-k1（1.58）均输 rand，说明此前"cu 在 task-05 退化"很大程度是 run 间方差与 checkpoint 波动，并非方法本身。task-15：rand=2.09，M0.1=1.64（−21%）最优，M0.5=1.83、M1=1.87、M2=1.73、M5=1.87，越窄越好。两任务宽度响应方向相反（05 最优 r≈2、15 最优 r≤0.1），证实最优宽度是任务相关量，应以支持集相对规模而非绝对 M 表达。无方法触发 rand 等价跳过（r=5 最大覆盖 33.5%/1.1%）。局限：单 seed、单 budget，M1/M2 优势幅度待多 seed 确认。

0818:
测试改进之后的dynamic方法，使用一个seg内部的前后两半+变化+对应分析扩展到51D，对任务帮助不大。方法有小幅度的提升，但是仍然无法解决退化问题。因此怀疑最本质的原因是，描述的是边际特征，并不能等价于模型对Loss的降低。
进行了系统性的分析，挑选了1000个segment对其进行分析，计算它的16D51D特征，以及它在Grok Transformer的预训练checkpoint上，中间层的最后一层产生的梯度，还有此checkpoint给出的loss。分析结果表示，16D或者51D特征之间的距离跟梯度之间的余弦距离几乎无任何相关性。16D特征比51D特征对loss距离有相对更明显的相关性，但相关性仍然不是很强，大约在0.3左右。

补充归档（outputs/analysis/）：①chronos-tiny-probe-20260818：1000 段、全配对 499500 对。16D 特征距离 vs loss 距离 Pearson≈0.36、Spearman≈0.26；51D 仅 Spearman≈0.21；两种特征 vs 梯度余弦方向相关均≈0.003，确认特征距离与梯度方向无关。②gradient-audit-20task-{c095,c100}-20260818-v2：20 任务中 10 个可审计（其余缺≥32 个合法 512+64 参考窗口），用预训练 Chronos-Tiny 逐窗口梯度审计预测 d-ot 是否胜同支持 d-cu。ΔA vs ΔP 汇总 Spearman 仅 +0.135（p=0.57），符号一致 10/20；c095 ρ=−0.03、c100 +0.35 均不显著；早期 5-draw 的 ρ=0.73 系抽样噪声（单 draw ΔA std≈0.17）。绝对对齐度 struct≈candidate（均 0.11）且远高于纯随机（0.02）；struct vs 纯随机 ΔA 13/20 为正、均值 +0.07。结论：梯度对齐来自候选剪枝（dynamic top-k）而非 OT 权重；审计暂不能替代训练作 go/no-go 信号，需补配对纯随机训练基线验证剪枝收益可预测性。

完成修改后 DynamicTransition d-ot 实验：18 个已完成 task（00–17），单 seed、20K from-scratch，含 1K/2K/4K/8K/16K/20K 六个 checkpoint；比较 d-ot-c095/c100、同支持 d-candidate-uniform，并与 `archive/run-9m-six-methods-20260816` 的 rand、s-ot-k64/k256/k1024 配对。直接结论：相对 rand，d-ot 的 SQL 平均改善约 7.1%–7.3%、WQL 约 6.6%–6.8%，但仍有 task 级回退；相对 s-ot-k256 小幅占优，相对较强的 s-ot-k64 仅大致持平，不能声称普遍优于 s-ot。confidence=0.95/1.0 的影响随指标和 checkpoint 改变，无稳定单调方向。与同支持 candidate-uniform 的差异较小，当前收益更可能主要来自 transition-aware support，而非 OT allocation。结论受单 seed、18 个完成 task 及当前/旧 catalog 差异限制；task18 缺失、task19 未完成。

风险判断：
| 条件 | 退化风险 |
|---|---|
| 参考集特征异质性高 | 上升，但非单调 |
| 支持域变宽且有效精度未提高 | 上升 |
| 真实近邻、未来一致覆盖高 | 下降 |
| 支持域过窄、遗漏关键模式 | 上升 |
| OT 质量过度集中 | 进一步上升 |

0816：为定位当前方法的适用边界，以及收益/回退的真正来源，启动第一次机制试验：20 个 FEV 任务、每任务 6 种策略（Rand、S-OT k64/k256/k1024、复用 k256 正质量支持但候选内均匀的 candidate-uniform，以及 lambda=0.5 混合）。以下仅是已完成任务在 20K、单 seed 下的初步判断。

结论：candidate-uniform 与完整 OT 共享逐点相同的候选支持。支持集筛选在 14 个可配对任务中 11 个优于 Rand，但仍有 3 个回退（最大为 task05 的 WQL +36.6%），故并非普遍安全。完整 OT 相对同支持的均匀采样既有增益也有回退：task17 额外改善 16.4%，task04 额外恶化 47.4%。这说明质量分配可能放大收益，也可能带来严重负迁移；目前尚未显示稳定、普适的额外优势。两种方法是否适用的共同线索是候选域“窄度”：候选支持覆盖的原始质量、参考有效 segment 数越大，回退越多（当前 Spearman rho 约 +0.74/+0.63）。但在同一支持内应信任均匀采样还是完整 OT，ESS 比、熵降低、最大质量等集中指标均不能解释增益/回退分布。

后续：完成 20 任务并做多 seed；将参考异质性、候选/实际窗口覆盖、历史未来一致性和局部验证风险纳入未来盲诊断，探索何时只用支持集、何时收缩或启用完整 OT。

0815：task17/18的9M from-scratch六策略实验已完成。两任务中OT均优于Rand，但最优support依任务而变：17为k256，18为k64；k1024无额外收益，lambda=0.5显著稀释收益。历史task18在预训练模型后期曾负迁移，说明transfer sign依赖初始化与训练阶段，不能宣称OT普遍安全。下一步按偶数/奇数配置跑完20个task，再统一审计。

0814: 更换使用的参数量为46M的模型，之前的tiny是9M。使用预训练模型进行全参数微调后训练，并且针对单个任务逐个进行，仍然使用原本的10k设置-发现Loss基本不下降，并且退火方法基本上是两者之间的一个直线。而且OT方法效果比小模型更差。

0813：FEV dynamic OT 消融结果归档于 `archive/fev-ablation-1b-segl512s512-20260813`。实验代码定位 commit：`9ac014c`（`dev/d-seg`，`feat: schedule experiment commands in parallel`）。统计当前所有已有结果：task-00–06 四方法完整，task-07 缺 ot，task-08 仅 rand。相对 rand，ot 在 70 个配对 checkpoint 中 50 个更优，平均 SQL 改善约 6.20%，但任务差异明显；d-ot 相对 ot 平均退化约 15.03%，暂无明显改善；anneal 相对 d-ot 在 80 个 checkpoint 中 53 个更优，平均改善约 4.80%，有一定修复作用，但相对 ot 仍落后约 8.54%。建议主对照为 rand、ot、d-ot-anneal，d-ot 保留为消融组。

0811
昨天晚上启动训练进程，卡在CPU阶段，发现是因为修改参数导致计算量翻了16倍到4倍。CPU已无法承担这么大的计算量，因此今天重构所有非串行依赖计算，全部移动到GPU进行计算，同时使用外部缓存保存计算结果。

0806:
验收了一下昨天跑的20个任务单独提取reference，然后训练的实验结果。
发现每个不同的任务表现差异很大，Task 17表现就特别好，然后Task 0和1表现也还不错，但有的就是纯负提升。
在昨天的训练脚本里面已经把checkpoint上传到了rogerspyke/TS3FEVModels0805，然后把评估结果上传到r-ot-kde-1b-1-10k-20-seperate-tasks-0805。
今天的主要任务就去找一下成功的、效果好的这些任务到底有什么特征，然后尝试去总结和改进方法。


7-23:
进行了最小实验验证，发现Ours方法打不过Random和MetaData，其中Random最强。检查之后发现有几个问题。第一是评估侧的问题，评估侧没有利用好Chronos的概率预测，Top K设置为一。第二是最主要问题是当前存在很多及其稀疏的网格。这些网格内的样本量甚至不足32个，所以如果强制每一个batch必须来自同个网格的话，将会对这些边缘结构造成极强的曝光，反而让结构极其不均衡。长尾轨迹池与稀疏 grid 交互，使旧版 TS3-E 过度重复少量轨迹和窗口，带来来源身份集中、窗口覆盖下降及潜在梯度相关性。
| 方法 | 轨迹 | scope/grid 行为 | batch distinct trajectories | batch 内重复 slot |
| --- | ---: | --- | ---: | ---: |
| Random | 8,723/8,723 | 按长度/点质量加权；两条超长序列约占 19% slot | 27.701 | 13.43% |
| Metadata | 8,723/8,723 | 43 个 scope 近似均匀，每个约 2.33% | 28.204 | 11.86% |
| TS3-E（legacy-only） | 8,656/8,723 | 477 grids 近似均匀；旧版每 batch 100% 同 grid | 5.835（中位数 3） | 81.77% |

（legacy-only）TS3-E 还有 27.87% batch 只有一条轨迹，47.6% batch 同时来自一个 scope。477 个 grid 中 429 个少于 32 条轨迹，129 个是 singleton；占用率仅 11.65%。这些数字描述旧版网格约束，不是当前逐槽 sampler 的性质。

这已引起接下来的改造：首先，采样器现在输入需要采样n条轨迹的时候，不再会强制n条轨迹一定来自同个grid。现在其行为等价于独立采样n次，而不是一次给出同个grid中的n个轨迹。同时，这也显示出一个问题，那就是当前的纯粹结构网格公平可能过于极端，因为大部分情况下很难避免某些少量数据单独共享一个网格。对应的方法，要么是把这些极端网格按照一定的算法分配到L1距离最近的几个网格，将人口强行归纳。要么就是通过调整alpha参数，让采样器均衡一下每个网格中的人口（轨迹数据点数），决定采样概率。其中第一个方法是相对机械的，因为它需要显式定义规则。

另外一个它的改进是将数据量从1%提升到10%，然后重新进行处理，这就导致如果现在把所有的数据索引，包括网格索引都放进内存的话，可能会导致 cPU内存溢出，因此设计了新的方案，使用网格索引联动。 Catalog index进行数据轨迹的索引，然后将训练时读取的网格index保存到外存中。对于Random算法，它直接读取Catalog Index。对于另外两个依赖网格的算法，它直接读取Grid Index，然后间接联动到Catalog Index进行对应索引。

当前发生的另外问题是，由于原始存储的格式是Parquet，而要获得一个窗口，必须先获取整条轨迹，因此会导致大量超长轨迹的反复解压和来回搬运。现在还书面的瓶颈并非在GPU方向传播，反而是在CPU来回搬运数据上。因此现在需要进行新的工作，是数据上的数据资产格式重构。使用快速读取的二进制数据配合Arrow索引进行针对性的手术刀式打击，获取所需要的精确窗口。现在主干调度部分在调度之前需要适配器发起请求，提供必要的切分信息，然后再由核心部分切出数据直接返回给适配器，适配器消费后再返回消费报告。

当前进度：
- 实验进行最小实验验证有效果
- 论文进行同步更新
- 论文补充图表
- 论文梳理问题和结构采用推荐的叙事方式：具体问题，要做什么，具体挑战，如何解决挑战并给出方法，在问题和挑战并存的条件下完成任务达到表现预期结果
- 明确问题的具体定义
- 预期贡献的表格

已经做了的：
明确整个项目流程为三个阶段，并且每一个阶段之间必须用规定的通信协议进行通信。

最终补充（2026-08-29，bin shard 复用协议与主实验启动）：

- 发现并修复了三个 source Catalog 共享同一个 trajectory.arrow 导致后生成 Catalog 覆盖先生成 Catalog 的问题。现在 profile-10b-segl256s256、profile-10b-segl512s512、profile-10b-segl1024s1024 分别使用独立 trajectory 表；三个协议的 source/ref Catalog 都保持协议独立，不能跨协议共用 Catalog。
- 将 Load 身份接口固化为 load/metadata.json（ts3.load-identity.v1），记录 dataset_id、batch_id、raw fingerprint、scope、选择参数和 shard 规格。新增 assets/dataset_processing/load_reuse.py：仅当 profile/batch/raw fingerprint、scope、Load 选择参数及 shard 规格全部匹配时，才在目标 bundle 自己的 load/shards 目录下建立相对符号链接；目标仍保留自己的 index.arrow 和 meta.arrow。身份缺失或不匹配时明确输出 BIN SHARD REUSE FALLBACK 并回退到 raw 重建。source Load、FEV reference Load 和 scripts/assets.py 均已接入，控制台与配置的日志文件都会记录检查、链接或 fallback。
- 当前 source bundle 已规范化：l256 保留唯一实体 bin shard；l512/l1024 各自保留 index/meta，仅链接 load/shards 到同一 profile-10b raw batch。l256 与 l1024 的 189 个 bin 文件逐文件内容比较一致后，已删除 l1024 重复实体副本。当前三个 source bundle 均有独立身份元数据并通过 bundle 校验。
- 当前 20 个 FEV reference asset 的三个协议版本共 60 个 bundle 均通过校验；l256/l512 的 bin shard 已按资产链接到 l1024 对应实体，三个协议仍各自保留独立 bundle、Feature 和 Catalog。
- 复用协议提交为 3c71677。新增复用/身份测试及既有 Load、Reference、Dedup、Catalog、Bundle reader 回归测试共 45 passed。真实 source Load CLI 演练成功从同 profile-10b sibling bundle 复用 bin shards，日志确认没有重新读取 raw。
- 主实验配置仍为 ChronosSmall 46M from-scratch、batch=128、10,000 optimizer steps、seed=0、四方法 rand/cu-r0.1/cu-r1/cu-r10；三组 run_id 为 chronos46m-anaseg-l256s256-s0、chronos46m-anaseg-l512s512-s0、chronos46m-anaseg-l1024s1024-s0。三组 dry-run 均为 discovered=20、eligible=13、gated=7，均使用各自 source/ref Catalog。
- 串行主实验 driver 于 16:47:14 启动，顺序为 l256s256 -> l512s512 -> l1024s1024。l256 rand 已在 16:50:32 达到 600/10000 steps，GPU 利用率约 99%、显存约 23.6 GiB，训练无错误并持续推进；首轮 torch compile 的 CPU 启动阶段已结束。
- watchdog 于 16:50:48 在 tmux chronos46m-anaseg-watchdog 启动，监控 driver session、DONE/FAILED/RUNNING 标记及 GPU 状态。driver 当前保持 RUNNING，主实验尚未完成，后续由串行脚本自动继续三个协议。

- 16:58 update: the shared long-catalog resolver import was fixed and the AnaSeg launcher adapter was promoted from temp/run_anaseg_main.py to scripts/exp/run_anaseg_main.py; commit fa25987. The active l256s256 seed-0 run remains uninterrupted and reached 4500/10000 optimizer steps with the watchdog reporting about 99% GPU utilization and no training errors.

0830:
- 定位到 0829 主实验 GPU 释放的直接原因：l256s256 四个方法训练和四个 checkpoint 的评估均已完成，但 train_meta 为空时无条件执行 loss 绘图，visualize.loss_step_line 找不到 losses.csv，导致 scheduler 误将绘图错误作为整个 run 的硬失败；l512/l1024 因 driver 停止而未启动。
- 修复为 loss 绘图按 train_meta 优先、train 回退搜索；两处都没有 losses.csv 才报错；绘图命令改为 optional，失败只记录 warning。默认 prune_training_artifacts=true，训练后归档轻量 metadata，并清除模型 checkpoint、transfer cache 和 run-local training_audit；同时删除空 cache run 父目录。
- 确认之前错误使用了自写 watch_chronos46m_anaseg.sh，它只检测 FAILED 后退出而不会关闭容器。现改为兼容包装器，实际统一调用 scripts/dock_shut.sh --mode term；当前重启的 driver 已由官方 dock_shut 监控。
- 相关提交：8939e7f、66eff06、3f457d7、ecd61a8；相关测试 28 passed。重启后 l256 已完成收尾清理并进入 l512 seed0，l512 训练已进入 GPU 高利用率阶段。
