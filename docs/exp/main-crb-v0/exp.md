## 基础信息

| 字段 | 内容 |
|---|---|
| 实验代号 | `main-crb-v0` |
| 目标 | 保留历史 crb-histgram 与 TIME1 四臂批次；新增完整 src-256 q16、八任务 ref q8 的逐任务重新选择、训练和四臂评估 |
| 假设 | 端点梯度加和范数越接近 0（抵消越强）的区间，其内部梯度变化越剧烈，值得在同等总预算下优先加密；因此 CRB 的两阶段采样应比严格均匀采样在相同锚点数下得到更多高余弦配对 |
| 文档创建时间 | 2026-11-16 CST |
| 最后修改时间 | 2026-11-16 18:26 CST |
| 实际实验启动时间 | 2026-11-16 02:29:10 CST |
| 当前状态 | q16-src256-eight-task 原四臂及新增 CRB+REFT/UNI+REFT 均于 2026-11-16 18:22:58 CST 完成；40 个训练臂、48 次评估和完成标记均核验。另一个独立补充批次 eight-task-baselines 仍为方案阶段 |
| 继承实验 | 历史 `crb-histgram` 只读复用 `main-ads-v0` 缓存；本次四臂批次复用其 TimerXL 84M LoRA 作为梯度探针，并按方向协议为完整 src-256 与 TIME1 ref 计算新稀疏梯度，CRB/REFT/UNI 重新训练 |

## 模型与训练

- `crb-histgram` 历史批次只读复用 `main-ads-v0` 的缓存；本次四臂批次使用该 LoRA 检查点作为梯度探针，重新计算 q16 src-256 CRB/UNI 锚点和 q8 TIME1 ref 锚点梯度。
- 梯度探针：`outputs/train/main-ads-v0/checkpoint/adapter/`（TimerXL 84M，LoRA `q_proj`/`v_proj`，r=8、alpha=16、dropout=0，训练 2,000 步，seed 0）。底座：`/root/autodl-tmp/pretrained_models/timer-base-84m-84m`。
- 投影：Count-Sketch 4,096 维、seed 0；原始投影梯度以 BF16 资源缓存，每条再拆成 L2 幅值 float32 和单位方向 float16。CRB kappa、方向插值、余弦打分与筛选均只使用方向，幅值仅供审计。
- CRB、REFT、UNI 使用同一 TimerXL 84M 底座和 4,000 步 LoRA recipe；有效 batch=32、micro-batch=8、lr=1e-4、cosine、warmup=3%、seed=0。ORI 直接评估预训练底座，不训练。
- TimerXL 训练窗口沿用 V6 原生损失与 480+96 窗口；TIME1 下游评估沿用滚动配置 `configs/eval/time/v1-rage-ads-time-01.yaml`。本次产物落在 `outputs/grad_cache/main-crb-v0/q16-src256-time1-q8/`、`outputs/selection/main-crb-v0/q16-src256-time1-q8/`、`outputs/train/main-crb-v0/q16-src256-time1-q8/` 与对应 eval 目录。

## 数据处理

- 本次四臂批次的 src 集合定义为完整 `src-256` 池，256 条轨迹、6,382,020 个合法窗口；禁止缩减为 ordinal 0。窗口起点沿用 `TimeSeriesFragment.valid_starts(context=480, prediction=96, stride=1)`。
- TIME1 ref 使用安全历史输入 `data/ref/shard_dataset/v1-crb-redo-12tasks/time-01/references.npz`，对应 `CPHL/15T`，10 段、3,250 个合法窗口，清单标记 `answer_values_included=false`。ref 锚点按 q=8 均匀采样。
- 历史 `crb-histgram` 的矩阵仍使用 ordinal 0/1；那一批的既有结果保持原口径，不代表本次四臂批次的数据范围。
- 本批仅按字段校验轨迹数、窗口数、形状、有限性与完成标记；不执行哈希校验。

## 历史 crb-histgram 核心选择算法（CRB）

以下算法参数只适用于已完成的 `crb-histgram` 历史批次；本次 q16 批次按下文 1:0:1 二分协议执行。

定义（k = 合法窗口位置数，j = 锚点位置序号）：

1. **Cover 阶段（预算 q_cov）**
   - 覆盖间距 `cover_spacing = cover_spacing_ratio * q`，默认 `cover_spacing_ratio = 1.5`。
   - 用 `rage.ads.uniform_anchor_positions(N, cover_spacing)` 生成锚点，含首尾端点，得到 `cover` 锚点集合与相邻锚点间的闭区间。
   - 端点被计入 cover 预算。
2. **Cancellation 排序（抵消强度）**
   - 对每个区间 (left, right) 计算
     `kappa = ||g_left + g_right|| / (||g_left|| + ||g_right||)`；
     端点梯度范数相加为 0 时 `kappa = 1`（无抵消证据）。
   - kappa 缩放不变：0 表示完全抵消，1 表示方向一致。**kappa 越小排名越高**。
   - 并列时先取更宽区间，再取更左区间，完全确定性，不使用随机数。
3. **Split 阶段（预算 q_can）**
   - 分割预算 `split_budget = floor(split_fraction * interval_count)`，默认 `split_fraction = 0.5`。
   - 按 kappa 升序依次选中区间，各取一个中点锚点；不可分割区间（宽度 < 2）跳过且不计入预算。
   - 最终锚点 = cover 锚点 ∪ split 中点锚点，两集合互不重叠。

**预算语义（与用户确认）**：`q` 是最终等效采样间隔（锚点总数 ≈ N/q）。

| 参数 | 默认 | 含义 |
|---|---|---|
| `q` | 32 | 最终等效采样间隔 |
| `cover_spacing_ratio` | 1.5 | `cover_spacing = 1.5 q`，cover 锚点数 ≈ N/(1.5q) |
| `split_fraction` | 0.5 | split 新增锚点数 = 0.5 × 区间数 ≈ N/(3q) |
| 合计 | — | 锚点数 ≈ 2N/(3q) + N/(3q) = N/q，即 cover:split 约 2:1 |

说明：`q_cov`、`q_can` 在预算上对应「各阶段合计为 N/q 的份额」。默认 1.5+0.5 的选择使 cover:split 为 2:1，以保证抵消排序真正决定哪些区间被加密（若两阶段严格对半，split 预算恰等于区间数，会退化为均匀网格）。等价间隔在 source 上实测 7.999/15.998/31.989/63.951（q=8/16/32/64），严格贴合 q。

## 评估方案

- `crb-histgram` 历史批次主指标：BF16 余弦相似度矩阵的精确分数统计（均值、标准差、负值占比、非负占比、>0.05 占比、最小值、最大值）。
- 历史矩阵对照：`uni-budget-matched`，按 CRB 最终逐侧锚点数均匀布点。
- 本次四臂主指标：TIME1 `CPHL/15T` rolling evaluation 的官方任务指标，由 `evaluate.run` 输出 `time_metadata.json` 与对应结果文件。比较 CRB、REFT、ORI、UNI 四臂。
- 辅助产出：100 桶与 256 桶相对频率直方图、CRB 各 q 的选择审计（区间 kappa 排名、被分割/保留区间的平均 kappa）、锚点位置文件（cover/split/最终）便于复核。
- 分数精度：GPU BF16 矩阵乘，结果 clamp 到 [-1,1]，矩阵存 uint8 256 等宽桶 ID；直方图 100 桶用桶中心聚合。精确统计同时保存，不再依赖桶中心近似。
- 评估的 q 取 8/16/32/64；用户指定的默认关注值 32 在列，q=8/16/64 作为尺度对照。

## 本次四臂方向协议

- CRB：每条 src-256 轨迹最终锚点数 `K_i=max(2,ceil(N_i/16))`；覆盖阶段含两端点，剩余预算按 `coverage:error:split=1:0:1` 分给 cover 与 recursive minimum-kappa cancel split。cover 使用最长区间中点二分，cancel 使用端点方向 kappa 最小优先的递归二分。
- UNI：每条轨迹使用与 CRB 完全相同的 `K_i`，在首尾之间等距取锚点。两法最终各 399,044 个 source 锚点。
- ref：TIME1 的 10 段分别按 q=8 均匀锚点，共 410 个。CRB/UNI 各自相邻 source 锚点形成块，每块代表方向为左右锚点单位方向的均值再单位化；ref 相邻 q8 锚点同样形成共享参考方向块。
- source 窗口打分与 ref 聚合：每个 source 合法窗口的方向由相邻 source block 方向按块中心线性插值后单位化；10 段 TIME1 ref 的 410 个 q8 锚点形成 400 个相邻区间块，块方向均为两端单位方向的均值再单位化。用 BF16 矩阵乘计算窗口方向与全部 400 个 TIME1 ref 块方向的 cosine 近似分数，并取最大值作为窗口分数；本批 ref 意见按 max 聚合，不做线性求和，也不另乘 ref 权重。
- 分数处理与筛选：除方向的单位化/插值和 ref 维度 max 聚合外，不对 cosine 分数应用 ReLU、clamp、softmax、温度缩放或 rank 映射。按分数降序从完整 src-256 合法窗口中硬筛前 10%（638,202 / 6,382,020），并列按 src-256 全局窗口序号升序；筛选阈值只决定窗口是否入选。该 10% 预算继承已归档 CRB-vs-uniform TimerXL 协议。
- 训练数据使用：CRB/UNI 将筛选结果写为固定 `training_windows.npz` 训练子集；本批不把 cosine 当作窗口抽样概率、样本权重或 loss 权重，也不按 cosine 做概率采样。BF16 矩阵乘得分转为 float32 后参与排序；`crb-histgram` 历史直方图另有 clamp 到 [-1,1] 的统计步骤，该 clamp 不用于本 q16 筛选。梯度幅值不参与 CRB kappa、方向插值、余弦打分和筛选。
- REFT 直接使用全部 3,250 个安全 ref 窗口训练；ORI 直接评估底座、不训练；CRB/UNI 各用 638,202 个筛选后的 src 窗口训练。四臂统一 4,000 optimizer steps，再滚动评估 TIME1。

## 执行细节

- 四臂方向实验入口：`scripts/exp/main-crb-v0/run_directional_time1.py`。梯度资源复用 `assets.gradient.GradientCache`/`FragmentGradients`；CRB 使用 `src/rage/crb.py::CRBEngineMulti` 的最长区间 cover 二分和 recursive minimum-kappa cancel 二分；训练、评估编排复用 `scripts/tools/run.py`、`scripts/tools/train.py`、`scripts/tools/eval.py`。
- `scripts/tools/train.py` 增加可选 `training.recipe` 支持，以复用 TimerXL 已有的 `TrainingRecipe`（不影响不配置 recipe 的原有流程）。
- 历史直方图入口仍为 `scripts/exp/main-crb-v0/run_main_crb_v0.py`，输出 `outputs/selection/main-crb-v0/crb-histgram-bf16-u8/`，历史 tmux 会话 `main_crb_v0`/`main_crb_v0b`。
- 本次入口：方向资源由 `scripts/exp/main-crb-v0/run_directional_time1.py --stage prepare` 完成；结果队列使用临时编排 `scripts/temp/run_priority_main_crb_v0.py`，复用 `scripts/tools/train.py` 与 `scripts/tools/eval.py`。tmux 会话 `main_crb_q16_time1`。TimerXL 训练及评估运行在 `rage-timerxl`（Transformers 4.40.1），通过 `PYTHONPATH` 加载锁定的 TIME 源码；TIME 环境自带的 Transformers 4.57.1 与 TimerXL 自定义 generation cache 不兼容。队列顺序为 ORI、CRB/UNI 并行、REFT。梯度探针配置 batch=128、batched-VJP=128，NumPy 向量化窗口 gather，OOM 时 NativeProbe 自动折半重试。
- 本次日志写入 `logs/main-crb-v0/` 与 `logs/tools/`；全部大型缓存/窗口/模型/评估产物通过项目 `outputs/` 路径保存，实际落在 AutoDL 数据盘。

## 批次管理

| 批次 | 内容 | 状态 | 主要参数 | 输出 |
|---|---|---|---|---|
| `crb-histgram` | 复用 main-ads-v0 梯度缓存；q=8/16/32/64 四档 CRB(cover+抵消排序+split) 与逐侧预算匹配 uni 的 BF16 余弦矩阵、精确统计、100/256 桶直方图、选择审计 | 完成 | cover_spacing_ratio=1.5、split_fraction=0.5、bins=100、BF16/uint8、ref=ordinal 1、src=ordinal 0 | `outputs/selection/main-crb-v0/crb-histgram-bf16-u8/` |
| q16-grad-budget-time1-ref-q8 | 旧的 ordinal 0 单轨迹预算草案 | 已被 src-256 全池定义取代；历史记录保留在操作日志 | — | 无结果文件 |
| q16-src256-time1-q8-four-arm | 完整 src-256 q16 CRB/UNI 方向梯度、TIME1 q8 ref 方向梯度与 CRB/REFT/ORI/UNI 训练评估 | 完成；14:56:51 REFT 评估结束；四臂及 train/eval 根目录 `.complete` 均核验 | src: 399,044 锚点/方法，CRB cover/split=199,846/199,198；ref: 410 锚点；CRB/UNI 各筛选 638,202 个 src 窗口；训练臂各 4,000 步 | `outputs/grad_cache/main-crb-v0/q16-src256-time1-q8/` 及 selection/train/eval 同名目录 |

## 决策记录

| 项目 | 设置 | 状态 |
|---|---|---|
| 历史直方图继承来源 | `main-ads-v0`，只读复用其训练检查点与两轨迹梯度缓存 | 已由用户确认 |
| 本次梯度探针 | `main-ads-v0` TimerXL LoRA；重新计算 src-256 q16 与 TIME1 q8 稀疏梯度 | 按当前实验配置 |
| 历史 crb-histgram 是否重训/重算 | 否，直接复用 | 已由用户确认 |
| 本次四臂是否重训/计算稀疏梯度 | CRB/REFT/UNI 重训；完整 src-256 q16 与 TIME1 ref q8 重新计算方向资源 | 已由用户要求 |
| 本次 source 范围 | 完整 src-256 池（256 条轨迹） | 已由用户明确更正 |
| 本次 q16 CRB 比例 | coverage:error:split=1:0:1；cover 与 split 均按二分 | 已由用户指定 |
| 本次方向处理 | 幅值与单位方向分别保存；仅方向参与 kappa、插值及余弦筛选 | 已由用户指定 |
| 历史直方图预算语义 | q 为目标等效采样间隔；cover:split 约 2:1 | 已用于 crb-histgram |
| 本次 q16 source 预算 | 每轨 `ceil(N_i/16)`；端点后新增预算按 coverage:error:split=1:0:1 均分 | 已由用户指定 |
| source 余弦筛选比例 | 10%，继承已归档 CRB-vs-uniform TimerXL 协议 | 项目既有方案；用户未单独确认，本批按此配置启动 |
| 训练 recipe | 4,000 步及 TimerXL LoRA recipe 继承 v1-rage-ads-v2 | 项目既有方案；用户未单独确认，本批按此配置启动 |
| 抵消强度定义 | `kappa = ||g_l+g_r||/(||g_l||+||g_r||)`，越小越优先；零分母 kappa=1 | 已确定（沿用 CRB kappa 契约） |
| 并列排序 | 先更宽区间，再更左区间；无随机 | 已确定 |
| 分割粒度 | 每个选中区间加 1 个中点 | 已确定 |
| 评估 q 档位 | 8/16/32/64，默认关注 32 | 已确定 |
| 直方图口径 | 100 桶（桶中心聚合）+ 256 桶原始计数 + 精确分数统计 | 已确定 |
| 预算匹配 uni | 逐侧精确锚点数，端点间均匀 | 已确定 |
| 下游任务 | 唯一启用的 TIME1/CPHL/15T rolling eval；CRB、REFT、ORI、UNI 四臂 | 已由用户要求 |

## q16 src-256 与 TIME1 ref q8 资源核算及队列

- 核算时间：2026-11-16 CST；首次启动于 13:02:44 CST。观察到梯度阶段 GPU 利用率 11%、显存 562 MiB、Python CPU 100%；用户要求并行并提高显卡利用率后，于 13:54 左右中断当前阶段以调优，复用已持久化的 436,582 行/3,405 个 shard 梯度缓存，并于 13:57:18 CST 在同一 tmux 恢复。恢复后不再轮询。操作记录见两份 `docs/logs/` 启动与调优日志。
- source 为完整 `src-256`，256 条轨迹合计 6,382,020 个合法窗口；逐轨预算 `K_i=max(2,ceil(N_i/16))`，每个 source 方法共 399,044 锚点。CRB(1:0:1) 的 cover/error/split 合计为 199,846/0/199,198。UNI 与 CRB 逐轨锚点数完全匹配。
- TIME1 ref 安全输入为 10 段、3,250 个合法窗口；逐段 q=8 均匀锚点合计 410。输入路径 `data/ref/shard_dataset/v1-crb-redo-12tasks/time-01/`，不含答案值。
- 两种 source 方案及 ref 的锚点梯度通过一个追加式 BF16 cache 去重准备；每条唯一梯度另保存幅值 float32 与单位方向 float16。raw projected gradient 保留供审计；本批 CRB kappa、source block direction、线性插值、余弦筛选均不使用幅值。
- 四臂训练/评估按 ORI → CRB 与 UNI 并行 → REFT 排队；模型 TimerXL 84M，LoRA q_proj/v_proj、r=8、alpha=16、dropout=0；训练臂 4,000 步、effective batch=32、micro-batch=8、lr=1e-4、cosine、warmup=3%、seed=0；TIME1 rolling 评估使用 `configs/eval/time/v1-rage-ads-time-01.yaml`。
- 入口：`scripts/exp/main-crb-v0/run_directional_time1.py --stage all`；它先准备 gradient cache、方向/幅值分量、插值打分和训练窗口，再依次调用 `scripts/tools.run`。远程 Conda 环境为 `rage-timerxl`，tmux 会话为 `main_crb_q16_time1`。
- 资源目录：`outputs/grad_cache/main-crb-v0/q16-src256-time1-q8/`、`outputs/selection/main-crb-v0/q16-src256-time1-q8/`、`outputs/train/main-crb-v0/q16-src256-time1-q8/{crb,reft,ori,uni}/`、`outputs/eval/main-crb-v0/q16-src256-time1-q8/{crb,reft,ori,uni}/`；日志位于 `logs/main-crb-v0/` 与 `logs/tools/`。
- 本段定义替代旧 `q16-grad-budget-time1-ref-q8` 单轨迹草案；历史核算日志不覆盖，新增更正和队列日志见 `docs/logs/`。


## 新批次：q16-src256-eight-task

- 目标：对八个已登记任务分别重新计算任务 ref q8 梯度方向，按各任务 ref 对完整 src-256 窗口重新打分和选择，重新训练 CRB/UNI/REFT，并重新评估 ORI/CRB/UNI/REFT。TIME1 也纳入新批次，旧批次结果不并入。
- 安全 ref 任务及资源：FEV09 / hospital_admissions_1D（4,766 个合法窗口、600 个 q8 锚点）；TIME1 / CPHL/15T（3,250、410）；TIME2 / CPHL/30T（4,227、534）；TIME3 / CPHL/H（13,332、1,670）；TIME4 / ECDC_COVID/D（4,131、520）；TIME6 / epf_electricity_price/H（259,085、32,390）；TIME7 / Commodity_Import/M（1,577、199）；TIME8 / Commodity_Production/M（61、8）。输入目录为 data/ref/shard_dataset/v1-crb-redo-12tasks/<task>/，manifest 要求 answer_values_included=false；窗口长度为 480+96。合计 290,429 个 ref 合法窗口、36,331 个 q8 锚点。
- source 为完整 src-256 池，256 条轨迹、6,382,020 个合法窗口。q=16 下 CRB 与 UNI 各 399,044 个 source 锚点，CRB 预算为 1:0:1。复用旧批次中与 ref 无关的 q16 source 锚点、block index 和单位方向资产；八个任务各自重新计算 ref q8 梯度方向、对全部 source 窗口重新打分，并各自新选 top 10%（638,202 个窗口）。不复用旧 task-specific 选择结果、训练窗口或训练检查点。
- 本批新算的 reference 投影梯度分别保存幅值 float32 和单位方向 float16；幅值仅供审计。source 块代表方向取左右锚点单位方向的均值并单位化，块间方向线性插值并单位化；余弦打分和筛选只使用方向。
- 每个任务包含 ORI、CRB、UNI、REFT 四臂。ORI 直接评估 TimerXL 84M 底座，不训练；CRB/UNI 各训练该任务新选的 638,202 个 source 窗口；REFT 用该任务全部 ref 合法历史窗口。三个训练臂均从 TimerXL 84M 底座新训 4,000 步，LoRA q_proj/v_proj，r=8、alpha=16、dropout=0，effective batch=32、micro-batch=8、lr=1e-4、cosine、warmup=3%、seed=0。合计 24 个新训练、32 个评估。
- 复用 scripts/exp/main-crb-v0/prepare_eight_task.py、run_directional_time1.py::score_and_select 及 scripts.tools.run/train/eval。队列入口为 scripts/exp/main-crb-v0/run_eight_task_queue.py --stage all --parallelism 8；远端 Conda Python 为 /root/autodl-tmp/rage/envs/rage-timerxl/bin/python，tmux 会话 main_crb_q16_eight_tasks。梯度、选择、训练评估依次执行，任务阶段最多 8 路并行；GPU 为 RTX 4080 32 GiB。
- 输出：outputs/grad_cache/main-crb-v0/q16-src256-eight-task/、outputs/selection/main-crb-v0/q16-src256-eight-task/<TASK>/、outputs/train/main-crb-v0/q16-src256-eight-task/<TASK>/<ARM>/、outputs/eval/main-crb-v0/q16-src256-eight-task/<TASK>/<ARM>/；日志 logs/main-crb-v0/eight-task-20261116.log。大文件通过项目 outputs/ 路径写入 AutoDL 数据盘。
- 启动时间：2026-11-16 16:21:19 CST。16:21:34 检查时队列处于八项 reference q8 梯度准备阶段，FEV09、TIME1、TIME2、TIME3 已完成，TIME4 正在计算；主进程与资源准备进程仍存活，GPU 利用率 67%、显存 3,396 MiB。尚未进入窗口选择和训练评估，暂无新指标。

| 任务 | ref合法窗口 | q8锚点 | 训练臂 | 评估臂 | 状态 |
|---|---:|---:|---|---|---|
| FEV09 / hospital_admissions_1D | 4,766 | 600 | CRB、UNI、REFT | ORI、CRB、UNI、REFT | 完成：CRB/UNI 选择及 ORI/CRB/UNI/REFT 训练评估 |
| TIME1 / CPHL/15T | 3,250 | 410 | CRB、UNI、REFT | ORI、CRB、UNI、REFT | 完成：CRB/UNI 选择及 ORI/CRB/UNI/REFT 训练评估 |
| TIME2 / CPHL/30T | 4,227 | 534 | CRB、UNI、REFT | ORI、CRB、UNI、REFT | 完成：CRB/UNI 选择及 ORI/CRB/UNI/REFT 训练评估 |
| TIME3 / CPHL/H | 13,332 | 1,670 | CRB、UNI、REFT | ORI、CRB、UNI、REFT | 完成：CRB/UNI 选择及 ORI/CRB/UNI/REFT 训练评估 |
| TIME4 / ECDC_COVID/D | 4,131 | 520 | CRB、UNI、REFT | ORI、CRB、UNI、REFT | 完成：CRB/UNI 选择及 ORI/CRB/UNI/REFT 训练评估 |
| TIME6 / epf_electricity_price/H | 259,085 | 32,390 | CRB、UNI、REFT | ORI、CRB、UNI、REFT | 完成：CRB/UNI 选择及 ORI/CRB/UNI/REFT 训练评估 |
| TIME7 / Commodity_Import/M | 1,577 | 199 | CRB、UNI、REFT | ORI、CRB、UNI、REFT | 完成：CRB/UNI 选择及 ORI/CRB/UNI/REFT 训练评估 |
| TIME8 / Commodity_Production/M | 61 | 8 | CRB、UNI、REFT | ORI、CRB、UNI、REFT | 完成：CRB/UNI 选择及 ORI/CRB/UNI/REFT 训练评估 |


- 队列修复记录：16:23:45 首次资源注册在写入训练配置时退出，原因是 outputs/ 软链接指向数据盘，Path.relative_to(ROOT) 无法处理解析后的真实路径。q8 梯度、方向/幅值分量和各任务 ref 窗口已落盘；训练、评估及 source 窗口选择未开始。已将输出路径配置改为绝对路径，配置加载器支持该格式；重启时复用落盘梯度缓存。


## 补充批次：q16-src256-eight-task-union-reference

- 目标：在八任务四臂已完成结果上，新增 CRB+REFT 与 UNI+REFT 两种训练集合，分别把该任务已选中的 source 窗口与该任务全部安全 ref 窗口共同用于训练，再对相同下游任务评估。
- 数据定义：`crb_reft = concatenate(CRB selected source windows, all task ref windows)`；`uni_reft = concatenate(UNI selected source windows, all task ref windows)`。每任务 source 为 638,202 窗口，ref 使用 manifest 中全部合法历史窗口；不去重。训练器在拼接后的单一窗口数组上按既有随机训练流程取样，因此样本数比例由两部分窗口实际数量决定。
- 训练与评估：从 TimerXL 84M 底座独立初始化，不复用既有臂 checkpoint；LoRA q_proj/v_proj，r=8、alpha=16、dropout=0，4,000 steps，effective batch=32、micro-batch=8、lr=1e-4、cosine、warmup=3%、seed=0。评估沿用各任务原 arm 的评估配置。
- 输出：合并数据写入 `outputs/selection/main-crb-v0/q16-src256-eight-task/<TASK>/{crb_reft,uni_reft}/training_windows.npz`，并记录 `metadata.json`；模型和指标分别写入同名批次 `outputs/train/` 与 `outputs/eval/` 下的新 arm 目录。配置在 `configs/experiments/main-crb-v0/eight-task/<TASK>/`。
- 执行入口：`scripts/exp/main-crb-v0/prepare_union_arms.py` 准备数据与配置；`scripts/exp/main-crb-v0/run_eight_task_queue.py --stage union --parallelism 8` 复用已有流程并跳过已完成旧臂。tmux 会话 `main_crb_union_train`；运行环境 `rage-timerxl`。
- 当前状态：16 个合并窗口集、16 个新增训练和评估均完成；队列于 2026-11-16 18:22:58 CST 成功退出。逐任务 MASE 矩阵见 results.md。
## 补充批次：q16-src256-eight-task-baselines

- 目标：在同一份完整 src-256、同一 TimerXL 训练配方和八个下游任务上，补充 Rand、TSDS、LP-Mirror、GraNd-static、CRB-traj、UNI-traj 六种基线。只新增本批次设计与索引；本次未启动基线选择、训练或评估，也未检查已有队列状态。
- 范围：source 为 256 条轨迹、6,382,020 个合法窗口，窗口长度 480+96、stride=1；任务为 FEV09、TIME1、TIME2、TIME3、TIME4、TIME6、TIME7、TIME8，各自复用本实验已定义的 ref q8 锚点及下游评估配置。
- 预算基准：CRB/UNI q16 的窗口预算比例为 10%，即 638,202 / 6,382,020。基线必须保留各自的抽样单位；轨迹/区间方法按其单位严格取 10% 的向下取整数量，并记录最终展开到窗口后的数量。
- 共同预测模型配方：TimerXL 84M 底座，LoRA q_proj/v_proj，r=8、alpha=16、dropout=0；4,000 steps、effective batch=32、micro-batch=8、lr=1e-4、cosine schedule、warmup=3%、训练 seed=0。评估沿用八任务当前配置。
- 随机数：所有随机选择和训练采样均使用局部 RNG、seed=0，不依赖进程全局 RNG 或任务执行顺序。固定集合的 ID 按全局窗口 ID 排序保存；轨迹方法每次训练抽样时在选中轨迹内均匀抽一个合法窗口。

### 基线方法定义

| 方法 | 匹配/选择单位与规则 | 训练和评估范围 |
|---|---|---|
| Rand | 直接在 src-256 合法窗口全集上用 NumPy PCG64(seed=0) 无放回均匀抽取 floor(0.1×6,382,020)=638,202 个窗口；将抽中的全局窗口 ID 排序保存。训练期间只从该固定子集取样。 | 一个共享预测模型；评估全部八个任务。 |
| TSDS | 沿用已接入的 TSDS 轨迹特征与概率分配链路：冻结 TimerXL probe encoder 的轨迹特征、FAISS IVF 邻居检索、KDE 与原生 OT 概率分配；为控制本实验的特征计算预算，source 特征只使用现有 UNI q16 锚点，目标 ref 特征使用对应任务 q8 锚点。每条轨迹对其锚点 embedding 求均值。保留 native max_k=64（256 条候选时有效值为 25）、kde_k=16、sigma=4、alpha=0.1、c=5；IVF nprobe=10、nlist=floor(sqrt(256))/2=8。按 native 概率保留 top floor(0.1×256)=25 条 source 轨迹，其他轨迹概率置零，再对 25 条轨迹的 native 概率重新归一化。每个训练抽样先按该分布抽轨迹，再在轨迹内均匀抽一个合法窗口；不把概率改为 cosine 分数或温度 softmax。 | source 轨迹特征可跨任务缓存；ref 特征、轨迹概率和预测模型按任务分别生成。八个任务各训练、评估一次。 |
| LP-Mirror | 沿用已接入的 ForwardINF 线性 proxy：按轨迹安全统计量标准化；用全 source pool 训练 channel-independent Linear(480,96) proxy，1 epoch、batch=4,096、Adam lr=0.01。每个任务从 ref 按 stride=8 取最多 4,096 个窗口，seed=0；proxy 在 ref 上适配 1 epoch、batch=256、SGD lr=0.001。对 source 全部 6,382,020 个窗口分块（batch=4,096）计算 loss_orig-loss_adapted，使用已接入的 value mapping、temperature=0.05，按分数取 top 638,202 个窗口。source proxy 可跨任务缓存；ref 适配与 source 全池打分按任务执行。 | 八个任务各自生成一份窗口集、训练一个预测模型并评估该任务；proxy 是选择用辅助模型，不计作预测模型训练臂。 |
| GraNd-static | 复用 UNI 提供的 q16 source 锚点位置，在锚点上计算 seed=0、Count-Sketch 4,096 维投影梯度的 L2 范数。每个相邻锚点区间的分数为两端点范数的算术平均；按区间分数降序取 top floor(0.1×398,788)=39,878 个区间，分数相同时按稳定的全局区间 ID 升序。训练数据为这些完整区间拥有的合法窗口，不拆区间；记录其展开后的窗口数，该数不强行等于 638,202。 | 一个共享预测模型；评估全部八个任务。 |
| CRB-traj | 把匹配单元改为 source trajectory。分别使用 CRB q16 source 锚点上的原始投影梯度，先对每条 source 轨迹的锚点梯度做算术平均，再 L2 归一化；目标 ref 用任务 q8 锚点梯度按同样方法形成轨迹向量。以 source trajectory 对全部目标 ref trajectory 的最大 cosine 为排序分数，保留 top 25 / 256 条 source trajectories；质量仅分配在这 25 条内且按轨迹等分，不在 top 25 内按 cosine 大小继续拉开权重。每次训练抽样按轨迹权重抽一条，再均匀抽该轨迹中的一个合法窗口。 | 按任务 ref 分别选轨迹并训练；八个任务各训练、评估一次。 |
| UNI-traj | 使用与 CRB-traj 相同的 trajectory 匹配、ref q8 表示、max-cosine top-25 支持集及轨迹内随机窗口规则；source 轨迹表示由 UNI q16 锚点生成。选中 25 条轨迹后轨迹间等概率抽样，不按 cosine 值作连续权重。随机窗口抽取使用局部 seed=0。 | 按任务 ref 分别选轨迹并训练；八个任务各训练、评估一次。 |

### 批次与执行记录

| 方法 | 预测模型训练数 | 评估数 | 当前状态 | 实际结果 |
|---|---:|---:|---|---|
| Rand | 1 | 8 | 未启动；方案已记录 | 无指标、无选择清单、无模型和评估文件 |
| TSDS | 8 | 8 | 未启动；原生方法参数与预算适配已记录 | 无指标、无选择清单、无模型和评估文件 |
| LP-Mirror | 8 | 8 | 未启动；proxy/ref 适配及全池打分规则已记录 | 无指标、无选择清单、无模型和评估文件 |
| GraNd-static | 1 | 8 | 未启动；区间定义与 topK 规则已记录 | 无指标、无选择清单、无模型和评估文件 |
| CRB-traj | 8 | 8 | 未启动；轨迹表示、top-25 支持集与抽窗规则已记录 | 无指标、无选择清单、无模型和评估文件 |
| UNI-traj | 8 | 8 | 未启动；轨迹表示、top-25 支持集与抽窗规则已记录 | 无指标、无选择清单、无模型和评估文件 |
| 合计 | 34 | 48 | 全部未启动 | 暂无结果 |

- 训练/评估一致性：Rand、GraNd-static 各只训练一个 source 模型，再用同一 checkpoint 评估八个任务；TSDS、LP-Mirror、CRB-traj、UNI-traj 是 ref 条件化方法，必须按八个任务分别选择数据、训练和评估。训练模型总数为 2 + 4×8 = 34；六种方法各评估八个任务，共 48 个方法-任务评估单元。
- 输出路径规划：固定/轨迹/区间选择清单写入 outputs/selection/main-crb-v0/q16-src256-eight-task-baselines/<METHOD>/；训练产物写入 outputs/train/main-crb-v0/q16-src256-eight-task-baselines/<METHOD>/；评估结果写入 outputs/eval/main-crb-v0/q16-src256-eight-task-baselines/<TASK>/<METHOD>/。这些是规划路径，目前不代表文件已经存在。
- 配置与执行入口：计划放入 configs/experiments/main-crb-v0/q16-src256-eight-task-baselines/ 和 scripts/exp/main-crb-v0/。配置和脚本尚未创建，因此目前没有可执行命令。本批次状态为未启动；需先实现入口并检查配置与已完成批次兼容后，再单独启动。
- 既有 q16-src256-eight-task 与 union-reference 批次的历史状态和产物路径保持原样；本补充不覆盖、不重算既有结果。


## 实际批次：q16-src256-eight-task-resample

- 批次目标：在 FEV09、TIME1、TIME2、TIME3、TIME4、TIME6、TIME7、TIME8 八个任务上，对完整 src-256 合法窗口池使用重采样概率训练，并评估 ORI、CRB、UNI、REFT、rand、TSDS、LP-Mirror、GraNd-static、CRB-traj、UNI-traj。该批次不包含随机 10% 固定子集，也不包含任何按比例截断成 top 10% 或 top 25% 的基线。
- 数据与评估：src-256 共 256 条轨迹、6,382,020 个合法窗口；窗口为 480 个历史点加 96 个预测点，stride=1。八项安全 ref、q8 锚点和评估配置复用 `q16-src256-eight-task` 已验证资源；src q16 锚点、区间块方向及梯度方向/范数复用既有缓存。评估仍按各任务原有配置执行。
- 共同预测模型配方：TimerXL 84M，LoRA 作用于 `q_proj`、`v_proj`，r=8、alpha=16、dropout=0；4,000 optimizer steps，effective batch=32、micro-batch=8、lr=1e-4、cosine、warmup=3%、训练 seed=0。使用本地 NumPy RNG(seed=0) 生成训练抽样，避免全局 RNG 和任务顺序影响结果。
- 重采样实现：训练器接收 `ResampledWindows` 概率载荷，在每个训练 step 按原生单位概率先抽单位，再从该单位拥有的合法窗口均匀抽窗；全池索引与原始 src values 独立保存，避免每个任务复制数 GiB 原始值。窗口、轨迹和区间概率都保存在 `outputs/selection/main-crb-v0/q16-src256-eight-task-resample/`。
- 聚合约定：所有基于 ref 梯度余弦的 CRB/UNI 窗口与轨迹变体统一使用 ref `linearSum`。窗口变体对每个 q8 ref 合法区间块按其拥有的合法起点占比加权后线性相加；轨迹变体对 ref 轨迹表示等权线性相加。归一化常数不改变线性聚合的相对打分。

### 方法组和训练范围

| 方法 | 抽样单位及概率 | 预测模型训练与评估 |
|---|---|---|
| `ori` | 无训练；直接使用 TimerXL 底座 | 0 次训练；八任务各评估一次 |
| `crb` | 复用 CRB q16 source 区间方向，在全 src-256 窗口插值打分；ref 方向采用 `linearSum`；按冻结 rank-mass 曲线对全部窗口形成概率，不做硬截断 | 每任务独立训练和评估，共 8 次训练、8 次评估 |
| `uni` | 与 CRB 相同的全池窗口级打分与概率映射，source 单位方向取 UNI q16 区间方向 | 每任务独立训练和评估，共 8 次训练、8 次评估 |
| `reft` | 使用该任务全部安全 ref 合法历史窗口，沿用普通训练器抽样 | 每任务独立训练和评估，共 8 次训练、8 次评估 |
| `rand` | 在完整 src-256 合法窗口池上均匀训练；不构造 10% 固定子集，不做方法性重加权 | 共享训练一次，同一 checkpoint 评估八任务，共 1 次训练、8 次评估 |
| `TSDS` | 复用已接入的 TimerXL encoder 特征及 IVF/KDE/OT 分配；source 特征取 UNI q16 锚点、ref 特征取各任务 q8 锚点；保留 native max_k=64（256 候选时有效为 25）、kde_k=16、sigma=4、alpha=0.1、c=5、nprobe=10、nlist=8。使用完整 native OT 概率，不加 top-25 人为支持集截断；抽轨迹后在轨迹内均匀抽窗 | 八任务各自生成 ref 条件概率、训练并评估，共 8 次训练、8 次评估 |
| `LP-Mirror` | 沿用 ForwardINF 线性 proxy 和 `loss_orig-loss_adapted`；source proxy 1 epoch、batch=4096、Adam lr=0.01；ref stride=8、最多 4096 窗口，适配 1 epoch、batch=256、SGD lr=0.001；对全池得分用 `value_mass`、temperature=0.05 形成概率，不取 topK | source proxy 缓存跨任务复用；八任务分别 ref 适配、形成概率、训练并评估，共 8 次预测模型训练、8 次评估 |
| `GraNd-static` | UNI q16 锚点相邻区间；区间分数为两端投影梯度 L2 范数均值；以区间中心 Voronoi 边界把全池合法窗口唯一分区；按冻结 rank-mass 和区间窗口数形成全区间概率，不取 top 10% 区间 | 共享训练一次，同一 checkpoint 评估八任务，共 1 次训练、8 次评估 |
| `CRB-traj` | CRB q16 锚点的原始投影梯度按 source 轨迹求均值后 L2 归一化；ref q8 轨迹表示同法构造；`linearSum` cosine 经 rank-mass 形成 256 条轨迹的概率；抽轨迹后均匀抽其合法窗口 | 八任务各自训练和评估，共 8 次训练、8 次评估 |
| `UNI-traj` | 与 CRB-traj 相同，source 轨迹表示改由 UNI q16 锚点构造 | 八任务各自训练和评估，共 8 次训练、8 次评估 |

### 批次数量与预期执行次序

| 阶段 | 方法 | 预测模型训练数 | 评估数 | 实际状态 |
|---|---|---:|---:|---|
| 已执行 | `ori`、`reft` | 8 | 16 | 八任务 `ori` 评估和 `reft` 训练评估均已完成 |
| 已执行 | `rand`、`GraNd-static` | 2 | 16 | 两个共享训练均已完成，八任务评估均已完成 |
| 排队 | `crb`、`uni`、`CRB-traj`、`UNI-traj` | 32 | 32 | q16/q8 资源复用；全池概率生成后按八任务队列训练评估 |
| 排队 | `TSDS`、`LP-Mirror` | 16 | 16 | 等待前一组 task-conditioned 队列完成，再生成 native 概率并训练评估 |
| 合计 | 十个方法组 | 58 | 80 | 前四组共 10 次训练、32 次评估完成；其余 40 次训练、48 次评估未完成 |

`Rand`（大写，固定随机 10% 子集）是上一版未启动方案中的旧定义，已被本批全池均匀 `rand` 替代，不属于本批。此前的 hard top-10% / top-25% 方案只保留为历史草案，不是当前执行配置。

### 执行入口、产物和启动记录

- 共享池与共享方法概率：`scripts/exp/main-crb-v0/prepare_resample_core.py`；概率训练入口扩展在 `scripts/tools/train.py`，基础训练循环复用 `src/rage/experiments/train_eval.py::fine_tune`。
- CRB/UNI/轨迹概率：`scripts/exp/main-crb-v0/prepare_resample_methods.py`；任务训练队列：`scripts/exp/main-crb-v0/run_task_resample.py`。
- TSDS/LP-Mirror native 概率：`scripts/exp/main-crb-v0/prepare_native_resample.py`；队列：`scripts/exp/main-crb-v0/run_native_task_resample.py`。
- 配置根目录：`configs/experiments/main-crb-v0/resample/`。训练、评估、概率载荷分别写入 `outputs/train/main-crb-v0/q16-src256-eight-task-resample/`、`outputs/eval/main-crb-v0/q16-src256-eight-task-resample/`、`outputs/selection/main-crb-v0/q16-src256-eight-task-resample/`。运行日志位于 `logs/main-crb-v0/` 和统一工具日志目录 `logs/tools/`。
- 首批启动时间：2026-11-16 19:26:05 CST。8 个 `ori` 评估及 8 个 `reft` 训练评估完成后，`rand` 与 `GraNd-static` 在 tmux 会话 `main_crb_resample_shared` 中共享训练并评估；随后 CRB/UNI/轨迹选择和 task-conditioned 训练队列由 `main_crb_resample_methods_gpu` 自动接续，TSDS/LP-Mirror 由 `main_crb_resample_native` 在前一队列完成后自动接续。
- 实际产物快照：截至 2026-11-16 19:44:41 CST，`ori`/`reft` 的八任务标记齐全；`rand`/`GraNd-static` 各一份共享训练标记及八项评估已完成。GPU 当时从共享模型阶段空出；CRB/UNI 打分脚本已改为 CUDA 分块内插及向量评分，以减少 CPU 评分造成的 GPU 空闲。

决策状态：方法组、全池重采样、ref `linearSum`、训练/评估范围和 seed=0 已确认；TSDS 与 LP-Mirror 按已有接入链路的本批适配参数执行。若底层 native 实现与上述冻结参数不一致，队列会以脚本日志中的实际参数为准并记录偏差，不把 hard subset 回退为默认实现。


## 排队批次：data-select-005-linear

- 目标：紧接 `q16-src256-eight-task-resample` 完成后，在相同八项任务、TimerXL 训练配方、评估配置和十方法组下运行固定数据选择对照。唯一核心差异是训练仅使用 src-256 合法窗口中边际抽样概率最高的 5%，不再使用重采样概率。src-256 共 6,382,020 个窗口，精确取 319,101 个。ORI 保持底座模型评估，不训练。
- 选择定义：对窗口级载荷，窗口边际概率就是载荷概率；对轨迹/区间级载荷，先将单位概率均匀分配到该单位所拥有的合法窗口，即 `window_probability = unit_probability / legal_window_count(unit)`，再对全部 6,382,020 个窗口排序。取最高的 319,101 个；阈值并列时按全局窗口索引升序补足，输出索引最终升序保存。每种任务条件方法分别使用其对应任务的完整概率载荷。
- rand 与 REFT：rand 的所有窗口概率相同，以 NumPy PCG64(seed=0) 无放回抽取 319,101 窗口；REFT 在对应任务安全 ref 窗口内以 PCG64(seed=0) 固定抽取 `max(1, floor(0.05 × ref_window_count))` 个。两者都是固定训练子集，不做概率重采样。共享 rand 只训练一次。
- ref 聚合：CRB、UNI、CRB-traj、UNI-traj 全部使用 ref `linearSum`。CRB/UNI q8 ref 区间按其合法窗口数权重线性相加；轨迹方向等权线性相加。该规则与更早 hard top-K 批次中的 max 聚合不同。本批复用 `q16-src256-eight-task-resample` 中按 `linearSum` 生成的概率载荷并取其 top 5%；不会复用旧 max 结果。
- 其他方法：GraNd-static 使用 UNI q16 相邻锚点区间梯度范数概率，并按区间概率/拥有窗口数排名，取 top 5% 合法窗口；TSDS 使用完整原生 OT 轨迹概率并转成每窗口边际概率排名；LP-Mirror 使用全池 `value_mass` 概率排名。CRB-traj、UNI-traj 将轨迹概率按轨迹合法窗口数分摊后排名。LP-Mirror/TSDS 不引入额外 cosine 聚合。
- 训练与评估：保持 TimerXL 84M、LoRA q_proj/v_proj、r=8、alpha=16、dropout=0、4,000 步、effective batch=32、micro-batch=8、lr=1e-4、cosine、warmup=3%、training seed=0；八项任务评估配置不变。rand 与 GraNd-static 各训练 1 次并评估 8 次；reft、crb、uni、crb-traj、uni-traj、TSDS、LP-Mirror 各任务独立训练评估，共 56 次；ORI 八任务只评估。因此计划共 58 次训练、80 次评估。
- 选择索引和 ref 子集：`outputs/selection/main-crb-v0/data-select-005-linear/`；训练、评估：`outputs/train/main-crb-v0/data-select-005-linear/`、`outputs/eval/main-crb-v0/data-select-005-linear/`。复用全池只读值/窗口索引 `outputs/selection/main-crb-v0/q16-src256-eight-task-resample/shared/pool/`。配置写入 `configs/experiments/main-crb-v0/data-select-005-linear/`。
- 执行入口：`scripts/exp/main-crb-v0/run_data_select_005_linear.py`。该入口等待前批 `native-task-conditioned.complete`、至少 80 个评估 `.complete` 和 58 个训练 `.complete` 后开始准备固定子集与训练；队列 tmux 会话为 `main_crb_data_select_005_linear`。固定子集通过索引视图按需读取共享窗口池，不复制完整窗口矩阵。训练使用 8 个任务并行；每项任务内按 REFT、CRB、UNI、CRB-traj、UNI-traj、TSDS、LP-Mirror 次序运行。
- 状态：2026-11-16 21:30:44 CST 自动满足前置批次完成条件并开始本批；固定子集生成后，rand 与 GraNd-static 共享训练于 21:30:46 CST 启动。运行环境为 AutoDL `rage-timerxl` Conda 环境。批次 `.complete` 仅在 58 次训练、80 次评估均结束后创建。当前尚无评估指标。


进度记录（2026-11-16 21:46 CST）：已完成 10/58 次训练、32/80 次评估；ORI/rand/GraNd-static/REFT 的八任务阶段结果已写入 `results.md`。其余六种 task-conditioned 方法仍在训练队列，尚无指标。GPU 利用率 99%，显存 7,756 MiB / 32,760 MiB。


中止记录（2026-11-16 22:25 CST）：用户要求停止后，已对八个 task worker 进程组发送 SIGINT；所有训练/评估进程退出，tmux 队列会话结束，GPU 利用率回到 0%。最终完成 42/58 次训练、64/80 项评估、58 份选择资产。完成方法为 ORI、REFT、rand、GraNd-static、CRB、UNI、CRB-traj、UNI-traj 各八任务评估；TSDS 八任务训练均在约 3,000–3,300/4,000 步时中止且未生成完成标记/检查点，LP-Mirror 未开始训练评估。已完成产物保留；批次根目录有 `.stopped` 标记，没有 `.complete`。最终阶段结果见 `results.md`。
