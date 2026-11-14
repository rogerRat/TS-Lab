# v1-rage-ads-v2：ADS-VIC 迭代阈值匹配与基线比较

## 基础信息

| 字段 | 内容 |
|---|---|
| 实验代号 | v1-rage-ads-v2 |
| 目标 | 在 v1 已完成的任务与模型评估协议上，比较 ADS-VIC、同预算均匀锚点、轨迹级均匀锚点、原始预训练模型和 reference-only 训练 |
| 文档创建时间 | 2026-11-14 |
| 最后修改时间 | 2026-11-14 |
| 实际实验启动时间 | 2026-11-14 16:41:56 CST（选择阶段；8 任务并行） |
| 当前状态 | src-256 已物化；ADS-VIC 与基线选择实现已启动；选择完成后由 tmux 自动衔接 12-worker 训练和 rolling 评估 |
| 继承实验 | `v1-rage-ads`；旧批次和指标保持独立历史记录 |

## 继承设置

除本文件明确修改的算法规则外，继承 `docs/exp/v1-rage-ads/exp.md` 的任务、模型和评估设置；source pool 按本版改为 256 条轨迹，不继承 src-96 数量。任务为 FEV 9、TIME 1/2/3/4/6/7/8 rolling 任务；TimerXL 84M；rolling-safe reference；共同梯度检查点、损失、可训练参数与投影；评估定义；运行环境 `rage-timerxl`。

学生模型沿用 v1 配置：TimerXL 84M、BF16 LoRA q_proj/v_proj，r=8、alpha=16、dropout=0；AdamW lr=1e-4、weight decay=0、betas=(0.9,0.999)、eps=1e-8；有效 batch 32、微批次 8、梯度裁剪 1；4000 次优化器更新、cosine 日程、3% warmup、min_lr_ratio=0，每 500 步保存状态。各训练方法的窗口来源遵循本版方法定义；`orig` 不训练。模型优化步数与选择阶段的梯度锚点预算分开统计。

## 方法定义

算法公式与流程以项目根目录 `ALGO.md` 为准。本版方法如下：

| 方法 | 计算与采样 |
|---|---|
| `ads-vic` / ADS-VIC | 匹配所有 `cosine > 0.1` 的配对；精算区间取命中锚点形成的连续内包段，不向外延拓；默认 `q_0=32`、q 间隔每轮缩小 4 倍、`R=2`。 |
| `uni` | ADS-VIC 完成后，按同任务 source/reference 实际使用的唯一锚点数分别精确配平，在完整轨迹池均匀布点；不做阈值驱动的后续精算，后续采样和评估流程一致。 |
| `traj` / `uni-traj` | source/reference 每条轨迹用 `q_0=32` 均匀布点并平均梯度；轨迹级矩阵对全部 reference 取 max，按正分数比例抽 source 轨迹，再在轨迹内均匀抽合法窗口；不筛除轨迹或锚点。 |
| `orig` | 不训练，直接评测原始预训练权重。 |
| `reft` / ref-trained | 只用对应任务的 reference 集合训练，再执行相同 rolling 评测。 |

ADS-VIC 沿用 v1 的 `eta=1`、`h=4`、锚点分数 max 聚合、重叠邻域 max 合并及其余模型/评估设置。v1 的 q=16 间隔参数、Top 20% `rho` 筛选和旧 R 语义不继承。

## 轮次、阈值与布点

- q 表示相邻锚点的间隔。默认 `q_0=32`、倍率 4，故 `q_1=8`；`R=3` 时 `q_2=2`。
- `R` 是包括粗算的总轮数：1 仅粗算；2 粗算一次并精算一次；3 粗算一次并精算两次。默认 `R=2`。
- 对长度 L、间隔 Q 的区间，局部点位为 `{0, L-1} ∪ {kQ-1 | 1≤k≤floor(L/Q)}`，去重后计算。例：`L=13,Q=4` 得到 `{0,3,7,11,12}`。
- 每轮只在阈值命中锚点形成的连续内包区间中布点；区间首尾无条件纳入候选。重复锚点复用，不重复反向传播。
- ADS 配对筛选为严格 `cosine > 0.1`，不再用固定 Top 20%。

## 同预算 uni 规则

ADS-VIC 先按 q 和实际命中区间自然运行，不限制为预设总预算。运行后按任务统计全部轮次中不同锚点位置的并集，source/reference 分别记为 `B_src`、`B_ref`。缓存复用点计入逻辑锚点数，同时另报实际新反向传播数。

`uni` 在完整 source/reference 轨迹池中分别均匀放置恰好 `B_src`、`B_ref` 个唯一锚点，source/reference 分侧匹配。各轨迹首尾强制点计入预算，剩余点按容量比例用最大余数法分配；每轨按等距浮点位置映射到整数起点，间隔可取相邻整数，不要求 q 为整数。若预算不足以覆盖强制端点，则报告不可匹配，不静默超预算。`uni` 不做阈值驱动的后续精算，但用相同的后续得分聚合、窗口概率、训练和评测流程。

该方法是按 ADS-VIC 实际预算事后匹配的控制实验，不会预先固定 ADS-VIC 预算；每个任务的 uni 有效平均间隔允许为非整数；用等距浮点位置映射到整数起点，预算数量精确，不把锚点数近似到整数 q。墙钟开销与锚点预算分开报告。

## 数据、评测、执行与产物

- 数据、任务和评估配置：继承 v1，任何测试目标均不得进入 reference、选择或训练输入。
- 新脚本目录：`scripts/exp/v1-rage-ads-v2/`；选择入口 `run_selection.py`，8 任务并行入口 `run_selection_parallel.sh`，训练评估入口 `train_eval.py`，并行 runner `run_parallel_train_eval.py`，阶段衔接 `run_training_after_selection.sh`。
- 选择输出：`outputs/selection/v1-rage-ads-v2/`；训练输出：`outputs/train/v1-rage-ads-v2/`；评估输出：`outputs/eval/v1-rage-ads-v2/`；日志：`logs/v1-rage-ads-v2/`。
- 当前无执行产物；批次成功后在对应输出目录生成 `.complete`。

## 批次管理

| 批次 | 内容 | 状态 | 输出 |
|---|---|---|---|
| `baseline-design-v2` | ADS-VIC 阈值迭代、q 间隔布点、事后同预算 uni、uni-traj、orig、reft 方法定义 | 设计已冻结 | 仅方法定义；无模型指标 |
| `rolling-baseline-v2` | src-256 上的 ADS-VIC、同预算 uni、uni-traj、orig 与 reference-only 训练；八项 rolling 任务 | 选择阶段运行中；40 个训练/评估单元排队由 tmux 自动执行 | `outputs/selection/v1-rage-ads-v2/`、`outputs/train/v1-rage-ads-v2/`、`outputs/eval/v1-rage-ads-v2/`；指标尚无 |

## Source pool（v2）

- 最终 source pool 为 256 条轨迹，配置名 `src-256`，窗口长度 576；必须完整包含现有 `src-96` 的 96 条轨迹。
- 父池为 `configs/data/src-4096.yaml` 与 `data/src/shard_dataset/src-4096/`。新加入的 160 条从 src-4096 中排除 src-96 已有轨迹后的 4000 条候选中分层抽取。
- 配置配额：对每个 source/subset 配置 c，令 `N4096_c` 为 src-4096 轨迹数、`N96_c` 为 src-96 已有轨迹数、`R_c=N4096_c-N96_c`。新抽取配额按 `160 * R_c / 4000` 计算，先取 floor，再按最大余数法补齐至 160；余数并列按配置键排序。配置内使用 seed 0 无放回抽取，新增轨迹不与 src-96 重复。最终顺序按父池 ordinal 排列并重编号。
- 最终配置数为 `N256_c=N96_c+Nadd_c`。由于必须保留 src-96，而 src-96 对父池中低占比配置有至少一条覆盖，最终 256 条的比例不可能对所有配置都严格等于 src-4096 的 1/16；按剩余父池比例扩展是满足“完整包含 src-96”约束下的配额规则。配置配比须在生成后回读 manifest 报告。
- 计划配置和数据路径：`configs/data/src-256.yaml`、`data/src/shard_dataset/src-256/values.npy`、`offsets.npy`、`manifest.json`。
- 检查状态：2026-11-14 16:37 CST 已按规则物化 `configs/data/src-256.yaml` 与 `data/src/shard_dataset/src-256/`。manifest 回读为 256 条、35 个配置、6,529,220 个点；96 条继承轨迹均按 parent ordinal 和 series_id 核对，offset 长度、总点数、有限值和完成标记均通过字段级检查。

## 执行记录（2026-11-14）

- 环境：`rage-timerxl`；模型 TimerXL 84M；投影维度 4096、seed 42；学生训练继承 4000 更新、有效 batch 32、seed 0。
- 选择启动命令：`bash scripts/exp/v1-rage-ads-v2/run_selection_parallel.sh`，tmux 会话 `v1_rage_ads_v2_selection`；八任务分别起独立 worker。
- 训练评估衔接：`bash scripts/exp/v1-rage-ads-v2/run_training_after_selection.sh`，tmux 会话 `v1_rage_ads_v2_train`；监听选择完成事件后运行 `run_parallel_train_eval.py --workers 12`，共 40 个任务/方法单元。
- 日志：`logs/v1-rage-ads-v2/selection-queue.log`、各任务 `selection-<task>.log` 与 `pipeline.log`。
- 16:41:56 首次并行尝试因窗口起点内部零基、`extract_window` 外部一基不一致而失败，未生成任何选择结果；映射改为 `index + 1` 后于 16:42:40 重启。重启后 GPU 利用率实测 100%，显存使用约 7,820 MiB。

- 16:50 CST 对未完成首轮选择执行逻辑审查后主动中断八个 worker；该轮未产生选择结果。修正稀疏锚点连续区间分组和 uni 重叠梯度复用后，16:51 CST 重启选择与训练事件监听。第二轮进程正在推进源侧粗锚点阶段。详见 `docs/logs/2026-11-14-v1-rage-ads-v2-correction.md`。

- 17:44 巡检后按容器 66.6 GB 内存上限调整并行度：保留三个进行中的 q1 worker，time-01/time-03 暂停并排入最多 2-worker 恢复。选择当前 1/8 完成，GPU 使用率 99%，显存约 2,935 MiB。详细状态见 `docs/logs/2026-11-14-v1-rage-ads-v2-memory-capacity.md`。

- 18:08 检查 cgroup 接近内存上限（64.9/66.6 GB，`oom_kill=0`）；旧 worker 的全量相似度临时矩阵导致峰值过高。已终止三个未完成旧 worker，选择器改为分块计算，恢复队列最多并发 2 项；18:10 内存约 7.7 GB，恢复任务 GPU 使用率约 99%。详见 `docs/logs/2026-11-14-v1-rage-ads-v2-memory-overflow-check.md`。

- 2026-11-14 19:22 容器重启后恢复：保留 FEV09、TIME01、TIME08 三项完成产物，按最多 2-worker 顺序恢复其余五项；monitor 会话已重启，训练/评估会在选择根完成后自动启动。详见 `docs/logs/2026-11-14-v1-rage-ads-v2-container-restart.md`。

- 21:13 CST 选择完成 5/8；TIME04、TIME06 正在运行，TIME07 待运行。详见已完成任务的锚点预算、配对规模和解释边界：`docs/logs/2026-11-14-v1-rage-ads-v2-status-2113.md`。


- 2026-11-14 21:57 巡检：选择完成 6/8；TIME06 进程仍在 CPU 上运行，距最近阶段日志 43.9 分钟，TIME07 待调度。训练/评测尚未启动，GPU 当前空闲；内存约 19.2/66.6 GB，未见 cgroup OOM。详见 `docs/logs/2026-11-14-2157-v1-rage-ads-v2-status.md`。

- 2026-11-15 01:07 巡检：选择完成 7/8；TIME06 已完成，耗时约 4 小时、末轮配对 37,028,337,900 对；TIME07 ADS-VIC 已完成，正在计算 uni source 梯度。训练/评测尚未启动。当前进程已载入旧版 CPU 相似度函数，不能热替换；重启会丢失 TIME07 未落盘状态，本轮保留运行进程。详见 `docs/logs/2026-11-15-0107-v1-rage-ads-v2-status.md`。



- 2026-11-15 01:46：选择全部完成；训练评估恢复为 12-worker tmux runner，完成 13/40 单元、0 个当前失败；概率支持审计 40/40 行完成。修复 reference 窗口 owner 索引与 FEV orig manifest 入口，runner 改为错误单元后继续剩余队列。相似度匹配增加 CUDA 分块路径、显存保护及 CPU fallback，CUDA parity 测试通过。见 docs/logs/2026-11-15-0146-v1-rage-ads-v2-status.md。
