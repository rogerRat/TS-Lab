# v1-crb-redo 操作日志

最后修改日期：2026-11-13

## 2026-11-13 选择耗时口径更正

复查远程 `logs/v1-crb-redo/` 中 32 个 traj/block 选择命令的运行摘要。此前结果文档把队列任务之间的间隔误累计为选择耗时，造成每批 53–59 分钟的错误表述。现已作废该累计值，并在 `docs/results/v1-crb-redo.md` 列出 8 个任务 × 4 个算法 arm 的逐项选择耗时。选择运行时间按命令运行摘要记录，包含特征准备/读取、评分和 RR 选样，不含训练评估；TIME 6 block 日志只显示约 1.4 分钟。选择索引文件大小按实际文件记录。

## 2026-11-13 全量评测指标补录

按用户要求明确 MASE 为主指标，同时必须逐任务、逐方法汇报评测配置实际产生的所有指标。已在 `docs/exp/v1-crb-redo.md` 记录 FEV/TIME 指标字段、任务内汇总和缺失值披露规则；在 `docs/results/v1-crb-redo.md` 补齐 RR traj/block 四个 arm 的 FEV 及 TIME 全量指标；在 `docs/results/v1-crb-redo-timerxl-baselines.md` 补齐 full 与 rand01 两个既有基线的全部 FEV/TIME 指标。待完成 win 批次的全量指标将在结果实际生成后追加，不预填。

## 2026-11-13 11:17 CST 运行状态查询

GPU 0（RTX 4080）由 `scripts/exp/v1-crb-redo/rr_select_win.py --family base-uni --task fev-09` 占用，属于 Uniform win RR 选样。tmux 会话为 `v1-crb-redo-win-queue`；日志进度为 1,536/4,766 个 ref 窗口，ETA 约 9.1 小时。查询时 GPU 利用率约 50%、显存占用 1,338 MiB；进程 CPU 约 100%、RSS 约 43.5 GiB。

## 2026-11-13 11:36 CST win RR 资源占用核查

检查 `rr_select_win.py`：当前阶段是 Uniform win 的 ref-to-source 候选相似度计算，不是训练。源方向 memmap 为 2,666,312×4096 FP32，文件大小 43,684,855,936 bytes（约 40.7 GiB）；每个 ref 窗口扫描全量 source windows，逐块保留 top-568 候选。该过程涉及约 52.3 万亿个向量维度乘加项（约 104.6 TFLOP）和约 208 TB 的累计源特征读取量。显存占用约 1.3 GiB；工作负载主要由重复全量扫描及内存到 GPU 的块传输造成。进程保持运行。

## 2026-11-13 RR 参考端权重代码核查

对照用户已确认的“参考块按合法起点数加权”核查 `rr_features.py`、`rr_select.py`、`rr_win_features.py`、`rr_select_win.py` 与 `rr_selection.py`。block 特征产物保存了 ref block 的 `counts`，但 `rr_select.py` 只读取 `reference_units.npz` 的 `directions`，轮询函数对每一行 ref 评分恰好取一次；ref block 的合法起点数没有进入 block RR 轮次/权重。traj 的单条 ref 轨迹代表在轨迹内部按区间合法窗口数加权，但不同 ref 轨迹在 RR 队列仍各一票。source block 的 `counts` 只决定该块被选中后计入窗口预算的数量，不改该块的分数或排序。当前运行的 win RR 以每个合法 ref 窗口作为一个队列项，故每个合法起点各一票，较长片段按合法窗口数自然获得更多轮次。该核查仅记录实现现状，未中断当前 win 队列或重跑已完成 block 批次。

## 2026-11-13 12:35 CST BF16 窗口评分矩阵切换

按更新后的实验决策，先修改 `docs/exp/v1-crb-redo/exp.md`，再替换 `scripts/exp/v1-crb-redo/rr_select_win.py`：评分向量点积使用 BF16 输入/输出，完整 ref×src 窗口分数一次写成 row-major BF16 矩阵；矩阵完整存在时可在选样中断后复用。选样结果通过预算、范围及唯一性校验并写入 manifest 后删除矩阵，再写 `.complete`。旧 FP32 top-k 进程于 12:29 CST 中断，未产生选择标记或触发训练。12:34 CST 在 `v1-crb-redo-win-queue` 重启队列；当前 Uniform FEV 9 正在生成 4,766×2,666,312 的 BF16 评分矩阵，预期大小 25,415,285,984 bytes（约 23.66 GiB）。

## 2026-11-13 12:37 CST 新 win 队列状态

 正在执行修订后的 Uniform win FEV 9 选样。BF16 完整评分矩阵已生成，程序正在从矩阵提取每个 ref 的候选排序；日志进度约 2,752/4,766，ETA 约 1.1 分钟。选样校验后队列将自动启动该任务的训练评估，并按序继续剩余 Uniform win 和 CRB win 任务。

## 2026-11-13 12:37 CST 新 win 队列状态

`v1-crb-redo-win-queue` 正在执行修订后的 Uniform win FEV 9 选样。BF16 完整评分矩阵已生成，程序正在从矩阵提取每个 ref 的候选排序；日志进度约 2,752/4,766，ETA 约 1.1 分钟。选样校验后队列将自动启动该任务的训练评估，并按序继续剩余 Uniform win 和 CRB win 任务。

## 2026-11-13 13:57 CST win 队列结果核查

核验新 win 队列的完成标记、选择 manifest、评估文件和日志。Uniform win 的 FEV 9、TIME CPHL/15T、CPHL/30T 三项选择/训练/评估均完成，MASE 分别为 0.670380、1.208172、1.025688；全部已配置评测指标已补入实验结果文档。TIME CPHL/H 的 BF16 评分矩阵选样完成（13,332 ref windows；峰值 71,094,543,168 bytes，选择耗时 1,317.83 s；索引 2,133,176 bytes），矩阵清理完成；训练日志截至 13:57 CST 为 100/4,000 步。所查日志未见实验报错。

## 2026-11-13 14:00 CST 部分 win 结果入档

将新 Uniform win 已完成的 FEV 9、TIME CPHL/15T、CPHL/30T 全部评测指标、逐任务 MASE、索引大小、BF16 矩阵峰值大小和选择耗时写入 `docs/exp/v1-crb-redo/results.md`；TIME CPHL/H 标记为选样完成、训练进行中（截至 13:57 CST 为 100/4,000 步）。同步 `docs/exp/v1-crb-redo/exp.md`、`docs/EXP.md` 和 `docs/RESULTS.md` 的批次状态。BF16 矩阵在验证后已删除，索引与 manifest 保留。

## 2026-11-13 14:06 CST win 选择与训练耗时对比

核对四个已完成 Uniform win 任务的选择 manifest 和训练评估日志：选择耗时分别为 15.08、9.15、12.71、21.96 分钟；训练入口的总耗时（含 4,000 步训练、导出、评估及清理）分别为 6.9、8.1、7.7、8.1 分钟。因此选择耗时即使对照完整训练评估入口也更长。选择日志的 `stage=score-matrix-progress` 明确记录 `device=cuda`，评分矩阵用 BF16 CUDA matmul 计算；候选排序读取矩阵后由 NumPy 在 CPU 上执行。该差异解释了 GPU 只负责评分阶段，选择总耗时还包括候选排名和 RR 轮询。
