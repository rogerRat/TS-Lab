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
