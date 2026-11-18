# main-crb-supp 与 Chronos-2-small 队列 ETA 检查

- 检查时间：2026-11-17 02:12 CST。
- TimerXL：`main-crb-supp` 已完成，run 日志记录 2026-11-16 23:52:59 至 2026-11-17 02:10:30，耗时约 2.3 小时；12/12 个任务、192/192 个评估点完成，队列完成标记存在。
- Chronos-2-small：依赖队列已通过 TimerXL 完成检查，02:10:50 开始 prepare、02:10:53 开始首任务 validate；检查时仍在验证，GPU 利用率约 7%，显存占用约 2.5 GiB。正式 24 任务训练尚未开始，验证通过后由 tmux wrapper 自动运行。
- ETA 估算：TimerXL 12 任务实测 2.3 小时，按 24 任务线性外推约 4.6 小时；新增 12 任务安全历史约有 3.5M 个 q8 reference anchor（由任务合法窗口数近似除以 8），当前验证测得 TimerXL probe 约 185 anchors/s，新增梯度准备单项粗估约 5.2 小时。考虑 Chronos-2-small 模型较小、任务级训练时间可能缩短及数据规模不均，正式 Chronos 队列暂估约 8–12 小时，完成窗口约 2026-11-17 10:30–14:30 CST。最大新增任务 azure2019_D/5T 有约 21.3M 合法窗口，可能是 ETA 的主要误差来源；该 ETA 是资源估算，不是进度条承诺。
- MASE 文档：TimerXL 已退出后重新运行 `scripts/exp/main-crb-supp/apply_mase_results.py`，按原始 TIME/FEV 指标更新 MASE 汇总和全局结果索引。
- 后续：依赖 wrapper 自动完成 validate 并启动 Chronos 正式 run；无须人工轮询。
