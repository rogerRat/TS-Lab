# 2026-11-17 main-crb-supp 进度检查

- 检查时间：2026-11-17 00:50（Asia/Shanghai）。服务器：tsl（AutoDL），项目：/root/TS-Lab。
- 实验：main-crb-supp。队列 tmux main-crb-supp 仍在运行，主进程 PID 295964，启动约 58 分钟。
- 总体进度：计划 12 个任务、192 个评估点；前 5 个任务已完成（80 个评估点）：Crypto/D、Supply_Chain_Customer/D、Supply_Chain_Location/D、NE_China_Wind/H、Coastal_T_S/20T。当前为第 6 个任务 time-australia-solar-h（Australia_Solar/H）。
- 当前任务：E1–E5 五臂并行训练；每臂已完成 step-1000 快照，训练日志最新约 1200/4000 step。step-2000、step-4000 尚未完成；Australia 的正式评估点尚未开始。E0 Rand 在本任务五臂流程后评估。
- 资源：GPU 0 为 RTX 4080，利用率 99%，显存使用约 7712/32760 MiB。未观察到异常或闲置。
- 证据：logs/main-crb-supp/run_20261116235259.log 最新队列事件显示 00:46:59 开始 Australia_Solar/H；五个训练臂日志位于 /root/autodl-tmp/rage/logs/tools/train/，截至 00:50:55 均记录 step 1200/4000；tmux、进程与 nvidia-smi 状态已核对。
- 文档同步：更新 docs/exp/main-crb-supp/exp.md 的最后记录时间和当前状态；更新 docs/EXP.md、docs/RESULTS.md 中 Australia 批次状态。无正式评估结果，本次未填写指标。
- 未解决事项：队列继续运行；后续由队列按完成批次写入指标。此次仅检查一次，没有设置自动巡检或轮询。
