# v1-rage-ads TIME08 恢复与批次完成

- 检查时间：2026-11-14 13:16–13:22 CST。
- 实验代号 / 批次：`v1-rage-ads` / `rolling-eight-task`。
- 初查状态：选择与审计完成；12-worker训练评估首轮有28/32单元完成，四个未完成单元均为TIME08；队列状态1，GPU空闲。
- 失败证据：四个TIME08训练日志均记录到step 4000，随后滚动评估退出，输出仅含启动元数据。单独以`--resume`复现的错误为TimerXL `ValueError: Input length must be at least 96`。原因是TIME08滚动协议包含短于模型最小输入patch的历史窗口。
- 修复：`src/adapters/foundation/timerxl.py`对短于模型`input_token_len`的历史用最早观测值执行左侧edge padding；不跳过窗口，不变更测试目标。`scripts/exp/v1-rage-ads/train_eval.py`对已有未完成评估目录启用`--resume`。
- 恢复：先独立完成TIME08/fullRand；随后以`run_parallel_train_eval.py --workers 12`恢复rage、ancUni、Traj。四个TIME08训练和滚动评估单元全部完成，训练与评估各32/32完成标记齐全。
- TIME08 MASE：fullRand 1.1410，rage 1.1329，ancUni 1.1432，Traj 1.1799。四个评估均含32条序列、5个rolling窗口。
- 汇总：`queue_rolling.sh`检测选择、审计、训练、评估根完成标记后跳过已完成阶段，生成32行`downstream-metrics.jsonl`、18项描述性关联及`.summary-complete`。八任务算术平均MASE为fullRand 1.4788、rage 1.4637、ancUni 1.4744、Traj 1.4905。
- 计时：初次训练评估manager 27.7分钟后完成28单元；四单元恢复运行合计约18秒。13:07:17至13:21:56为故障定位/修复等待，不记作计算耗时；历史6.4小时仅是旧八任务串行累计值。
- 资源记录限制：首轮runner因失败退出而没有写出内存中的部分采样。现存`parallel-run-manifest.json`只覆盖三单元恢复，10个采样点的GPU利用率均值9.6%、显存峰值1561 MiB、主机内存均值/峰值167866/170333 MiB，不代表完整训练批次。完整正式训练阶段资源均值未留存；正式并发选择参考128-step实测profile，12 workers。
- 最终状态：2026-11-14 13:22:44 CST全部阶段汇总成功；`logs/v1-rage-ads/queue.status`置为status=0。GPU当前空闲；未关机。
- 关键输出：`outputs/selection/v1-rage-ads/rolling/audit/downstream-metrics.jsonl`、`outputs/selection/v1-rage-ads/rolling/audit/alignment-outcome-correlations.json`、`outputs/train/v1-rage-ads/rolling/parallel-run-manifest.json`、`outputs/eval/v1-rage-ads/rolling/`。
