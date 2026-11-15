# v1-rage-ads rolling 队列启动记录

- 检查与启动时间：2026-11-14 04:20:47 CST。
- 实验代号 / 批次：`v1-rage-ads` / `rolling-eight-task`。
- 任务：FEV 9 与 TIME 1/2/3/4/6/7/8；四训练臂 `fullRand`、`rage`、`ancUni`、`Traj`。
- 启动命令：`tmux new-session -d -s v1-rage-ads-rolling "bash /root/TS-Lab/scripts/exp/v1-rage-ads/start_queue.sh"`。
- 队列步骤：滚动安全历史 reference 的 ADS 选择；每任务每臂按真实采样器有放回抽取 4,000 个窗口完成对齐与梯度审计；训练 32 个任务-臂单元；rolling 评估；汇总 MASE 与描述性对齐关联。
- 启动前检查：src-96 manifest、96 条轨迹和 2,666,312 个合法窗口与配置一致；step-1000 warmed source q32 梯度缓存字段身份通过；八项滚动 reference 均有 `.complete` 且明确不含答案值；八项评估配置均为 rolling。
- 修复：源池目录使用既有 manifest 与 packed-array 格式，没有 `.complete` 文件。选择与审计入口改为核对 manifest 字段、轨迹数、时间点数、偏移和合法窗口数，不为只读输入补造完成标记。审计单任务运行的 ETA 索引改为按所选任务重新编号。
- 启动状态：`logs/v1-rage-ads/queue.status` 记录 `status=started`；tmux 会话已创建。入口日志为 `logs/v1-rage-ads/rolling-eight-task_20261114042047.log`。
- 实际结果：启动时尚无本批次训练或评估指标；后续只记录由输出文件实际支持的结果。
- 下一步：队列按脚本顺序继续运行；无需轮询。失败或阶段完成后根据状态文件、阶段 `.complete`、日志与产物更新实验文档。
