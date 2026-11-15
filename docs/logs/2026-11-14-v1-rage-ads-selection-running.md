# v1-rage-ads 选择阶段运行记录

- 状态检查时间：2026-11-14 04:28:50 CST。
- 实验 / 批次：`v1-rage-ads` / `rolling-eight-task`。
- 进程证据：tmux `v1-rage-ads-rolling` 存在；`logs/v1-rage-ads/queue.status` 为 `status=started`。重启后的入口日志 `logs/v1-rage-ads/rolling-eight-task_20261114042750.log` 记录 FEV 9 的 R=1 ADS `stage=ads-start`。
- 已有阶段产物：FEV 9 的 step-1000 warmed q=16 reference 粗梯度 `outputs/train/v1-rage-ads/rolling/gradients/fev-09/reference_coarse.npz`，共 168 anchors；`fullRand/.complete` 已生成。尚无 rage 选择结果、训练结果或评估指标。
- 梯度缓存：启动前 source q=32 缓存通过逐字段身份校验；runner 以只读 cache lookup 复用，没有重算或复制 source 梯度缓存。新生成的仅为本实验滚动安全 reference 粗梯度文件。
- 资源快照：单次 `nvidia-smi` 观测为 GPU 0%、显存占用 922 MiB；采样时程序在 R=1 ADS 选择阶段运行，日志心跳显示进程存活，单次瞬时读数不足以判定持续闲置。
- 下一阶段：按既定队列完成八任务四臂概率抽样审计、训练、rolling 评估和结果汇总。当前不轮询。
