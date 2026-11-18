# 2026-11-16 main-crb-supp 启动期显卡观察与并行调整

- 观察时间：2026-11-16 23:46–23:47（Asia/Shanghai）。
- 原正式队列：tmux main-crb-supp，首任务 Crypto/D 的 E1 已训练至约 1000/4000 step；尚无正式评估完成。
- 显卡观察：连续 12 次约 1 秒间隔采样的 SM 利用率为 11%–13%，显存约 3.5 GiB/32 GiB。单个 TimerXL LoRA 训练臂明显不能充分使用 GPU。
- 操作：向 tmux 发送 Ctrl-C，确认队列与训练子进程均退出。保留未完成 E1 的 step-1000 模型快照，原训练目录移动至 `outputs/train/main-crb-supp/interrupted/2026-11-16-first-run-time-crypto-d-e1-uniform-mean/`；未删除该现场或日志。首任务安全 reference 和 block 选择资产可复用。
- 修改：同一任务内先准备四个 block 选择臂，再并行运行 E1–E5 五个独立训练进程；五臂都结束后按 arm/step 顺序评估。跨任务仍严格串行。所有训练超参数、采样概率、随机种子、训练步数与三点评估节点不变。通用脚本 `scripts/tools/run.py` 仍用于每个训练和评估入口。
- 状态：并行队列代码已完成静态检查，正在独立 validation 目录复验五臂并发与三 checkpoint 评估；验证通过并清理后才重新启动正式队列。
- 待办：确认峰值显存与利用率，记录结果，继续 12 任务顺序队列。
