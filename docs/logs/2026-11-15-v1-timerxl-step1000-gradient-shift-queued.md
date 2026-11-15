# TimerXL step-1000 梯度差异实验排队记录

- 时间：2026-11-15 14:26 CST。
- 实验代号：`v1-timerxl-step1000-gradient-shift`。
- 队列：tmux `timerxl_grad_shift_wait`；入口脚本 `scripts/temp/queue_timerxl_gradient_shift.sh`（临时脚本，不纳入 git）。
- 队列条件：等待 ads-v2 直方图进程退出，并要求 GPU utilization <= 10%、显存占用 <= 4 GiB；随后激活 `rage-timerxl` 并以无缓冲输出运行梯度比较入口。
- 当前状态：排队中；梯度计算尚未启动。队列日志：`logs/v1-timerxl-step1000-gradient-shift/queue.log`。
- 运行约束：不打断现有 GPU 进程；不关机；正式运行使用 tmux，训练与结果写入项目标准目录。
