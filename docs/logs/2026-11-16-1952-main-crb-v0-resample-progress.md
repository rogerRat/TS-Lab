# main-crb-v0 resample 批次进度

- 检查时间：2026-11-16 19:52:33 CST
- 批次：`q16-src256-eight-task-resample`
- 当前任务：tmux `main_crb_resample_methods_run` 正在按任务准备 CRB/UNI 全池窗口概率及 CRB-traj/UNI-traj 轨迹概率。CUDA 分块评分已成功处理 FEV09 与 TIME1；TIME1 的 CRB、UNI、CRB-traj 概率载荷已写入新批次目录，后续任务继续由同一脚本完成。
- 当前队列：`main_crb_resample_native` 等待四方法 task-conditioned 队列完成标记，随后自动执行 TSDS/LP-Mirror 概率和训练评估。
- 资源：上一快照 GPU 0%、550 MiB 显存；评分本身以短 CUDA 批次执行，轨迹表示聚合在 CPU 分块完成。尚无新增训练或评估指标；不轮询后续状态。
- 下一步：等待 CRB/UNI/轨迹八任务概率全部生成，`run_task_resample.py` 随后启动八路任务训练评估；native 队列自动接续。
