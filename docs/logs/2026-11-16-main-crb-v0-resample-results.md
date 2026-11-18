# main-crb-v0 resample 已完成方法结果整理

- 检查时间：2026-11-16 20:47:53 CST
- 批次：`q16-src256-eight-task-resample`
- 指标来源：FEV09 各 arm 的 `summary.csv`；TIME1/2/3/4/6/7/8 各 arm 对应 `metrics.npz` 的 MASE 数组。TIME MASE 按序列、窗口、变量轴算术平均，与既有结果文档口径一致。
- 已完成：ORI、CRB、UNI、REFT、rand、GraNd-static、CRB-traj、UNI-traj 共 64 个评估；训练 42 次。最低 MASE 逐任务为 FEV09=CRB 0.728157，TIME1=UNI 0.708068，TIME2=UNI-traj 1.272067，TIME3=REFT 1.198468，TIME4=REFT 2.588262，TIME6=REFT 1.011427，TIME7=UNI 1.527212，TIME8=ORI 1.107539。
- TSDS/LP-Mirror 尚无结果。首次 native 特征准备报错：把 `build_model` 返回的 adapter engine 传给了 `encoder_embeddings`，但该函数需要暴露 encoder 方法的 TimerXL gradient probe。已改为传入 `_probe`，于 20:47:26 CST 在 tmux `main_crb_resample_native_run` 重启 native 准备；20:47:53 检查时 Python 进程仍运行，GPU 利用率 24%、显存 940 MiB。
- 详细逐任务 MASE 表已写入 `docs/exp/main-crb-v0/results.md`；原始指标文件保留在 `outputs/eval/main-crb-v0/q16-src256-eight-task-resample/`。
