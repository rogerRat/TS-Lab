# main-crb-v0 全池重采样批次启动记录

- 检查时间：2026-11-16 19:50 CST
- 实验：`main-crb-v0`
- 批次：`q16-src256-eight-task-resample`
- 用户确认范围：八任务；方法组为 `ori`、`crb`、`uni`、`reft`、小写 `rand`、`TSDS`、`LP-Mirror`、`GraNd-static`、`CRB-traj`、`UNI-traj`。大写 `Rand` 固定随机 10% 子集不纳入；全部筛选方法使用完整池的概率采样。ref cosine 聚合默认改为 `linearSum`。

## 启动与产物

- 19:26:05 CST 启动 bootstrap 阶段。该首段通过 SSH 命令启动，未在 tmux 内；它很快完成，后续阶段均进入 tmux 队列。
- `ori` 八任务评估已完成；`reft` 八任务训练与评估已完成。任务配置复用 `configs/experiments/main-crb-v0/eight-task/<TASK>/`，结果写入新的 `outputs/{train,eval}/main-crb-v0/q16-src256-eight-task-resample/`，未覆盖旧批次。
- 19:37:10 CST，tmux `main_crb_resample_shared` 启动 `rand` 与 `GraNd-static` 两个共享训练。两次训练和八任务评估均于 19:44:05 CST 正常完成；日志为 `logs/main-crb-v0/q16-src256-eight-task-resample_20261116193710.log`，单方法训练日志在 `logs/tools/train/`，评估日志在 `logs/tools/run/`。
- 重用 source q16 与八任务 ref q8 资产；没有重新计算投影梯度。src 池共享索引和 `rand`/`GraNd-static` 概率载荷已生成于 `outputs/selection/main-crb-v0/q16-src256-eight-task-resample/`。全池为 6,382,020 个合法窗口。
- `scripts/tools/train.py` 新增 `data.resampling_payload` 与共享池索引加载分支；原有数组训练分支保留。共享训练以概率池对象交由 `fine_tune`，每 step 使用 seed=0 的局部 RNG。

## 修复记录

- GraNd 首次准备时将 `fragment_counts` 误当成锚点数；实际字段记录每条轨迹的合法窗口数。已按 q16 注册规则重新推导逐轨锚点数，并校验与锚点总数相等。
- GraNd 范数索引使用 source 梯度行号，应读取已有 `q16-src256-time1-q8/components/magnitudes_fp32.npy`；此前误指向只含新 ref 行的八任务缓存，已改正。
- CRB/UNI/轨迹概率首次准备生成 FEV09 的 CRB/UNI 载荷后，在 `CRB-traj` source 锚点路径处失败。原因是轨迹方法名应映射到既有 `crb/` 或 `uni/` 资源目录；代码已修正并从全批次入口重新运行，重复生成的 FEV09 概率会原位覆盖同一批次文件。
- 首次 CRB/UNI 选择计算落在 CPU，观测到 GPU 空闲。该次在完成少量准备后中断；现已改为 CUDA 分块插值与向量评分。GPU 版本通过 tmux `main_crb_resample_methods_run` 启动，日志追加在 `logs/main-crb-v0/resample-method-prep.log`。

## 当前队列

- `main_crb_resample_methods_run`：生成八任务 CRB/UNI 全窗口概率及 CRB/UNI 轨迹概率；随后自动运行 `run_task_resample.py --parallelism 8`，每任务顺序训练和评估四个方法。
- `main_crb_resample_native`：等待 `outputs/eval/main-crb-v0/q16-src256-eight-task-resample/task-conditioned.complete`，随后复用已接入 TSDS、LP-Mirror 链路生成原生概率，并运行八任务训练评估。
- 19:44:41 CST 的资源快照为 RTX 4080 32 GiB；shared 训练评估后 GPU 空闲。CUDA 评分队列已启动，后续任务在 tmux 自动接续；未关机。
- 计划总量 58 次预测模型训练、80 个方法-任务评估；当前已完成 10 次训练与 32 项评估，尚无新批次的 MASE/MSE 汇总表。数值汇总以已生成的任务 metadata 为准，不预填结果。
