# main-crb-v0 data-select-005-linear 入队记录

- 记录时间：2026-11-16 CST
- 实验代号与批次：`main-crb-v0` / `data-select-005-linear`
- 前置批次：`q16-src256-eight-task-resample`，八任务 TSDS/LP-Mirror native 训练评估仍在运行；GPU 快照 99%。
- 操作：新增固定子集训练视图 `FixedSubsetWindows`，使训练按需从共享 src-256 pool 读取被选窗口，不复制整个数据池；新增批次队列入口 `scripts/exp/main-crb-v0/run_data_select_005_linear.py`。队列待前一批训练与评估标记齐全后生成选择配置并启动。
- 批次定义：top 5% 精确为 319,101/6,382,020 个合法窗口。窗口级使用其概率；轨迹/区间级将单位概率除以该单位合法窗口数以得到每窗口边际概率；概率并列时按全局索引升序。rand 与 ref-only REFT 固定使用 PCG64 seed=0 无放回选择。CRB/UNI 轨迹与窗口匹配统一采用 ref `linearSum`。
- 方法与规模：ORI 底座评估；rand、GraNd-static 各共享训练一次；REFT、CRB、UNI、CRB-traj、UNI-traj、TSDS、LP-Mirror 八任务训练；合计计划 58 次预测模型训练、80 次评估。
- 输出：`outputs/{selection,train,eval}/main-crb-v0/data-select-005-linear/`；本批尚未启动，无实际指标。实验依据已补入 `docs/exp/main-crb-v0/exp.md`，索引已补入 `docs/EXP.md`、`docs/RESULTS.md`。
- 队列会话：`main_crb_data_select_005_linear`。


## 自动接续启动

- 前置批次完成并满足启动门槛：2026-11-16 21:30:44 CST。队列核验到 TSDS native 完成标记、80 个评估完成标记和 58 个训练完成标记。
- 固定子集生成：八任务每个概率条件方法均精确 319,101 个窗口；并列规则和各方法阈值写入选择 metadata。
- 实际启动：2026-11-16 21:30:46 CST，rand 与 GraNd-static 两个共享 TimerXL 预测模型并行开始训练；之后先评估 ORI/rand/GraNd-static，再由八任务并发队列运行 REFT、CRB、UNI、CRB-traj、UNI-traj、TSDS、LP-Mirror。
- 启动状态快照：rand 与 GraNd-static 均记录 `step=1/4000`；GPU 利用率 93%，显存 1,942 MiB / 32,760 MiB。tmux 会话 `main_crb_data_select_005_linear` 持续运行。
- 状态：批次运行中；目前没有本批评估指标。选择目录为 `outputs/selection/main-crb-v0/data-select-005-linear/`，训练和评估目录为同批次名下 `outputs/train/`、`outputs/eval/`。
