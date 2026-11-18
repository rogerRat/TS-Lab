# 2026-11-17 main-crb-supp-chronos2-small 开发与数据预检

- 时间：2026-11-17 01:15（Asia/Shanghai）；服务器 tsl，项目 /root/TS-Lab。
- 用户决策：Chronos-2 small（28M）覆盖全部 24 个任务；当前 main-crb-supp 的 12 任务先完成，新增 12 个目标任务后，Chronos-2-small 执行完整 24 任务。
- 模型现状：仓库已有 Chronos-2-small 训练 adapter、LoRA 配置、固定 revision、专用 rage-chronos2 环境及本地权重。本次补齐本地 checkpoint 的通用 TIME/FEV 评估加载和原生分位数预测。CPU 本地权重 smoke inference 返回有限值，输出形状为 2×3×96。
- 新增评估开发：Foundation TIME/FEV evaluator 接受 chronos-2-small；评估从已训练 checkpoint 本地加载模型；TIME/FEV 使用 Chronos-2 原生 quantile_preds，并按目标 quantile levels 插值；point MSE 使用 0.5 quantile。
- 代码静态检查：修改文件 py_compile 通过；git diff --check 通过。未启动 GPU 训练。
- 任务预检：11 个新增 TIME 候选和 FEV hospital_admissions_1D 通过安全历史检查。其合法窗口数见 docs/exp/main-crb-supp-chronos2-small/exp.md。新增任务 reference 输入已写入 data/ref/shard_dataset/main-crb-supp-chronos2-small/run/。
- 候选筛除：缺少至少 576 个有限安全历史点的候选未纳入；完整列表见实验 exp.md。包含 SG_Carpark、若干短历史 TIME 项和 FEV M_DENSE/us_consumption/world_co2。
- 正式队列尚未启动；计划在 main-crb-supp 的 outputs/eval/main-crb-supp/run/.complete 出现后，以 tmux 串行执行 prepare、validate、run。前置会话未成功完成时停止后续队列。
- 待决策：CRB 选择 probe 是否继续使用 TimerXL q16/q8 梯度，或重算 Chronos-2-small q16/q8 梯度。该选择改变实验含义与运行成本，已向用户提出二选一；确认后更新 exp.md 和队列脚本。
- 未解决事项：等待 probe 模型决定；随后更新新任务表、准备全局 EXP/RESULTS 的 384 行批次索引，并启动 tmux 依赖队列。
