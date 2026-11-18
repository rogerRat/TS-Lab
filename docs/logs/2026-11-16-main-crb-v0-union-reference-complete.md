# main-crb-v0 source+reference 两新增训练臂完成巡检

- 检查时间：2026-11-16 18:26 CST。
- tmux main_crb_union_train 已退出；队列日志 logs/main-crb-v0/q16-src256-eight-task_20261116180536.log 于 18:22:58 记录 stage=union-batch-complete。未见活动训练/评估进程，GPU 空闲。
- 八项任务的 crb_reft 和 uni_reft 均有 eval .complete 与指标文件；共 16 个新增评估。批次 train/eval 根 .complete 均存在；加上原臂，本批共 40 个训练臂和 48 次评估。
- 已从八任务的评估产物提取 MASE/FEV SQL 并逐任务标记最低值。完整六臂矩阵已写入 docs/exp/main-crb-v0/results.md；状态索引同步至 docs/EXP.md、docs/RESULTS.md。
- 输出：新增评估在 outputs/eval/main-crb-v0/q16-src256-eight-task/<TASK>/{crb_reft,uni_reft}/；训练集与其组成计数在 outputs/selection/main-crb-v0/q16-src256-eight-task/<TASK>/{crb_reft,uni_reft}/metadata.json。
