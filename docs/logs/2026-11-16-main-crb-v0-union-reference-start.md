# main-crb-v0 新增 source+reference 训练臂启动

- 操作时间：2026-11-16 18:05:36 CST。
- 用户要求新增 CRB+REFT、UNI+REFT，并把每种选定 source 窗口与该任务 ref 窗口共同用作训练集合。
- 新臂名：crb_reft / uni_reft。数据按 selected source windows 后接该任务所有 ref training windows 拼接；不去重。每臂各含 638,202 个所选 source 窗口加 61–259,085 个 ref 窗口。16 个合并 NPZ、元数据和数据 .complete 均已生成。
- 新臂从 TimerXL 84M 底座独立训练，4,000 steps；LoRA q_proj/v_proj r=8 alpha=16；effective batch=32、micro-batch=8、lr=1e-4、cosine、warmup=3%、seed=0。评估复用对应任务的 CRB/UNI 评估配置。
- 实现：新增 scripts/exp/main-crb-v0/prepare_union_arms.py；更新 scripts/exp/main-crb-v0/run_eight_task_queue.py 增加仅运行并集臂的 --stage union，执行前清除旧批次的根/任务完成标记，避免新臂完成前误报整批完成。两脚本编译通过。
- 启动：tmux main_crb_union_train，命令 python -u scripts/exp/main-crb-v0/run_eight_task_queue.py --stage union --parallelism 8，环境 rage-timerxl。8 个 task worker 已创建；每个任务按 crb_reft、uni_reft 顺序训练和评估。原四臂保留且由完成标记跳过。
- 日志：logs/main-crb-v0/union-prepare-20261116.log；队列和各训练/评估阶段由共享工具记录在 logs/main-crb-v0/、logs/tools/。
- 当前新增指标尚未生成；等待队列完成后按实际文件更新结果表。
