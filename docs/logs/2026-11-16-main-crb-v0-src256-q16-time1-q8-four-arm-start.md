# main-crb-v0 src-256 q16 / TIME1 q8 四臂启动记录

- 检查与操作时间：2026-11-16 13:02 CST（tmux 会话创建时间 13:02:44 CST）。
- 实验批次：`q16-src256-time1-q8-four-arm`。
- 任务状态：远程 AutoDL 项目 `/root/TS-Lab` 当前 checkout 为 `rage`（未发现 `main-crb` 分支；因工作树含用户改动未切分支）；tmux 会话 `main_crb_q16_time1` 已创建并启动入口 `scripts/exp/main-crb-v0/run_directional_time1.py --stage all`；入口先准备梯度与选择资源，再顺序执行 CRB、REFT、ORI、UNI 的 TIME1 评估。启动后按规则不轮询进度。
- 资源检查：启动前 RTX 4080 显存占用 0 MiB；`/root/autodl-tmp` 可用约 435 GiB。目标输出目录此前不存在。
- 数据与预算：src-256 全池 6,382,020 个合法窗口，q=16 的 CRB/UNI 每种 399,044 锚点；CRB cover/error/split 为 199,846/0/199,198。TIME1 安全 ref 为 10 段、3,250 窗口，q=8 共 410 锚点。
- 方法约束：所有稀疏投影梯度拆分为 L2 幅值和单位方向；仅方向参与 CRB kappa、块方向均值、线性插值与后续余弦筛选。幅值只作审计，不进入筛选。CRB/UNI 各选 src-256 全池 top 10%（638,202 窗口）训练；REFT 使用全部安全 ref 窗口训练；ORI 不训练。
- 执行配置：Conda `rage-timerxl`；TimerXL 84M LoRA，4,000 步、effective batch 32、micro-batch 8、lr 1e-4、cosine、warmup 3%、seed 0；TIME1 rolling evaluation。训练 recipe 沿用 v1-rage-ads-v2，未由用户单独确认。
- 修改内容：新增 `scripts/exp/main-crb-v0/run_directional_time1.py`；`scripts/tools/train.py` 增加可选 `TrainingRecipe` 配置以复用项目已有训练 recipe；更正 `docs/exp/main-crb-v0/` 与全局实验/结果索引中的 source 范围和预算。
- 启动命令：`tmux new-session -d -s main_crb_q16_time1 "bash -lc 'cd /root/TS-Lab && source /root/miniconda3/etc/profile.d/conda.sh && conda activate rage-timerxl && python -u scripts/exp/main-crb-v0/run_directional_time1.py --stage all'"`。
- 启动前语法编译、CLI 帮助、输入路径和缓存目标目录检查通过；未运行测试。启动时未观察到报错；运行进度与结果待任务自身日志和完成文件记录。
- 输出路径：`outputs/grad_cache/main-crb-v0/q16-src256-time1-q8/`、`outputs/selection/main-crb-v0/q16-src256-time1-q8/`、`outputs/train/main-crb-v0/q16-src256-time1-q8/`、`outputs/eval/main-crb-v0/q16-src256-time1-q8/`。
