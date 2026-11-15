# v1-rage-ads 并行优化与正式队列启动

- 检查时间：2026-11-14 12:36 CST
- 实验代号：v1-rage-ads
- 批次：rolling-parallelism-profile；rolling-eight-task
- 状态：性能试跑已完成；正式八任务审计已启动，训练/评估等待审计完成
- 启动会话：tmux v1-rage-ads-rolling-optimized
- 项目：/root/TS-Lab

## 既有选择计时口径

旧选择队列是任务级串行。每任务四臂合计 44.67–50.52 分钟，均值 47.75 分钟；八任务串行累计约 6.4 小时。6.4 小时只作为旧串行队列历史值，不作为单任务实际耗时或本次并行队列耗时。旧选择执行无显存/主机内存周期采样，GPU 0%、922 MiB 只是单点记录，未补造平均值。

## 性能试跑

- Warmed gradient：每 worker 1,024 窗口；全部梯度由 step-1000 warmed TimerXL checkpoint 计算。batch32/VJP32 单 worker 608.1 windows/s；1/2/4/8 workers 总吞吐 608.1/931.8/1,081.2/1,112.8 windows/s。8 相对 4 只提高 2.9%，显存峰值从 3,912 MiB 增至 7,820 MiB。
- Audit：TIME01、每臂正式抽样 4,000、梯度 batch32/VJP32、2 CPU threads/worker。1/2/4/8 workers 吞吐为 0.0131/0.0227/0.0349/0.0497 task/s；8-worker GPU 均值 77.1%，显存均值/峰值 5,036/16,524 MiB，worker RSS 均值/峰值 31,637/43,091 MiB，主机 CPU 利用率均值约 16.0%。
- Train：实际模型、sampler、LoRA/AdamW recipe；32-step扫描8/12/16/24 workers为0.5964/0.7123/0.6343/0.5997 cells/s；128-step确认8/12/16 workers为0.3135/0.4036/0.3395 cells/s。12 workers最快。128-step的12-worker GPU均值66.4%，显存均值/峰值8,577/11,635 MiB，RSS峰值24,587 MiB，主机内存峰值131,503 MiB。
- Eval：TIME01完整rolling配置、预训练 TimerXL 权重，仅作容量试跑。1/2/4/8/16/24/32 workers吞吐为0.1834/0.3662/0.6027/1.0312/1.3571/1.1780/1.0748 jobs/s；16 workers最快。正式runner用12 workers以适配训练瓶颈。
- Sampler：紧凑 SourceProbabilitySampler 为994,688.9 draws/s，dense rng.choice(p=...)为1,328.9 draws/s；仅为抽样调用微基准。
- Audit复用完整q32 source unit-vector cache，83,330×4,096 float32，约1.36 GiB；该缓存不减少source锚点总体或对齐计算。

所有 profile JSON、逐worker日志及资源采样输出保留在项目 outputs/selection、outputs/train、outputs/eval/v1-rage-ads/parallelism/ 和 /root/autodl-tmp/rage/logs/v1-rage-ads/。

## 改动与正式执行

- run_selection.py 默认窗口/VJP批次更新到32/32，使用 warmed checkpoint；已完成的选择产物未重算。
- 审计新增完整源单位向量缓存、GPU批量对齐与进程安全JSONL并行写入。
- 并行runner增加 worker 进程树 CPU/RSS 采样与独立进程组中断回收，修复训练/评估调度器采样状态初始化。
- queue_rolling.sh 改为正式审计8 workers、梯度 batch32/VJP32，训练/评估12 workers。
- 正式启动前检查：selection根标记存在；audit、train、eval根标记尚不存在；source unit cache完整；RTX 4080 32 GiB空闲。
- 执行命令：bash scripts/exp/v1-rage-ads/start_queue.sh（tmux v1-rage-ads-rolling-optimized）。
- 12:36:52 起审计任务日志显示正式 production 任务运行；审计完成后自动进入训练/评估。
- 正式审计输出：outputs/selection/v1-rage-ads/rolling/audit/。正式训练输出：outputs/train/v1-rage-ads/rolling/。滚动评测输出：outputs/eval/v1-rage-ads/rolling/。

## 未解决与下一步

正式审计、训练与 rolling 评估仍在队列中；当前没有完整审计结果或模型效果指标。以每个阶段 .complete、manifest 和模型产物更新 docs/exp/v1-rage-ads/results.md、docs/EXP.md 与 docs/RESULTS.md；不得从性能 profile 推导模型效果。
