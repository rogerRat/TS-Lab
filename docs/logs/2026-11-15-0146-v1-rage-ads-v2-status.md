# v1-rage-ads-v2 进度、恢复与优化记录

- 检查时间：2026-11-15 01:46 CST
- 选择：8/8 完成。TIME07 于 01:10:20 完成，source/reference 锚点预算 663,852/222，耗时 35.1 分钟。
- 训练评估：v1_rage_ads_v2_train tmux runner 于 01:40:47 以 12 workers 恢复。13/40 单元已有训练和评估完成标记，当前 active=12、pending=15、failed=0。前次首批有 3 个失败单元，FEV orig 缺 FEV provenance manifest，reft 窗口索引未按 owner 选轨迹；两处已修复，前次 runner 也因失败提前停止，已改为继续调度其余单元。
- 训练资源：GPU 利用率约 100%，显存 11,635/32,760 MiB；cgroup 内存 20,750,819,328/66,571,993,088 bytes；memory.events 的 oom=0、oom_kill=0。
- 审计：audit_probability_support.py 已完成 40 行。每任务源池 6,382,020 个合法窗口。零概率比例：ADS-VIC 30.3042%，uni 19.7117%，traj 17.5960%，orig 0%；reft 不适用。结果在 outputs/selection/v1-rage-ads-v2/audit/probability-support/summary.json。
- 相似度优化：run_selection.py 增加 CUDA float32 分块 GEMM，保留 threshold/max score 计算、按显存预算选择 CUDA 或 CPU fallback。GPU parity 单测执行 1 项通过。该实验选择 8/8 已完成，本次未重跑已完成选择。
- 训练恢复修改：train_eval.py 的 reference owner 索引修复；FEV orig 创建仅含软链接和 manifest 的输出 checkpoint view，不修改预训练权重；run_parallel_train_eval.py 在单元失败后继续剩余队列。
- 提交状态：相似度优化及相关 v2 恢复/审计代码待提交。
- 待办：跟踪训练评估 40 单元、结果汇总；审计和测试代码均已启动/完成。
