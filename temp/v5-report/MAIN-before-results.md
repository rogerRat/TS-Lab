# 当前实验：V5

唯一配置：`configs/rags/main.json`。继承停止前 V4 的两模型、九共享任务、13 方法、seeds 0/1/2 和训练配方；V5 改动限定为 [冻结方法](method.md)。没有选定唯一 RAGS 主方法；Block/Traj 四种聚合均保留。三个 seed 按整轮完成屏障串行，轮内两模型并行。

V4 已按用户 2026-09-18 授权停止，停止记录位于既有结果目录 `STOPPED_FOR_DEVELOPMENT.json`；结果、检查点、原始梯度缓存保留。旧选集、CRB 重建资产、训练 cells 不能作为 V5 结果续跑。逐窗口梯度只有在模型、投影、精度、窗口及数据字段一致时才允许复用；新 CRB 和新概率必须重新构造。

## 唯一入口

在 `/root/TS3`、`rags` 分支执行：

```bash
python -m rags.experiments.formal_main_entry --config configs/rags/main.json --run /root/autodl-tmp/TS3/outputs/main-rank-volume --dry-run
python -m rags.experiments.prepare_suite --config configs/rags/main.json --run /root/autodl-tmp/TS3/outputs/main-rank-volume
python -m rags.experiments.full_pool_assets --run /root/autodl-tmp/TS3/outputs/main-rank-volume --student chronos-base
python -m rags.experiments.suite_report --run /root/autodl-tmp/TS3/outputs/main-rank-volume
```

准备入口默认不启用正式选择、训练、评测。用户实际决定启动正式实验后，在 run 的 `execution_control.json` 显式启用三项控制，再运行 `formal_main_entry`（不带 `--dry-run`）。2026-09-18 用户已明确授权启动，正式 V5 现运行于 `/root/autodl-tmp/TS3/outputs/main-rank-volume-20260918`，tmux 会话 `main-rank-volume`。科学代码版本 `6d3ead6`；执行优化参数独立记录在 run 的 `execution.json`，GPU 连续采样见 `gpu.csv`/`telemetry.json`。

各模型 worker 使用配置内的独立 Python 环境。输出默认在数据盘；源码临时备份和测试日志在仓库 `temp/v5-final/`。字段比对隔离缓存，不计算数据/模型/资产 hash。写缓存先同目录临时文件，再原子替换，单写者锁。选集 NPZ 保存原始分数、最终对象概率、窗口所有权和完整冻结协议，不保存全体参考×源配对矩阵。

## 验证

验证覆盖排序数值间隔去除、对象数量无关区间质量、精确并列、源窗口量先验、完整窗口所有权、冻结配置拒绝变更、无 P10/调度 RNG、精确预算、准备后零额外 BP、保存/抽样一致、旧资产拒绝、worker 训练与恢复、seed 轮次屏障及真实模型小规模训练/评测。实际通过数和运行产物见 `Progress.md`；开发样本结果不能说明正式效果。

旧 V1–V4 入口和版本后缀配置已清理，完整清理前源码备份是 `temp/v5-final/pre-cleanup-code.tar.gz`，不纳入 Git。历史结果和旧方法说明位于 `docs/archive/`。
