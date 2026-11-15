# main-ads-v0 实验建立与旧梯度资源清理

- 检查时间：2026-11-15 CST。
- 实验代号：`main-ads-v0`。
- 批次：`traj-replay`。
- 状态：文档和运行入口已建立；训练及新梯度计算尚未启动。
- 继承设置：参考 `v1-rage-ads-v2` 的 TimerXL 训练设置，使用 src-256、LoRA q/v、r=8、alpha=16、seed 0；训练步数设为 2,000。
- 清理检查：`/root/autodl-tmp/rage/outputs/` 当前为空，原有模型检查点和梯度缓存均不存在；没有遗留梯度文件需要删除。
- 新运行入口：`scripts/exp/main-ads-v0/run_main_ads_v0.py`，包括训练、每 10 步 loss 记录、loss 图和两条轨迹全部合法窗口的 4096 维 seed 0 梯度缓存。
- 轨迹：ordinal 0 `monash_australian_electricity/T000004`；ordinal 1 `monash_fred_md/T000066`。
- 文档：新增 `docs/exp/main-ads-v0/exp.md` 和 `results.md`，并更新全局 EXP、RESULTS 索引。
- Replay 内容：待定。
