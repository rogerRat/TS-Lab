# main-crb-v0 八任务队列路径修复与恢复

- 检查时间：2026-11-16 16:55:57 CST。
- 实验批次：main-crb-v0 / q16-src256-eight-task。
- 故障：16:23:45，prepare_eight_task.py 在 write_configs 中对 outputs/train 真实数据盘路径调用 relative_to(ROOT)，因 outputs/ 是数据盘软链接而抛出 ValueError。队列在 reference 资源注册结束后退出，GPU 空闲。
- 故障影响：36,331 条 ref q8 梯度已计算；方向 float16 和幅值 float32 sidecar、八项 ref block directions、REFT 全窗口资源与 resource_manifest.json 已生成。资源根 .complete 未写入；source 窗口评分、训练、评估均未开始。
- 修复：configs 中 outputs 路径改写为绝对路径；pipeline 中位于项目 configs/ 下的配置引用仍使用相对路径。修改 scripts/exp/main-crb-v0/prepare_eight_task.py；两个入口脚本源码编译通过。
- 恢复：从 --stage all 重启。GradientCache 读取已有 shards，跳过已完成的梯度请求，再重写资源侧文件并生成配置；不删除或覆盖历史批次。
- 运行环境：rage-timerxl；tmux main_crb_q16_eight_tasks；并行度 8；日志 logs/main-crb-v0/eight-task-20261116.log。
- 重启时间：2026-11-16 16:55:57 CST。训练评估仍未开始，尚无指标；后续由队列继续执行。
