# v1-rage-ads-v2 ads-rev 启动与稳定性检查

- 检查时间：2026-11-15 03:51 CST
- 实验代号：v1-rage-ads-v2
- 新批次：ads-rev-reverse-complement
- 算法：从 q0=32 粗锚点开始；对参与 `cos>=0.1` 配对的 source/reference 锚点做排除；未命中粗锚点连续段按 q1=16 均布一轮；最终按现有 source 最大分、负分截零、h=4 邻域概率规则处理。未执行标准 ADS-VIC q1=8；未运行 uni。
- 启动命令：`tmux new-session -d -s v1_rage_ads_v2_ads_rev "bash scripts/exp/v1-rage-ads-v2/run_ads_rev_pipeline.sh"`
- 进程：tmux 会话 `v1_rage_ads_v2_ads_rev`；选择 PID 28340 持续运行；训练评估将在选择完成后由同一 pipeline 自动启动，runner 配置 12 workers。
- 稳定性观察：启动后约 80 秒，FEV09 q0 source 粗梯度推进到 fragment 51/256；Python 进程约 100% 单核；GPU 利用率 53%、显存 980 MiB；容器内存约 11.1/66.6 GB；无异常退出或 OOM。
- 代码入口：`scripts/exp/v1-rage-ads-v2/run_ads_rev_selection.py`；流水线：`scripts/exp/v1-rage-ads-v2/run_ads_rev_pipeline.sh`；训练臂限定为 ads-rev，不改动原五臂输出。
- 日志：`logs/v1-rage-ads-v2/ads-rev-pipeline.log` 与 `logs/v1-rage-ads-v2/ads-rev-selection_20261115034952.log`。
- 状态：选择运行中；尚无选择、训练或评估完成标记，尚无 MASE 结果。
