# ADS-VIC 直方图重算改为双 worker

- 修改时间：2026-11-15 14:01 CST。
- 用户要求继续并行以缩短绘图时间。
- 首次顺序重算在 FEV09 q0 梯度计算中途停止，尚未产生最终文件；GPU 已释放。
- 更新 `scripts/exp/v1-rage-ads-v2/plot_ads_vic_cosine_histograms.py`，使用两个独立 worker，每个 worker 初始化一次相同 gradient probe，并按固定任务组计算：worker 0 处理 FEV09、TIME01、TIME02、TIME03；worker 1 处理 TIME04、TIME06、TIME07、TIME08。
- 固定分组让 TIME06 与 TIME08 在同一 worker 中串行，避免两个最大任务同时争用显存；每个 worker 单独日志文件。GPU 启动前为 0% 使用、0 MiB 显存。
- 当前 tmux 会话：`ads_vic_hist`；命令 `python -u scripts/exp/v1-rage-ads-v2/plot_ads_vic_cosine_histograms.py --bins 100 --workers 2`。
