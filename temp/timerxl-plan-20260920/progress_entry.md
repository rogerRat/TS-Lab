
## 2026-09-21 — v5-timerxl 正式 run 完成 + reference-census 队列启动

- v5-timerxl（main-rank-volume-timerxl-20260920）387 unique cells（9 任务 × (14 臂×3 seeds + orig)）全部完成，state.json status=complete。收尾异常：seed2 worker 因宿主中断在 time_eweld_15t/rage-block-linearSum 报缺 epoch-0.500.pt checkpoint，supervisor 幂等重启后补齐，无重复计算。
- 冻结配方 lr 1e-4/epochs 4/schedule_epochs 6/batch 32/wd 0.01/cosine（两阶段 safe-history 标定 + 预注册 ext6/ext8 扩展确认）。
- 结果整理于 docs/TIMERXL.md（15 臂 × 9 任务 × 3 seeds 全量 + 跨模型同臂对比）。全局形态与 chronos/timesfm 反转：orig 最优（1.0802），所有选择臂差于 orig；block-vs-uni 混合（expSum uni −0.0039 / linearSum uni +0.0117）。
- 正式交接 reference-census-20260921 机制实验：census_queue.sh 部署至 /root/TS3/temp/reference-census-20260921/，tmux 会话 census-queue 自动按 阶段A（timerxl→chronos+timesfm 并行参考梯度）→B（对齐）→C（关联统计） 执行；启动门控：GPU 全空 + 磁盘 ≥130GiB；probe_validation 失败即停不绕过。输出 /root/autodl-tmp/TS3/outputs/reference-census-20260921/，queue.log 同目录。
