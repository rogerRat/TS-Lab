# main-crb-supp TimerXL 队列完成结果

- 检查时间：2026-11-17 02:12 CST。
- 队列状态：完成标记存在；12/12 个任务、192/192 个方法与步数评估点均已完成。运行日志记录队列从 2026-11-16 23:52:59 到 2026-11-17 02:10:30，耗时约 2.3 小时。
- 主指标：MASE，越低越好。逐任务最优值按 4000 step 的六种方法比较，从 TIME 原始 `metrics.npz` 和 FEV `summary.csv` 重新读取。
- 逐任务最优：Crypto/D e2-uniform-shape 7.1397204；Supply_Chain_Customer/D e5-target-only 0.69473948；Supply_Chain_Location/D e5-target-only 0.71726027；NE_China_Wind/H e3-crb-mean 0.56122152；Coastal_T_S/20T e5-target-only 0.85960794；Australia_Solar/H e5-target-only 0.59419195；Finland_Traffic/15T e5-target-only 0.46593138；Jena weather e4-crb-shape 0.51416732；ETT e4-crb-shape 1.1593857；MetroPT-3/5T e1-uniform-mean 0.69469144；current_velocity/5T e0-rand 0.59170645；azure2019_U/5T e5-target-only 0.85138624。
- 汇总路径：`docs/exp/main-crb-supp/results.md`；全局索引：`docs/EXP.md` 与 `docs/RESULTS.md`；所有评估点有对应原始指标路径。
