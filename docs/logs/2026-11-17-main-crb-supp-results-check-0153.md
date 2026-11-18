# 2026-11-17 main-crb-supp 结果检查

- 检查时间：2026-11-17 01:53（Asia/Shanghai）。
- 实验：main-crb-supp，TimerXL 84M，tmux 队列仍运行；总完成 10/12 个任务、160/192 个正式评估点。
- 当前批次：第 11 个任务 current_velocity/5T；E1–E5 并行训练最新约 1700–1800/4000 step；step-2000/4000 及正式评估未完成。第 12 个任务 azure2019_U/5T 尚未开始。
- 资源：RTX 4080 利用率 99%，显存 7712/32760 MiB。
- 已完成任务的 4000-step 最佳方法与 MSE：Crypto/D E2=13,447,696；Supply_Chain_Customer/D E5=39,559,942；Supply_Chain_Location/D E5=21,804,137；NE_China_Wind/H E3=0.11635569；Coastal_T_S/20T E5=0.027590468；Australia_Solar/H E5=2.5126767e10；Finland_Traffic/15T E5=795.31765；FEV jena_weather_1H E5=1088.8574；FEV ETT_1H E4=14.351423；MetroPT-3/5T E5=3.8407086。每任务所有方法与 step 的原始记录见 docs/exp/main-crb-supp/results.md。
- Chronos-2-small：无正式训练/评估结果。其 tmux 队列仍等待 main-crb-supp/run/.complete；前置标记尚不存在。
- 无错误或修复操作；未设自动巡检或循环查询。
