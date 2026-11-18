# main-crb-supp 主指标切换为 MASE

- 检查时间：2026-11-17 02:00 CST。
- 实验：`main-crb-supp` TimerXL 队列及其依赖队列 `main-crb-supp-chronos2-small`。
- 队列状态：TimerXL 12 个任务中 11 个完成，176/192 个评估点已完成；第 12 个任务 `time-azure2019-u-5t` 已进入模型训练，尚无全队列 `.complete`。GPU 训练仍在运行。
- 主指标：按用户要求统一为 MASE，越低越好。结果文件使用原始 TIME `metrics.npz` 中的 `MASE` 和 FEV `summary.csv` 中 `MASE` 字段；各任务独立报告，不跨任务求宏平均。
- 已有结果：从已完成 11 个任务的 step-4000 原始指标重新读取 MASE，逐任务最优方法为：Crypto/D e2=7.1397204；Supply_Chain_Customer/D e5=0.69473948；Supply_Chain_Location/D e5=0.71726027；NE_China_Wind/H e3=0.56122152；Coastal_T_S/20T e5=0.85960794；Australia_Solar/H e5=0.59419195；Finland_Traffic/15T e5=0.46593138；Jena weather e4=0.51416732；ETT e4=1.1593857；MetroPT-3/5T e1=0.69469144；current_velocity/5T e0=0.59170645。具体原始文件路径见 `docs/exp/main-crb-supp/results.md`。
- 处理：当前 TimerXL 进程在本次切换前已载入旧 MSE 结果写入逻辑，保持其运行不改动；源码后续采用 MASE。为避免与运行队列并发写文档，新增转换脚本 `scripts/exp/main-crb-supp/apply_mase_results.py`，由 Chronos-2 依赖启动脚本在 TimerXL `.complete` 后调用，从每个原始结果文件重建 MASE 表格及全局索引。Chronos-2-small 的 24 任务队列结果头已设为 MASE。
- 下步：TimerXL 正常完成后自动转换已有结果并启动 Chronos-2-small 的 prepare、validate、run。
