# v1-crb-redo 两基线批次并入实验文档日志

时间：2026-11-13 CST
性质：将独立结果文件 `v1-crb-redo-timerxl-baselines.md` 作为批次并入同实验的 `exp.md` / `results.md`，不新建实验。全部表格数值由脚本从两个源 Markdown 直接解析，无手工转抄。

## 1. 文件变更

- 删除 `docs/exp/v1-crb-redo/v1-crb-redo-timerxl-baselines.md`（`git rm`，历史可追溯）。
- `docs/exp/v1-crb-redo/results.md`：并入 `timerXL-baseline-full`（表中 baseline full）与 `timerXL-baseline-rand01`（表中 baseline rand01）两个批次的全部结果。
- `docs/exp/v1-crb-redo/exp.md`：两批次的设计小节本就存在，仅把 2 处结果文件引用改指 `results.md` 的 TimerXL 两基线小节。
- `docs/EXP.md`、`docs/RESULTS.md`：4 处批次表路径引用改指 `docs/exp/v1-crb-redo/results.md`；删除 RESULTS.md 报告索引中独立的 `v1-crb-redo-timerxl-baselines` 行。

## 2. results.md 结构调整

- 新增"批次与方法"小节：六个 arm 的关系、两基线训练协议、原始指标文件、训练清单、随机索引与数据修订。
- MASE 主指标汇总表扩为六列（baseline full、baseline rand01 加原四个 RR arm），共 12 个任务行：
  - 八个可行任务六值齐全；四个 RR 放弃任务（FEV M_DENSE_1D、FEV us_consumption_1Y、FEV world_co2_emissions、TIME ECDC_COVID/W）只有两基线值，RR 单元格以 — 标记并在行名注明放弃原因。
  - 平均行三行：RR 八任务宏平均（仅 RR 四列）、基线 FEV 四任务平均、基线 TIME 八任务平均（后两者仅基线两列），口径不同不混合。
- 新增"两基线 MASE 差值（rand01 - full）"12 行表，完整保留原文件差值列的全部数值，该表不加粗。
- 全量指标：FEV 扩为四个任务小节（hospital 六方法行，三个放弃任务各两方法行）；TIME 大表扩为八个任务块（可行任务六行，ECDC_COVID/W 两行）。
- RR 选择耗时表原样保留；新增说明：两基线不执行 RR 选样，无可比选择耗时。
- 原文件的完成情况、数据来源、10/12 项对比结论、单 seed 显著性限制、manifest 路径等事实全部以正文保留。

## 3. 加粗规则（重新计算）

全部指标为误差/损失类，同条件比较取最小值加粗；放弃任务只在两基线之间比较；平均行在该行实际有值的列内比较；并列最小同时加粗；差值表与非比较性文字不加粗。

## 4. 校验

- 脚本：`scripts/temp/merge_baselines.py`（生成）、`scripts/temp/validate_merge.py`（校验），均不纳入 git。
- 数值多重集：两个源文件全部表格数值在新文件中按多重集完全守恒，无丢失、无新增；仅 4 个基线均值（6.447586、6.703599、1.710087、1.736942）从原正文提升为表格单元格。
- 加粗正确性：8 张表逐比较组独立复算最小值，加粗位置 0 处不符。
- 结构：MASE 表 15 行、FEV 4 小节（6/2/2/2 行）、TIME 8 块（6 或 2 行）、— 仅出现在放弃任务与平均行；无数字任务代号残留；RR 耗时表逐字符串保持原样。
- 引用：除 `docs/logs/` 冻结历史日志外，全仓库无对已删除文件的残留引用；本次改动文件涉及的 98 个 Markdown 链接中，v1-crb-redo 相关链接全部有效。

## 5. 环境观察（非本次操作）

校验时发现另一并行操作已把 35 个实验文件夹移入未跟踪的 `docs/exp/archive/`，导致 `docs/EXP.md`、`docs/RESULTS.md` 中指向这些实验的既有链接悬空。该归档与本次合并无关，本次未改动其任何文件，留待归档操作的负责方更新索引。
