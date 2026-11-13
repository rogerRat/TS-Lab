# v1-crb-redo 结果文档收敛为八个兼容任务日志

时间：2026-11-13 CST
文件：`docs/exp/v1-crb-redo/results.md`
决策（用户指示）：舍弃不兼容任务，不汇报，只汇报八个任务。

## 1. 舍弃范围

四个安全历史不足 576 点、无法支持筛选的任务整体移出结果汇报：

- FEV M_DENSE_1D（原 FEV 11）
- FEV us_consumption_1Y（原 FEV 18）
- FEV world_co2_emissions（原 FEV 19）
- TIME ECDC_COVID/W（原 TIME 5）

保留的八个兼容任务：FEV hospital_admissions_1D；TIME CPHL/15T、CPHL/30T、CPHL/H、ECDC_COVID/D、epf_electricity_price/H、Commodity_Import/M、Commodity_Production/M。

## 2. results.md 变更

- MASE 主指标汇总表：删除四个舍弃任务行，现 8 个任务行，每行六个 arm 全部有值；删除基于含舍弃任务任务集计算的两个基线均值行（FEV 四任务平均、TIME 八任务平均），原四个均值（6.447586、6.703599、1.710087、1.736942）不再出现；RR 八任务宏平均行保留（其任务集恰好等于现汇报范围），基线两列以 — 标记。
- 两基线 MASE 差值表：12 行收敛为 8 行。
- 全量指标：FEV 仅保留 hospital_admissions_1D 小节（六方法行）；TIME 小节改为七个 short-term 任务，每任务六方法行；ECDC_COVID/W 块删除。
- 全部保留表格重新按同条件最小值加粗，并列最优同时加粗；RR 选择耗时表本就只含八任务，原样保留。
- 正文更新：开头说明只汇报八个兼容任务并列名四个舍弃任务；基线胜负统计按汇报范围改为 full 6 项、rand01 2 项；删除全部 — 占位与"放弃"表述（舍弃任务在正文以"已舍弃，不在本文汇报"说明）。
- 历史事实保留：正文注明两个基线批次历史上仍按原 12 任务协议完成评估，未汇报任务的产物保留在各自批次输出目录。

## 3. 未改动项

- `exp.md` 的 12tasks-ref-grad-prepare 设计表仍逐项记录四个任务的放弃原因，属设计与执行记录，不删改。
- `docs/EXP.md`、`docs/RESULTS.md` 批次表中"12任务评估"描述的是批次实际执行范围（12 项评估确实运行并留有产物），保持原样；汇报口径以 results.md 为准。

## 4. 校验

- 重新由两个源文件（RR-only 与基线报告的 /tmp 留存副本）经 `scripts/temp/merge_baselines.py` 生成，无手工转抄。
- `scripts/temp/validate_merge.py` 独立校验 0 错误：
  - 数值多重集差异恰好等于四个舍弃任务在基线源表中的全部单元格，无其他丢失、无新增数值；
  - 5 张表逐比较组复算最小值，加粗位置全部正确，差值表无加粗；
  - MASE 表 9 行（8 任务 + 1 宏平均），— 仅出现在宏平均行基线两列；FEV 1 小节 6 行；TIME 7 块各 6 行；差值表 8 行；
  - 无舍弃任务名、无数字任务代号、无"放弃"字样残留。
