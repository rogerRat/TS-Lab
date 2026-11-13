# v1-crb-redo 结果文档格式修订日志

时间：2026-11-13 CST
文件：`docs/exp/v1-crb-redo/results.md`
性质：机械格式修订，不改动任何指标数值、表格结构与结论。

## 1. 修订内容

### 1.1 任务代号改为完整任务名

按 `docs/exp/v1-crb-redo/exp.md` 中 12tasks-ref-grad-prepare 的锁定映射替换：

| 旧标识 | 新标识 |
|---|---|
| FEV 9 | FEV hospital_admissions_1D |
| TIME 1 / TIME 01 | TIME CPHL/15T |
| TIME 2 / TIME 02 | TIME CPHL/30T |
| TIME 3 / TIME 03 | TIME CPHL/H |
| TIME 4 / TIME 04 | TIME ECDC_COVID/D |
| TIME 6 / TIME 06 | TIME epf_electricity_price/H |
| TIME 7 / TIME 07 | TIME Commodity_Import/M |
| TIME 8 / TIME 08 | TIME Commodity_Production/M |

输出目录名 `fev-09`、`time-01` 等路径字符串保持原样；完整代号-名称-目录对照见同实验 `exp.md`。小节标题 `### FEV 9` 改为 `### FEV hospital_admissions_1D`，`### TIME 任务` 改为 `### TIME 八个 short-term 任务`。

### 1.2 同条件最优方法加粗

全部指标均为误差/损失类，越小越优；选择耗时越短越优。

- traj/block MASE 汇总表：逐任务、宏平均行，在四个 arm 中加粗最小 MASE。
- FEV 全指标表：逐指标列在四个方法行中加粗最小值（本任务六项均为 CRB traj）。
- TIME 全指标表：按任务分块，逐指标列在该任务四个方法行中加粗最小值。
- 选择耗时表：逐任务加粗最短耗时；相同耗时并列加粗。
- 全部数值保留原精度，仅添加 Markdown `**` 标记。

## 2. 校验

- 脚本：`scripts/temp/bold_v1_crb_redo.py`（生成）、`scripts/temp/validate_v1_crb_redo.py`（校验），均不纳入 git。
- 修订前后行数一致；逐行核对除加粗标记与任务改名外无任何内容漂移。
- 独立复算四张表全部同条件比较的最小值，加粗位置与复算结果 0 处不符；并列最小值均正确并列加粗。
- 除输出路径外，文档中不再出现纯数字任务代号。
