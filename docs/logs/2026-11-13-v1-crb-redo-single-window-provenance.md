# v1-crb-redo Single-window 配置来源核查

- 检查时间：2026-11-13 14:58:10 CST
- 实验代号：v1-crb-redo
- 范围：FEV/TIME 评测窗口模式及指标二样本量核对
- 状态：只读核查完成，未修改评测配置或已有结果。

## 证据

- FEV 通用模板在提交 9f2ff8d 中新增 `evaluation_mode: single_window`，注释说明该模式只取最新 cutoff，rolling 遍历声明的 cutoff。
- TIME 模板将 `single_window` 注明为默认模式，并说明不会把预测结果反馈为后续历史。
- TimerXL Trev 配置于提交 1631706 中沿用该模式；v1-crb-redo 的 FEV 配置于提交 1bc43ee 中沿用该模式。
- `docs/exp/v1-crb-redo/exp.md` 记录了沿用既定配置，未发现关于评测方差或滚动评测独立性的进一步决策记录。
- 当前 FEV single-window 适配器将声明的 20 个 cutoff 收敛到最后一个；FEV 9 有 8 条有效序列，因此当前共 8 个任务窗口。TIME 当前每条序列/通道一个窗口。
- 对 8 个可行任务，依据现有序列数、测试长度和预测长度推得：single-window 为 70 个窗口；全量 rolling 预计为 1,522 个，约 21.7 倍。TIME 窗口数按每条序列 `ceil(test_length / prediction_length)` 计算，FEV 按 8 条序列乘 20 个 cutoff 计算。

## 解释与后续约束

单 cutoff 降低时间覆盖，结果可能对该时间位置更敏感；rolling 增加时间覆盖，但同一序列的历史窗口重叠且目标序列有自相关，窗口数不等于独立重复数。若指标二采用 rolling，TIME 的安全历史必须重新截断到测试段第一个 cutoff 之前并重新准备梯度，当前 single-window 安全历史缓存不可直接复用。
