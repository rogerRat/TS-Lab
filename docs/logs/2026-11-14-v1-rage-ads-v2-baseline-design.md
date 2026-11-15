# v1-rage-ads 基线设计澄清

- 检查时间：2026-11-14 CST
- 实验代号：v1-rage-ads
- 批次：baseline-design-revision
- 状态：设计已澄清；实现与评测未开始
- 检查范围：远程 `/root/TS-Lab` 的实验文档、选择/训练入口及工作区状态

## 操作与结论

- 记录用户确认的五种方法定义：`traj` 即 `uni-traj`；`uni` 与 ADS-VIC 等梯度总预算且只取消二次精细计算；ADS 维持原设计并称 ADS-VIC（vicinity）；`orig` 不训练直接评测；`reft`（ref-trained）只在 reference 集合训练。
- 将旧版已完成的 `rolling-eight-task` 与修订版基线区分。旧运行使用 `fullRand`、`rage`、`ancUni`、`Traj`，其历史数值不代表新定义。
- 更新 `docs/exp/v1-rage-ads/exp.md`、`docs/exp/v1-rage-ads/results.md`，并将修订版待执行批次加入全局索引。
- 未修改代码、配置、训练产物或既有指标；未启动新任务。

## 待办

- 实现修订版五个方法的选择/训练/评测入口与配置。
- 记录并确认统一训练预算、`reft` 输入构造和各方法的命令与产物路径。
- 完成新批次后追加真实结果和 `.complete` 标记。

## 范围更正

本次补充确认：旧 `v1-rage-ads` 批次已经结束，本次新定义属于独立实验 `v1-rage-ads-v2`。现已撤回本日志先前指向 v1 的修订记录位置，v1 的实验文档和结果恢复为只记录旧批次；新设计与待办记录在 `docs/exp/v1-rage-ads-v2/`，全局索引使用新实验代号。v1 输出与历史指标未改写。
