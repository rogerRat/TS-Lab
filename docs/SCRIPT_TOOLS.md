# 脚本复用与迁移说明

## 通用入口

新实验优先使用 `scripts/tools` 中的配置驱动入口：

- `warmup.py`：短程训练检查，指定模型、预处理窗口数据和训练参数。
- `grad_cache.py`：对带 `windows` 和 `window_ids` 的 NPZ 计算并缓存投影梯度。
- `select.py`：在统一的分片 NPZ 数据格式上运行 `uni`、`ads` 或 `crb`。
- `train.py`：对窗口 NPZ 执行完整训练并导出模型；LoRA 从配置启用。
- `eval.py`：复用统一评估入口，支持 FEV、TIME、Quito。
- `run.py`：按实验清单串联多个阶段；阶段之间以子进程运行，失败即停止。
- `assets.py`：已有数据获取、加载、去重、预训练模型准备和梯度资源入口。

配置示例见 `configs/tools/README.md`。阶段工具读取 YAML；`--set section.field=value` 可覆盖已有配置或添加可选字段。输出写入 `outputs/`，运行日志写入 `logs/tools/`；成功阶段写入 `.complete`。

选择器输入使用 NPZ，其中 `values` 是一维拼接序列，`offsets` 是从 0 开始的分片边界，`fragment_ids` 和 `entity_ids` 可选。分片之间不会跨界生成窗口。CRB 输出参考分片对候选块的 top-score 矩阵；ADS 输出算法内部的 score-matrix 汇总。均匀抽样不产生分数矩阵。

## v1 中可合并的重复操作

| 重复操作 | 现有代表脚本 | 通用归属 |
|---|---|---|
| Source shard、offset 和合法窗口读取 | `v1-rage-ads/train_eval.py`、`v1-rage-ads-v2/train_eval.py`、`v1-rage-ads-v2/run_selection.py` | 后续抽取为统一 fragment/window 数据加载接口 |
| 模型导出、LoRA 合并和 manifest 写入 | 三个 v1 目录的 `train_eval.py` / `rr_train_eval.py` | `train.py` 及共享 checkpoint 导出函数 |
| 训练后评估子进程和结果完成校验 | 三个 v1 目录的训练评估脚本 | `eval.py` 及流水线阶段编排 |
| 并行 worker 启停、资源采样、失败回收 | v1 ADS 与 ADS v2 的 parallel runner | `parallel.py` 通用有界队列 |
| ADS probability-support 检查 | v1 ADS 与 ADS v2 的 audit 脚本 | 通用审计入口，协议特定字段由实验配置提供 |
| 任务队列和恢复执行 | 各 v1 目录的 queue/recover shell 脚本 | 配置驱动的 `run.py` / `parallel.py` |

## 保留实验专属逻辑

ADS v2 的反向细化、CRB redo 的块/窗口粒度选择、特定数据集的聚类与直方图、实验指标表和一次性 GPU benchmark 都含有协议假设，暂留对应实验目录。接入统一工具时只抽出稳定的数据、缓存、训练和评估接口；不以同名通用选项替换不同的算法定义。

legacy 与 temp 中的研究脚本按是否存在重复调用逐项迁移。一次性修复、历史复现、结果绘图和诊断脚本保留原位置；不批量删除或改写历史产物。


## 迁移边界

本次新增的通用入口已经可独立运行，但现有 v1 runner 和队列暂时保留原路径。v1 的 source sampler、任务清单和结果协议各不相同；直接替换会改变既有实验行为，部分 shell 队列也仍引用旧入口。后续迁移应先为这些数据与方法协议补配置适配器，再把队列改成新入口调用。legacy/temp 中的一次性分析继续作为历史复现工具保留。
