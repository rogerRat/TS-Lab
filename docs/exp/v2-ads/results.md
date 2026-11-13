# v2-ads 结果记录

## 批次：implementation-validation

- 状态：完成。
- 环境：Conda rage，Python 3.10.22，NumPy 2.2.6，PyTorch 2.7.1+cu128。
- 命令：/root/autodl-tmp/rage/envs/rage/bin/python -m pytest -q tests/test_ads.py tests/test_rage_run.py tests/test_selection_execution.py tests/test_region_delivery.py tests/test_package_layout.py
- 结果：53 passed；退出码 0。
- 覆盖内容：粗锚点端点和密度、eta、零范数余弦、全局 Top-K 并列顺序、候选区间扩展、细算最长间隔、四种 source 列聚合、有限距离衰减、全零概率回退、pipeline 去重与计数、目标密度模式、source 加权抽样、缓存锚点复用和 CLI/YAML 参数透传。
- 文件：测试实现位于 tests/test_ads.py；没有保留 pytest 临时报告文件。
- 模型效果指标：未运行正式训练评估，无 MASE 结果。

## 批次：ads-main-formal

- 状态：未开始。
- 配置：正式实验数据、模型、预算、训练、评估和种子尚未冻结。
- 指标与输出：无；不得据代码测试推断模型效果。
