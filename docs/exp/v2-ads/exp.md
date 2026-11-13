# v2-ads：自适应密度采样算法接入

## 基础信息

| 字段 | 内容 |
|---|---|
| 实验代号 | v2-ads |
| 目标 | 实现 docs/ads.md 的四阶段 ADS，并接入当前 source/reference 梯度选择和主表实验流程，供后续替换 CRB |
| 假设 | source 与 reference 的真实梯度处于相同投影坐标系；Top-K 余弦相似度可用于确定 source 锚点分数 |
| 文档创建时间 | 2026-11-14 01:02 +0800 |
| 最后修改时间 | 2026-11-14 |
| 实际实验启动时间 | 尚未启动正式 ADS 训练或评估 |
| 当前状态 | 算法、选择器、配置和 CLI 接入完成；代码测试通过；正式实验参数待确认 |

## 模型与训练

ADS 算法 API 位于 src/rage/ads.py，主表调用通过 src/rage/experiments/arms.py::ads_select。选择梯度使用主表当前的 Chronos-T5 探针以及同一 batch_gradient_sketch 投影路径。

configs/rage/ads.yaml 是选择阶段的开发模板，不是正式实验冻结配置：

| 项目 | 模板值 | 状态 |
|---|---|---|
| 探针模型 | amazon/chronos-t5-small | 主表默认值；正式模型待确认 |
| context / prediction | 512 / 64 | 开发模板值，正式协议待确认 |
| 梯度投影维度 | 1024 | 开发模板值，正式协议待确认 |
| 训练与下游评估 | selection_only: true | 模板只运行数据选择，不启动学生训练 |
| Conda 环境 | rage | 项目主环境 |

## 数据处理

开发配置使用 ETT_1H，reference 序列上限 2，source pool 为 2560 个非重叠窗口当量，pool seed 为 0。source 和 reference 的梯度位置均由 (context_length, prediction_length, pool_stride) 的合法窗口起点确定。

该数据子集只用于接入验证。正式实验的数据集、reference 序列数量、source pool 规模和数据版本均待确认；未启动正式数据准备。

## ADS 参数与待确认决策

| 参数 | 开发模板值 | 说明与状态 |
|---|---:|---|
| ads_q_coarse | 32 | 粗阶段目标间隔；ADS 文档未给固定默认值，正式值待确认 |
| ads_q_fine | 8 | 细阶段目标间隔；正式值待确认 |
| ads_eta | 1.0 | reference/source 目标密度倍率；正式值待确认 |
| ads_top_k | 64 | 全局 Top-K；与 ads_top_ratio、ads_target_fine_density 互斥；正式预算模式待确认 |
| ads_top_ratio | 未设置 | 可选比例预算 |
| ads_target_fine_density | 未设置 | 可选目标细算密度，搜索 K 并最小化实际密度差，平局取较小 K |
| ads_rounds | 1 | 细算与重匹配轮数；与 ADS 文档默认一致 |
| ads_decay_distance | 4 | 单侧线性衰减距离；与 ADS 文档默认一致 |
| ads_aggregation | max | source 列分数聚合模式；与 ADS 文档默认一致 |
| ads_sampling_replacement | true | 开发模板按概率放回抽取，避免稀疏权重的正支持集小于训练样本数；正式抽样策略待确认 |
| ads_selection_seed | 101 | 开发模板抽样种子；正式种子待确认 |

上述开发模板值不是正式实验确认值。正式运行前需冻结模型、数据、预算模式、q 参数、抽样数量/放回策略、训练日程、随机种子和评估任务。

## 评估方案

本次仅验证算法输出、梯度计数、采样概率和接口透传。未运行正式训练或评估，没有 MASE 或其他模型效果指标。

## 执行细节

算法函数接口：

- rage.ads.ADSConfig
- rage.ads.ADSPipeline(config, gradient_computer, context_length=..., prediction_length=..., window_stride=..., gradient_cache=...).run(reference_fragments, source_fragments). Cache keys are (side, fragment_id, physical_start).
- rage.experiments.arms.ads_select(...)：将 ADS 概率映射到 WindowIndex，按调用方抽样设置生成固定窗口选择集。

开发模板脚本：

    tmux new -s ads-dev 'bash scripts/exp/v2-ads/run_ads.sh'

从项目 Conda rage 环境启动，执行 python -u -m rage.experiments.run --config configs/rage/ads.yaml。ADS 参数可通过配置文件设置，也可覆盖，例如：

    bash scripts/exp/v2-ads/run_ads.sh --ads-q-coarse 16 --ads-q-fine 4 --ads-top-ratio 0.05

调试配置和运行冻结文件位于 configs/rage/ads.yaml 与 outputs/rage-<batch>-<timestamp>/；运行日志位于 logs/rage_runs/<run_id>/。完整实验输出实际写入项目 outputs/ 路径。

## 批次管理

| 批次 | 方法与工作 | 配置 / 命令 | 状态 | 输出 |
|---|---|---|---|---|
| implementation-validation | ADS 单元、端到端接口及配置/CLI 流程测试 | tests/test_ads.py；/root/autodl-tmp/rage/envs/rage/bin/python -m pytest -q tests/test_ads.py tests/test_rage_run.py tests/test_selection_execution.py tests/test_region_delivery.py tests/test_package_layout.py | 完成：53 项通过 | 测试代码见 tests/test_ads.py；pytest 临时报告未保留为实验产物 |
| ads-main-formal | 正式 ADS 选择、训练与评估 | 正式配置尚未冻结 | 未开始 | 无输出、无指标 |
