# v0-gradient-proxy-test

最后修改日期：2026-11-12

## v0-gradient-proxy-test

首次提出与最后执行时间、逐批次参数及原始修改记录见 [英文历史原文](../../../docs_legacy/EXP.md)。评估 32–576 起点区块内稀疏输出头代理对真实梯度方差和方向离散度的排序能力，并评估跨轨候选排序及目标离散程度。各批次完成；详情见 docs/exp/v0-gradient-proxy-test/results.md。2026-11-12 为给之后正式 v1 实验预留数据盘空间，已删除十条轨迹的未投影 FP32 输出头梯度数组（约50.14 GiB）；派生分析结果保留，删除范围见 outputs/train/v0-gradient-proxy-test/OUTPUT_HEAD_GRADIENTS_REMOVED.md。
