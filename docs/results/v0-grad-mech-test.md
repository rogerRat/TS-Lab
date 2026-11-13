# v0-grad-mech-test

最后修改日期：2026-11-13

## v0-grad-mech-test

完成时间：2026-11-07。探针使用固定 warm-up checkpoint 和 FP32 梯度处理了全部 1,000 条选定的 M5 轨迹。窗口覆盖 512 个上下文点和 64 个预测点，stride 为 1。下面的统计量按窗口配对计数加权；SD 为总体标准差。

| Lag | 配对计数 | 余弦均值 | 余弦 SD | 相对范数差异均值 | 相对范数差异 SD |
|---:|---:|---:|---:|---:|---:|
| 1 | 1143885 | 0.753878 | 0.253848 | 0.128111 | 0.135564 |
| 8 | 1136885 | 0.464294 | 0.249467 | 0.231456 | 0.170033 |
| 32 | 1112885 | 0.239849 | 0.268146 | 0.321463 | 0.213633 |
| 64 | 1080885 | 0.088391 | 0.268003 | 0.366607 | 0.233860 |
| 128 | 1016885 | 0.043590 | 0.237899 | 0.382731 | 0.242262 |
| 288 | 858074 | 0.024572 | 0.214649 | 0.393459 | 0.247553 |
| 575 | 581872 | 0.013485 | 0.195253 | 0.402461 | 0.253443 |

在本数据集中，余弦相似度随 lag 增大而下降，并在 lag 575 处趋近于零。这描述的是观测到的聚合结果，并不表示单对样本的余弦为零，也不表示该模式可推广到其他数据集或模型。配对计数随 lag 不同而变化，且同一轨迹内的观测相互依赖。

产物：archive/v0-grad-mech-test/outputs/train/v0-grad-mech-test/；完整可视化位于 vis/snapshot_20261107_123811_utc_n1000/。仅归档探针结果和可视化；其原始路径为符号链接。数据、日志、配置、脚本、checkpoint、输入清单以及 CRB 所有的梯度缓存仍保留在原位置。

### 数据集、轨迹索引和结果文件

- 数据集：autogluon/chronos_datasets，m5 配置，revision eeecad0b82a8c237e212ce6f8d1abecb513e2cec。按源顺序取前 1,000 条合格、有限、非常量、长度至少 576 的序列；不做随机采样。
- 序列 ID、从零开始的源行号、长度和从零开始的清单序号：archive/v0-grad-mech-test/results/trajectory_index.csv。完整的 input_manifest.json 已复制到同一目录。
- 逐轨迹逐 lag 统计：archive/v0-grad-mech-test/results/trajectory_lag_similarity.csv.gz，包含 575,000 行。列给出配对计数、余弦均值、总体 SD、最小值和最大值，以及相对范数差异均值和总体 SD。指标为空表示没有配对。未保留单对数值。
- 原始逐轨迹摘要和直方图：archive/v0-grad-mech-test/outputs/train/v0-grad-mech-test/probe/trajectories/trajectory-NNNN.npz；NNNN 为从零开始的清单序号。
- 汇总的 575-lag CSV 和 JSON，以及 PNG、SVG 和 HTML 图表：archive/v0-grad-mech-test/outputs/train/v0-grad-mech-test/vis/snapshot_20261107_123811_utc_n1000/。汇总均值按配对计数对轨迹加权。
- 已在每个 lag 上对照汇总配对计数和余弦均值验证了导出的表格。


## 数据集、轨迹索引和结果文件

- 数据集：autogluon/chronos_datasets，m5 配置，revision eeecad0b82a8c237e212ce6f8d1abecb513e2cec。按源顺序取前 1,000 条合格、有限、非常量、长度至少 576 的序列；不做随机采样。
- 序列 ID、从零开始的源行号、长度和从零开始的清单序号：archive/v0-grad-mech-test/results/trajectory_index.csv。完整的 input_manifest.json 已复制到同一目录。
- 逐轨迹逐 lag 统计：archive/v0-grad-mech-test/results/trajectory_lag_similarity.csv.gz，包含 575,000 行。列给出配对计数、余弦均值、总体 SD、最小值和最大值，以及相对范数差异均值和总体 SD。指标为空表示没有配对。未保留单对数值。
- 原始逐轨迹摘要和直方图：archive/v0-grad-mech-test/outputs/train/v0-grad-mech-test/probe/trajectories/trajectory-NNNN.npz；NNNN 为从零开始的清单序号。
- 汇总的 575-lag CSV 和 JSON，以及 PNG、SVG 和 HTML 图表：archive/v0-grad-mech-test/outputs/train/v0-grad-mech-test/vis/snapshot_20261107_123811_utc_n1000/。汇总均值按配对计数对轨迹加权。
- 已在每个 lag 上对照汇总配对计数和余弦均值验证了导出的表格。
