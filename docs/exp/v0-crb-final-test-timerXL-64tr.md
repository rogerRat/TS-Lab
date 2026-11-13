# v0-crb-final-test-timerXL-64tr

最后修改日期：2026-11-12

## v0-crb-final-test-timerXL-64tr

首次提出：2026-11-12。用户澄清q表示平均每q个有效窗口起点计算一次锚点。固定q={2,4,6,8,12,16,24,32,64}；每轨N个有效起点时锚点预算K=max(2,ceil(N/q))，Uniform与CRB严格匹配K并包含首尾锚点。CRB三个串行臂为coverage:error:split=1:0:1、1:1:0、1:0:0；按整数最大余数法分配K-2个额外锚点，余数相同优先coverage。复用冻结的CRB coverage、积分二次插值风险error、递归min-kappa split算法；Uniform为K个含端点的等距起点。

相邻锚点之间的闭区间定义为一个块，共享锚点计入左右两个块。对端点梯度作线性重建，以块内线性估计向量的均值作代表；每个有效窗口起点计算1-cos(真实梯度,代表)，按块内起点数加权。输入是已有64轨4096D BF16 LoRA投影梯度，共1,980,371个有效起点，不重新计算模型梯度。向量计算使用BF16量化的源值，不使用FP64。正确输出位于outputs/train/v0-crb-final-test-timerXL-64tr/spacing-q/，三个臂及顶层完成标记均已核验。结果见docs/results/v0-crb-final-test-timerXL-64tr.md。

此前将q误读为每轨锚点总数生成的旧产物已删除，不作为结果。正确spacing-q队列于2026-11-12 01:43在tmux v0-crb-final-spacing-q启动，02:12三个臂完成。

## v0 汇总与归档核对（2026-11-12）

本次按实际结果整理全部 v0 专题，汇总入口为 [v0 结果总览](../results/v0-consolidated-summary-20261112.md)，批次级结果仍以本文件前文和各专题报告为准。ordinal 0 ChronosTiny 批次完成；非临近窗口对余弦 Spearman=0.0758，相邻窗口对余弦 Spearman=-0.1379。64 轨 ChronosTiny 扩展已按用户调整停止，不把未完成缓存当作结果。当前进程检查未发现 v0 实验计算进程；已有批次按完成标记、日志和专题报告分别标为完成、停止或部分完成。

2026-11-12 已将所有非缓存 v0/v00 输出文件与日志归入 archive/v0-consolidated-20261112/。为维持报告中记录的输出路径可访问，原位置保留指向归档文件的符号链接。grad_cache/、gradients/ 等可供后续分析的梯度缓存原位保留，包括未完成批次的部分缓存。v0 配置、脚本及报告副本存于归档 source-and-report-snapshot/；项目源代码、配置与本文件继续留在原处。归档文件数量、字节数和缓存保留规则见归档 README.md。

最后修改日期：2026-11-12

### v0-crb-final-test-timerXL-64tr-spacing-q-dir

首次提出：2026-11-12；最后修改日期：2026-11-12。输入沿用 spacing-q 的64轨、1,980,371个有效窗口、4096D BF16 LoRA投影梯度缓存，不重新计算梯度。全局一次性以 float32 计算全部梯度模长，取第1百分位阈值；模长小于或等于阈值的真实方向置零，记录阈值、置零数量和比例。其余梯度除以各自模长，后续 CRB 选点、Uniform 对照、端点插值和离散度计算全部使用方向向量。零真实方向或零重建代表的余弦定义为0，因此对应误差为1。

q={2,4,6,8,12,16,24,32,64}，每轨 K=max(2,ceil(N/q))，包含首尾锚点；Uniform 按相同 K 等距取点。CRB coverage:error:split 依次为 1:0:0、1:0:1、1:1:0、1:1:1，按原最大余数法分配 K-2 个额外锚点。闭区间块包含两个端点，相邻块共享锚点。用两个端点的单位方向线性插值，块代表为插值向量的均值；逐窗口计算 1-cos(真实方向,块代表)。同时报告按块长度加权的误差与各块等权的误差，保留逐轨、逐块和总体结果。计算沿用 BF16 源值及 float32 归约。

| 批次 | 工作 | 状态 | 产物 |
|---|---|---|---|
| 阈值与四比例队列 | 全局第1百分位阈值；顺序运行四种 CRB 比例及匹配 Uniform | 全部完成，四比例及顶层完成标记已核验 | outputs/train/v0-crb-final-test-timerXL-64tr/spacing-q-dir/ |

入口 scripts/temp/v0_crb_final_test_timerxl_64tr_dir.py；队列 scripts/temp/queue_v0_crb_final_test_timerxl_64tr_dir.sh；环境 rage-timerxl。各比例以及顶层 .complete 生成后才视为完成。禁止自动巡检。

### spacing-q-dir q=128,256 追加批次
首次提出：2026-11-12；最后修改日期：2026-11-12。沿用原方向阈值、64轨缓存、四种CRB比例、匹配Uniform、方向线性重建与两种加权误差，仅将q改为128和256。沿用原预算公式 K=max(2,ceil(N/q))，每轨含首尾锚点；实际相邻锚点平均间距为(N-1)/(K-1)，CRB局部间距可不等。新产物独立写入 outputs/train/v0-crb-final-test-timerXL-64tr/spacing-q-dir/extra-q-128-256/，每比例及顶层完成标记为完成条件。状态：已完成，四比例及顶层完成标记均已核验；结果见 docs/results/v0-crb-final-test-timerXL-64tr-spacing-q-dir.md。
