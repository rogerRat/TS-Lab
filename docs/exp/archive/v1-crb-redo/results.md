# v1-crb-redo 实验结果

最后修改日期：2026-11-13 20:21 CST

MASE 为主指标；每个任务、每个方法均完整汇报评测配置实际产生的全部指标。所有数值均来自已完成单任务评估产物。本文只汇报八个兼容任务（1 个 FEV 加 7 个 TIME）；四个安全历史不足 576 点、无法支持筛选的任务（FEV M_DENSE_1D、FEV us_consumption_1Y、FEV world_co2_emissions、TIME ECDC_COVID/W）已舍弃，不在本文汇报。两个基线批次历史上仍按原 12 任务协议完成评估，未汇报任务的产物保留在各自批次输出目录中。TopK 聚合方案不属于本轮 RR 执行。

## 批次与方法

本文合并同一实验的两个基线、四个 traj/block RR arm 和两个 win RR arm；CRB win 已对用户限定的五项完成选样、训练和评估；其余 win 任务按用户决定不执行。两个基线批次 `timerXL-baseline-full`（表中 baseline full）与 `timerXL-baseline-rand01`（表中 baseline rand01）各完成 4,000 次优化更新，模型均从原始 TimerXL 预训练权重独立开始，单一 seed 0；full 使用 src-96 全部 2,666,312 个合法窗口，rand01 以 seed 0 固定抽取其中 266,631 个合法窗口，两批各累计抽样 128,000 次窗口曝光，随机批次可以重复抽到同一固定窗口。四个 RR arm（Uniform traj、Uniform block、CRB traj、CRB block）的方法定义与训练协议见同实验 `exp.md`。

基线原始指标文件：`outputs/eval/v1-crb-redo/<批次>/fev/fev_results.csv` 与 `outputs/eval/v1-crb-redo/<批次>/time/<数据集>/<频率>/short/metrics.npz`。FEV MASE 直接取 CSV 的任务行；TIME MASE 对各任务 `metrics.npz` 的 `MASE` 数组在序列、窗口和变量轴求算术均值。训练清单为 `outputs/train/v1-crb-redo/<批次>/training_manifest.json`；随机窗口索引为 `outputs/train/v1-crb-redo/timerXL-baseline-rand01/fixed_window_indices.npy`。FEV 本地数据源修订 `d3e7e720f4158eab03ebcb021f1cce779cd92acc`；TIME 数据源修订 `83e3d0b3be28d11c7182bffcc1892d19b36c4da1`。

## MASE 主指标汇总

| 任务 | baseline full | baseline rand01 | Uniform traj | Uniform block | CRB traj | CRB block | Uniform win | CRB win | 全源方向 cos（CRB / Uniform） | 筛选后方向 cos（top10% RR 四 arm 均值） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FEV hospital_admissions_1D | 0.667800 | 0.668297 | 0.666990 | 0.668801 | **0.665673** | 0.672526 | 0.670380 | 0.675108 | +0.00172246 / +0.00027066 | 0.00618355 (4/4) |
| TIME CPHL/15T | **1.105670** | 1.120716 | 1.287051 | 1.201087 | 1.288009 | 1.181190 | 1.208172 | 1.213646 | +0.00029571 / +0.00088666 | 0.00863612 (4/4) |
| TIME CPHL/30T | 0.834222 | **0.817478** | 1.097394 | 1.189182 | 1.058316 | 1.140804 | 1.025688 | 1.134415 | +0.00379170 / -0.00048141 | 0.00923204 (4/4) |
| TIME CPHL/H | 1.536111 | 1.552243 | **1.401344** | 1.496067 | 1.549088 | 1.423393 | 1.441965 | 1.455521 | +0.00032629 / -0.00040733 | 0.00181076 (4/4) |
| TIME ECDC_COVID/D | 3.154493 | 3.235430 | 3.899349 | 3.186459 | 3.496406 | 3.143595 | 3.120710 | 3.167945 | +0.00064966 / +0.00132277 | 0.02182099 (4/4) |
| TIME epf_electricity_price/H | 1.375741 | 1.382876 | **1.163970** | 1.167872 | 1.302919 | 1.210748 | — | — | +0.00075751 / +0.00041224 | 0.00260055 (4/4) |
| TIME Commodity_Import/M | **1.997317** | 2.041115 | 2.095419 | 2.069385 | 2.154830 | 2.131106 | — | — | -0.00766274 / -0.00554982 | 0.00657378 (4/4) |
| TIME Commodity_Production/M | 1.257721 | 1.256141 | **1.201335** | 1.242544 | 1.205709 | 1.227719 | — | — | -0.00017508 / -0.00098100 | 0.02680804 (4/4) |
| 八任务等权宏平均（RR 四 arm） | — | — | 1.601606 | 1.527675 | 1.590119 | **1.516385** | — | — | -0.00003681 / -0.00056590 | 0.01045823 |

RR 宏平均是本文八个汇报任务 MASE 的算术平均，不作任务尺度归一化；跨 FEV 与 TIME 的任务尺度不同，比较应同时查看逐任务结果。基线对比中，在汇报的 8 项里 full 在 6 项、rand01 在 2 项 MASE 更低；单一 seed 和这些任务不足以估计统计显著性。

### 两基线 MASE 差值（rand01 - full，正值表示 full 误差更低）

| 任务 | rand01 - full | — |
| --- | ---: | --- |
| FEV hospital_admissions_1D | +0.000497 | — |
| TIME CPHL/15T | +0.015046 | — |
| TIME CPHL/30T | -0.016744 | — |
| TIME CPHL/H | +0.016132 | — |
| TIME ECDC_COVID/D | +0.080937 | — |
| TIME epf_electricity_price/H | +0.007135 | — |
| TIME Commodity_Import/M | +0.043798 | — |
| TIME Commodity_Production/M | -0.001580 | — |
## 全量评测指标

下表逐任务列出全部已完成 arm 的评测指标。MASE 作为主指标；其他指标同样完整保留。FEV 数值来自各任务 `fev_results.csv`；TIME 的任务级数值为对应 `metrics.npz` 各输出值的算术平均，保持已有结果表的任务内汇总口径。SQL、WAPE、WQL 及 ND、CRPS 即使与其他指标数值相同也照常汇报。

### FEV hospital_admissions_1D

| 方法 | MAE | MASE | MSE | SQL | WAPE | WQL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline full | 2.976004 | 0.667800 | 14.818351 | 0.667800 | 0.491611 | 0.491611 |
| baseline rand01 | 2.978237 | 0.668297 | 14.876325 | 0.668297 | 0.491980 | 0.491980 |
| Uniform traj | 2.972377 | 0.666990 | 14.676767 | 0.666990 | 0.491012 | 0.491012 |
| Uniform block | 2.980329 | 0.668801 | 14.774854 | 0.668801 | 0.492326 | 0.492326 |
| CRB traj | 2.966797 | **0.665673** | 14.570810 | 0.665673 | 0.490090 | 0.490090 |
| CRB block | 2.997070 | 0.672526 | 14.850407 | 0.672526 | 0.495091 | 0.495091 |
| Uniform win | 2.987863 | 0.670380 | 14.890307 | 0.670380 | 0.493570 | 0.493570 |
| CRB win | 3.008650 | 0.675108 | 14.805036 | 0.675108 | 0.497004 | 0.497004 |

### TIME 七个 short-term 任务

| 任务 | 方法 | MSE | MAE | RMSE | MAPE | sMAPE | MASE | ND | CRPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TIME CPHL/15T | baseline full | 0.138391 | 0.258444 | 0.299595 | 0.269856 | 0.219902 | 1.105670 | 0.239620 | 0.239620 |
| TIME CPHL/15T | baseline rand01 | 0.141230 | 0.261374 | 0.303125 | 0.273977 | 0.223242 | 1.120716 | 0.243218 | 0.243218 |
| TIME CPHL/15T | Uniform traj | 0.190050 | 0.302854 | 0.345306 | 0.313210 | 0.246507 | 1.287051 | 0.277774 | 0.277774 |
| TIME CPHL/15T | Uniform block | 0.172181 | 0.284919 | 0.326609 | 0.291656 | 0.229585 | 1.201087 | 0.257906 | 0.257906 |
| TIME CPHL/15T | CRB traj | 0.185298 | 0.302366 | 0.342665 | 0.311943 | 0.247917 | 1.288009 | 0.278391 | 0.278391 |
| TIME CPHL/15T | CRB block | 0.158556 | 0.277324 | 0.318275 | 0.286933 | 0.230629 | 1.181190 | 0.255283 | 0.255283 |
| TIME CPHL/15T | Uniform win | 0.168375 | 0.284486 | 0.325725 | 0.293614 | 0.233783 | 1.208172 | 0.260640 | 0.260640 |
| TIME CPHL/15T | CRB win | 0.166736 | 0.285439 | 0.325090 | 0.294275 | 0.235183 | 1.213646 | 0.262014 | 0.262014 |
| TIME CPHL/30T | baseline full | 0.117076 | 0.240963 | 0.293927 | 0.096473 | 0.094816 | 0.834222 | 0.093056 | 0.093056 |
| TIME CPHL/30T | baseline rand01 | 0.130010 | 0.234316 | 0.308247 | 0.093577 | 0.092870 | 0.817478 | 0.091056 | 0.091056 |
| TIME CPHL/30T | Uniform traj | 0.131545 | 0.278892 | 0.317494 | 0.123597 | 0.119484 | 1.097394 | 0.119636 | 0.119636 |
| TIME CPHL/30T | Uniform block | 0.139014 | 0.305161 | 0.328794 | 0.139633 | 0.129128 | 1.189182 | 0.129857 | 0.129857 |
| TIME CPHL/30T | CRB traj | 0.125853 | 0.279205 | 0.312063 | 0.121359 | 0.116184 | 1.058316 | 0.116122 | 0.116122 |
| TIME CPHL/30T | CRB block | 0.137244 | 0.277687 | 0.326392 | 0.129411 | 0.123612 | 1.140804 | 0.123476 | 0.123476 |
| TIME CPHL/30T | Uniform win | 0.122765 | 0.267271 | 0.306953 | 0.118605 | 0.113301 | 1.025688 | 0.112300 | 0.112300 |
| TIME CPHL/30T | CRB win | 0.126028 | 0.288657 | 0.314397 | 0.133282 | 0.123503 | 1.134415 | 0.123698 | 0.123698 |
| TIME CPHL/H | baseline full | 0.546750 | 0.468257 | 0.511013 | 0.219462 | 0.236956 | 1.536111 | 0.215186 | 0.215186 |
| TIME CPHL/H | baseline rand01 | 0.571016 | 0.477667 | 0.520279 | 0.222262 | 0.241163 | 1.552243 | 0.218727 | 0.218727 |
| TIME CPHL/H | Uniform traj | 0.395754 | 0.402545 | 0.456170 | 0.211255 | 0.218977 | 1.401344 | 0.202951 | 0.202951 |
| TIME CPHL/H | Uniform block | 0.489673 | 0.445626 | 0.486303 | 0.220790 | 0.232405 | 1.496067 | 0.212770 | 0.212770 |
| TIME CPHL/H | CRB traj | 0.482868 | 0.450731 | 0.504179 | 0.222962 | 0.242860 | 1.549088 | 0.222007 | 0.222007 |
| TIME CPHL/H | CRB block | 0.433096 | 0.417932 | 0.462604 | 0.211808 | 0.219372 | 1.423393 | 0.201904 | 0.201904 |
| TIME CPHL/H | Uniform win | 0.430019 | 0.419303 | 0.463921 | 0.217003 | 0.223474 | 1.441965 | 0.205883 | 0.205883 |
| TIME CPHL/H | CRB win | 0.466338 | 0.433120 | 0.478670 | 0.214794 | 0.225536 | 1.455521 | 0.206914 | 0.206914 |
| TIME ECDC_COVID/D | baseline full | 192129.571427 | 204.597454 | 240.400180 | 2.092795 | 0.587243 | 3.154493 | 1.524695 | 1.524695 |
| TIME ECDC_COVID/D | baseline rand01 | 195028.869980 | 207.670139 | 242.445139 | 2.065119 | 0.604631 | 3.235430 | 1.526144 | 1.526144 |
| TIME ECDC_COVID/D | Uniform traj | 165858.058348 | 226.388194 | 263.346875 | 3.402330 | 0.576917 | 3.899349 | 2.344773 | 2.344773 |
| TIME ECDC_COVID/D | Uniform block | 151898.982784 | 197.330556 | 226.043018 | 2.120741 | 0.574681 | 3.186459 | 1.551759 | 1.551759 |
| TIME ECDC_COVID/D | CRB traj | 119048.386053 | 190.757407 | 226.143287 | 3.552395 | 0.595630 | 3.496406 | 2.396482 | 2.396482 |
| TIME ECDC_COVID/D | CRB block | 101633.589468 | 175.615278 | 201.136717 | 2.337210 | 0.572829 | 3.143595 | 1.697881 | 1.697881 |
| TIME ECDC_COVID/D | Uniform win | 116944.705700 | 180.480093 | 207.199930 | 2.254801 | 0.586423 | 3.120710 | 1.649478 | 1.649478 |
| TIME ECDC_COVID/D | CRB win | 112115.759693 | 180.900000 | 207.979900 | 2.339703 | 0.578688 | 3.167945 | 1.699007 | 1.699007 |
| TIME epf_electricity_price/H | baseline full | 54.162761 | 5.522399 | 6.409452 | 2.239720 | 0.299961 | 1.375741 | 0.361438 | 0.361438 |
| TIME epf_electricity_price/H | baseline rand01 | 54.701543 | 5.472399 | 6.351008 | 2.186305 | 0.296971 | 1.382876 | 0.343114 | 0.343114 |
| TIME epf_electricity_price/H | Uniform traj | 43.888875 | 5.060483 | 6.161979 | 1.694963 | 0.282667 | **1.163970** | 0.307070 | 0.307070 |
| TIME epf_electricity_price/H | Uniform block | 40.391272 | 4.934899 | 5.800717 | 2.229385 | 0.289455 | 1.167872 | 0.341818 | 0.341818 |
| TIME epf_electricity_price/H | CRB traj | 50.400103 | 5.138066 | 6.093946 | 2.189542 | 0.284162 | 1.302919 | 0.320326 | 0.320326 |
| TIME epf_electricity_price/H | CRB block | 44.187023 | 5.215983 | 6.149759 | 2.221628 | 0.290887 | 1.210748 | 0.346676 | 0.346676 |
| TIME Commodity_Import/M | baseline full | 4525458.350058 | 900.280426 | 1036.476283 | 0.700766 | 0.330515 | **1.997317** | 0.475944 | 0.475944 |
| TIME Commodity_Import/M | baseline rand01 | 4549422.864343 | 907.538838 | 1044.413717 | 0.688761 | 0.331094 | 2.041115 | 0.468856 | 0.468856 |
| TIME Commodity_Import/M | Uniform traj | 4176964.268288 | 869.775572 | 999.826382 | 0.538577 | 0.310218 | 2.095419 | 0.376661 | 0.376661 |
| TIME Commodity_Import/M | Uniform block | 4810920.522276 | 931.082250 | 1058.649960 | 0.646518 | 0.331738 | 2.069385 | 0.446489 | 0.446489 |
| TIME Commodity_Import/M | CRB traj | 4292896.896951 | 899.778006 | 1021.413034 | 0.628295 | 0.328065 | 2.154830 | 0.438261 | 0.438261 |
| TIME Commodity_Import/M | CRB block | 4653954.353352 | 919.036443 | 1045.203994 | 0.682229 | 0.333372 | 2.131106 | 0.466981 | 0.466981 |
| TIME Commodity_Production/M | baseline full | 197142.510395 | 117.638258 | 130.309849 | 0.349769 | 0.201019 | 1.257721 | 0.181454 | 0.181454 |
| TIME Commodity_Production/M | baseline rand01 | 197025.026060 | 116.979791 | 129.749526 | 0.349717 | 0.201291 | 1.256141 | 0.181382 | 0.181382 |
| TIME Commodity_Production/M | Uniform traj | 181936.637136 | 110.187876 | 123.219167 | 0.337313 | 0.195834 | **1.201335** | 0.177529 | 0.177529 |
| TIME Commodity_Production/M | Uniform block | 192630.132183 | 114.032575 | 126.512567 | 0.345790 | 0.199138 | 1.242544 | 0.180757 | 0.180757 |
| TIME Commodity_Production/M | CRB traj | 179583.650637 | 111.818251 | 124.730217 | 0.338187 | 0.197782 | 1.205709 | 0.179369 | 0.179369 |
| TIME Commodity_Production/M | CRB block | 182936.724942 | 110.915519 | 123.624205 | 0.343513 | 0.196915 | 1.227719 | 0.178813 | 0.178813 |

所有结果值均读取自实际评估产物。FEV 的 SQL、WAPE、WQL 和 TIME 的 ND、CRPS 均按产物分别报告，即使某些任务中数值重合也不合并。后续 win 批次及本实验后续新增评测均按同一规则，在本文档逐任务追加评测配置产生的全部指标；未完成批次不预填或推测指标。

## RR 选择耗时
| --- | --- | --- | --- | --- | --- | --- | --- |
| 任务 | Uniform traj | Uniform block | CRB traj | CRB block  | Uniform win (BF16) | CRB win (BF16) |
| --- | --- | --- | --- | --- | --- | --- |
| FEV hospital_admissions_1D | 25.6秒 | 30.2秒 | 22.6秒 | 30.1秒 | 15分05秒 | 20分55秒 |
| TIME CPHL/15T | 0.4秒 | 3.6秒 | 0.4秒 | 3.6秒 | 9分09秒 | 8分56秒 |
| TIME CPHL/30T | 0.4秒 | 3.8秒 | 0.4秒 | 3.9秒 | 12分42秒 | 11分34秒 |
| TIME CPHL/H | 0.4秒 | 6.4秒 | 0.5秒 | 6.7秒 | 21分58秒 | 20分46秒 |
| TIME ECDC_COVID/D | 0.5秒 | 3.7秒 | 0.5秒 | 4.1秒 | 11分56秒 | 10分40秒 |
| TIME epf_electricity_price/H | 2.3秒 | 约1.4分钟 | 2.2秒 | 约1.4分钟 | 不执行 | 不执行 |
| TIME Commodity_Import/M | 0.5秒 | 3.0秒 | 0.4秒 | 3.0秒 | 不执行 | 不执行 |
| TIME Commodity_Production/M | 0.4秒 | 2.5秒 | 0.4秒 | 2.8秒 | 不执行 | 不执行 |

选择耗时取自每个选择命令的运行摘要；包含本任务特征准备/读取、评分和 RR 选样，不含训练评估。TIME epf_electricity_price/H 的日志摘要以 0.1 分钟精度显示。此前记录的 53–59 分钟并非选样运行时间，而是误把队列事件间隔累计成了耗时；以上逐任务日志时间为准，旧累计值作废。选择文件为各任务的 `window_indices.npy`，dtype=int64，共 266,631 个起点，大小 2,133,176 bytes（2.03 MiB）；traj/block arm 各 8 项合计 17,065,408 bytes（16.27 MiB），Uniform win 与 CRB win 各 5 项合计 10,665,880 bytes（10.17 MiB）。临时余弦评分矩阵在 RR 选择完成后清理，最终选择索引和 manifest 保留。两个基线不执行 RR 选样：full 直接使用全部合法窗口，rand01 的随机索引由一次性无放回抽样生成，无可比的选择耗时，故不列入本表。

## win 批次状态

用户于 2026-11-13 15:38 CST 确认 win 阶段只保留 Uniform 已完成的 FEV 9、TIME 1–4 五项，并执行相同五项的 CRB win。Uniform win TIME 6–8 与 CRB win TIME 6–8 均标记为不执行。TIME 6 因 259,085 个 ref 窗口对应约 1.26 TiB 完整 BF16 矩阵，超过当时约 381 GiB 可用空间而跳过；TIME 7、8 的矩阵估算仅约 7.83 GiB、0.30 GiB，按用户限定的五任务范围跳过，容量本身不是原因。截至 2026-11-13 20:21 CST，CRB win 的 FEV 9、TIME 1–4 五项选样、训练和评估均完成，五个任务及批次 `.complete` 标记均存在。TIME 4 评估于 17:29 CST 完成；五项 MASE 和全部评测指标列于上表。

完整 BF16 矩阵体积按 N_ref × 2,666,312 × 2 bytes 估算；同一任务的 Uniform/CRB 窗口数相同，矩阵大小相同。TIME 6 的 ref 窗口数为 259,085，矩阵为 1,381,602,889,040 bytes（约 1.26 TiB），容量预检在评分开始前失败；当时数据盘剩余约 381 GiB，因此跳过。TIME 7、8 分别有 1,577、61 个 ref 窗口，矩阵估算为 8,409,548,048 bytes（7.83 GiB）、325,290,064 bytes（0.30 GiB）；两项并非容量超限，而是按用户确认的 CRB win 仅跑 Uniform 已完成五项这一范围跳过。已完成选择矩阵均通过索引预算和唯一性核验，并在 manifest 写入后清理。

| Uniform win 任务 | ref 窗口数 | BF16 矩阵体积（已生成或估算） | 选择耗时 | 选择索引文件 | 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| FEV hospital_admissions_1D | 4,766 | 25,415,285,984 bytes（23.67 GiB） | 15分05秒 | 2,133,176 bytes（2.03 MiB） | 训练评估完成 |
| TIME CPHL/15T | 3,250 | 17,331,028,000 bytes（16.14 GiB） | 9分09秒 | 2,133,176 bytes（2.03 MiB） | 训练评估完成 |
| TIME CPHL/30T | 4,227 | 22,541,001,648 bytes（20.99 GiB） | 12分42秒 | 2,133,176 bytes（2.03 MiB） | 训练评估完成 |
| TIME CPHL/H | 13,332 | 71,094,543,168 bytes（66.21 GiB） | 21分58秒 | 2,133,176 bytes（2.03 MiB） | 训练评估完成 |
| TIME ECDC_COVID/D | 4,131 | 22,029,069,744 bytes（20.52 GiB） | 11分56秒 | 2,133,176 bytes（2.03 MiB） | 训练评估完成 |
| TIME epf_electricity_price/H | 259,085 | 1,381,602,889,040 bytes（约 1.26 TiB） | — | — | Uniform/CRB win 均跳过：完整矩阵超过当前处理范围 |
| TIME Commodity_Import/M | 1,577 | 8,409,548,048 bytes（7.83 GiB） | — | — | Uniform/CRB win 均按用户限定范围跳过；容量可容纳 |
| TIME Commodity_Production/M | 61 | 325,290,064 bytes（0.30 GiB） | — | — | Uniform/CRB win 均按用户限定范围跳过；容量可容纳 |


### CRB win 已完成任务的评分矩阵与选择产物

| 任务 | ref 窗口数 | BF16 矩阵峰值 | 选择耗时 | 选择索引文件 | 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| FEV hospital_admissions_1D | 4,766 | 25,415,285,984 bytes（23.67 GiB） | 20分55秒 | 2,133,176 bytes（2.03 MiB） | 训练评估完成 |
| TIME CPHL/15T | 3,250 | 17,331,028,000 bytes（16.14 GiB） | 8分56秒 | 2,133,176 bytes（2.03 MiB） | 训练评估完成 |
| TIME CPHL/30T | 4,227 | 22,541,001,648 bytes（20.99 GiB） | 11分34秒 | 2,133,176 bytes（2.03 MiB） | 训练评估完成 |
| TIME CPHL/H | 13,332 | 71,094,543,168 bytes（66.21 GiB） | 20分46秒 | 2,133,176 bytes（2.03 MiB） | 训练评估完成 |
| TIME ECDC_COVID/D | 4,131 | 22,029,069,744 bytes（20.52 GiB） | 10分40秒 | 2,133,176 bytes（2.03 MiB） | 训练评估完成 |

CRB win 五项的 MASE 与全部评测指标已列于上表；Uniform win TIME 6–8 和 CRB win TIME 6–8 按用户确认不执行，不报告评测结果。

## 补算方向对齐指标（sgwdev）

批次：sgwdev（single-window-eval）；指标一完成 16/16 组，指标二完成 16/16 组，筛选后指标记录 42 个已完成选样批次任务组合。全源指标为已计算锚点均值估计，筛选后指标也限于现有源锚点。

统一梯度配对协议（用户确认）：全源与 top10% 筛选后指标只配对已缓存的真实源锚点梯度和已缓存的同方法安全历史锚点梯度；安全历史—测试指标只配对安全历史缓存锚点与按 H 新算的真实测试样本梯度。三项都不用重建逐窗口梯度；全源和筛选后结果仅代表对应的缓存锚点子集估计。对已选梯度单位化后的全部交叉组合求余弦算术均值，以双方单位梯度均值的点积等价累加，不实体化余弦矩阵。

### 全源数据集任务平均方向评分

源侧直接读取 src-96 已有缓存，参考侧读取对应任务已有安全历史缓存；CRB 与 Uniform 两侧方法匹配。对单位方向执行 P1 过滤后，报告缓存锚点间全部两两余弦的算术平均；不重算源梯度、不对未计算窗口插值。

| 下游任务 | 方法 | 参考锚点数 | 源锚点数 | 均值估计 cos |
| --- | --- | ---: | ---: | ---: |
| FEV 9 hospital_admissions_1D | CRB | 144 | 83330 | 0.0017224574 |
| FEV 9 hospital_admissions_1D | Uniform | 144 | 83330 | 0.0002706576 |
| TIME 1 CPHL/15T | CRB | 100 | 83330 | 0.0002957125 |
| TIME 1 CPHL/15T | Uniform | 100 | 83330 | 0.0008866591 |
| TIME 2 CPHL/30T | CRB | 132 | 83330 | 0.0037917030 |
| TIME 2 CPHL/30T | Uniform | 132 | 83330 | -0.0004814057 |
| TIME 3 CPHL/H | CRB | 418 | 83330 | 0.0003262907 |
| TIME 3 CPHL/H | Uniform | 418 | 83330 | -0.0004073269 |
| TIME 4 ECDC_COVID/D | CRB | 132 | 83330 | 0.0006496567 |
| TIME 4 ECDC_COVID/D | Uniform | 132 | 83330 | 0.0013227684 |
| TIME 6 epf_electricity_price/H | CRB | 8100 | 83330 | 0.0007575118 |
| TIME 6 epf_electricity_price/H | Uniform | 8100 | 83330 | 0.0004122440 |
| TIME 7 Commodity_Import/M | CRB | 50 | 83330 | -0.0076627379 |
| TIME 7 Commodity_Import/M | Uniform | 50 | 83330 | -0.0055498154 |
| TIME 8 Commodity_Production/M | CRB | 3 | 83330 | -0.0001750796 |
| TIME 8 Commodity_Production/M | Uniform | 3 | 83330 | -0.0009809971 |

### 安全历史对真实测试样本的梯度方向代表性

安全历史侧使用已有原生 480+96 梯度缓存；测试侧使用真实 single-window 输入与标签，按任务 H 计算样本梯度。TIME 评测标签中的 NaN 位置按评测器有效标签掩码计算；H 内全缺失的样本不产生梯度并排除。

| 下游任务 | 方法 | H | 安全历史缓存梯度数 | 真实测试样本数 | 平均方向 cos | 真实样本 task-H MSE | 全缺失标签排除数 | 不足 480 点上下文排除数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FEV 9 hospital_admissions_1D | CRB | 28 | 144 | 8 | 0.00820602 | 0.88611317 | 0 | 0 |
| FEV 9 hospital_admissions_1D | Uniform | 28 | 144 | 8 | 0.00611989 | 0.88611317 | 0 | 0 |
| TIME 1 CPHL/15T | CRB | 12 | 100 | 2 | 0.00831781 | 0.76387602 | 0 | 0 |
| TIME 1 CPHL/15T | Uniform | 12 | 100 | 2 | 0.00717548 | 0.76387602 | 0 | 0 |
| TIME 2 CPHL/30T | CRB | 12 | 132 | 2 | -0.00697021 | 0.26637596 | 0 | 0 |
| TIME 2 CPHL/30T | Uniform | 12 | 132 | 2 | -0.00826498 | 0.26637596 | 0 | 0 |
| TIME 3 CPHL/H | CRB | 24 | 418 | 4 | 0.00122327 | 0.55605108 | 0 | 0 |
| TIME 3 CPHL/H | Uniform | 24 | 418 | 4 | -0.00073248 | 0.55605108 | 0 | 0 |
| TIME 4 ECDC_COVID/D | CRB | 30 | 132 | 9 | 0.01712149 | 0.05724077 | 0 | 0 |
| TIME 4 ECDC_COVID/D | Uniform | 30 | 132 | 9 | 0.02130944 | 0.05724077 | 0 | 0 |
| TIME 6 epf_electricity_price/H | CRB | 24 | 8100 | 5 | 0.00445988 | 0.50372446 | 0 | 0 |
| TIME 6 epf_electricity_price/H | Uniform | 24 | 8100 | 5 | 0.00619175 | 0.50372446 | 0 | 0 |
| TIME 7 Commodity_Import/M | CRB | 12 | 50 | 5 | -0.00631311 | 2.66750547 | 0 | 3 |
| TIME 7 Commodity_Import/M | Uniform | 12 | 50 | 5 | -0.00934393 | 2.66750550 | 0 | 3 |
| TIME 8 Commodity_Production/M | CRB | 12 | 3 | 3 | -0.04071205 | 0.14887695 | 0 | 29 |
| TIME 8 Commodity_Production/M | Uniform | 12 | 3 | 3 | -0.04071205 | 0.14887695 | 0 | 29 |

### 筛选后（top10%）源侧方向接近度

对每个已完成的 top10% 选样批次，从现有 src-96 源梯度缓存中取出全局窗口索引落在选中集合的锚点，与同方法安全历史锚点按全源指标相同 P1 规则求平均两两 cos。源侧不重算梯度；该指标是选中 top10% 中缓存真实锚点的对齐估计，同时报告其对全部源锚点的覆盖率。

| 任务 | 选样方法 | 选中源窗口 | 选中源锚点 | 源锚点覆盖率 | 参考锚点 | 平均方向 cos |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FEV 9 hospital_admissions_1D | Uniform traj | 266631 | 8333 | 10.0000% | 144 | 0.00435617 |
| TIME 1 CPHL/15T | Uniform traj | 266631 | 8330 | 9.9964% | 100 | 0.00543322 |
| TIME 2 CPHL/30T | Uniform traj | 266631 | 8333 | 10.0000% | 132 | 0.00256785 |
| TIME 3 CPHL/H | Uniform traj | 266631 | 8334 | 10.0012% | 418 | 0.00072261 |
| TIME 4 ECDC_COVID/D | Uniform traj | 266631 | 8334 | 10.0012% | 132 | 0.01284328 |
| TIME 6 epf_electricity_price/H | Uniform traj | 266631 | 8332 | 9.9988% | 8100 | 0.00239647 |
| TIME 7 Commodity_Import/M | Uniform traj | 266631 | 8334 | 10.0012% | 50 | 0.00094907 |
| TIME 8 Commodity_Production/M | Uniform traj | 266631 | 8334 | 10.0012% | 3 | 0.00831541 |
| FEV 9 hospital_admissions_1D | Uniform block | 266631 | 8333 | 10.0000% | 144 | 0.00544601 |
| TIME 1 CPHL/15T | Uniform block | 266631 | 8328 | 9.9940% | 100 | 0.00966046 |
| TIME 2 CPHL/30T | Uniform block | 266631 | 8328 | 9.9940% | 132 | 0.00859986 |
| TIME 3 CPHL/H | Uniform block | 266631 | 8331 | 9.9976% | 418 | 0.00085519 |
| TIME 4 ECDC_COVID/D | Uniform block | 266631 | 8333 | 10.0000% | 132 | 0.03872909 |
| TIME 6 epf_electricity_price/H | Uniform block | 266631 | 8336 | 10.0036% | 8100 | 0.00232875 |
| TIME 7 Commodity_Import/M | Uniform block | 266631 | 8332 | 9.9988% | 50 | 0.02036261 |
| TIME 8 Commodity_Production/M | Uniform block | 266631 | 8342 | 10.0108% | 3 | 0.03821157 |
| FEV 9 hospital_admissions_1D | CRB traj | 266631 | 7902 | 9.4828% | 144 | 0.00745219 |
| TIME 1 CPHL/15T | CRB traj | 266631 | 8336 | 10.0036% | 100 | 0.00878810 |
| TIME 2 CPHL/30T | CRB traj | 266631 | 8106 | 9.7276% | 132 | 0.01384357 |
| TIME 3 CPHL/H | CRB traj | 266631 | 7888 | 9.4660% | 418 | 0.00340801 |
| TIME 4 ECDC_COVID/D | CRB traj | 266631 | 8291 | 9.9496% | 132 | 0.01261977 |
| TIME 6 epf_electricity_price/H | CRB traj | 266631 | 8328 | 9.9940% | 8100 | 0.00317352 |
| TIME 7 Commodity_Import/M | CRB traj | 266631 | 8303 | 9.9640% | 50 | -0.00254030 |
| TIME 8 Commodity_Production/M | CRB traj | 266631 | 8318 | 9.9820% | 3 | 0.01904252 |
| FEV 9 hospital_admissions_1D | CRB block | 266631 | 8803 | 10.5640% | 144 | 0.00747983 |
| TIME 1 CPHL/15T | CRB block | 266631 | 7654 | 9.1852% | 100 | 0.01066268 |
| TIME 2 CPHL/30T | CRB block | 266631 | 7716 | 9.2596% | 132 | 0.01191690 |
| TIME 3 CPHL/H | CRB block | 266631 | 7312 | 8.7748% | 418 | 0.00225722 |
| TIME 4 ECDC_COVID/D | CRB block | 266631 | 6945 | 8.3343% | 132 | 0.02309183 |
| TIME 6 epf_electricity_price/H | CRB block | 266631 | 7550 | 9.0604% | 8100 | 0.00250347 |
| TIME 7 Commodity_Import/M | CRB block | 266631 | 7895 | 9.4744% | 50 | 0.00752373 |
| TIME 8 Commodity_Production/M | CRB block | 266631 | 8375 | 10.0504% | 3 | 0.04166267 |
| FEV 9 hospital_admissions_1D | Uniform win | 266631 | 8158 | 9.7900% | 144 | 0.00430159 |
| TIME 1 CPHL/15T | Uniform win | 266631 | 7762 | 9.3148% | 100 | 0.01030771 |
| TIME 2 CPHL/30T | Uniform win | 266631 | 7564 | 9.0772% | 132 | 0.00858809 |
| TIME 3 CPHL/H | Uniform win | 266631 | 7384 | 8.8612% | 418 | 0.00076094 |
| TIME 4 ECDC_COVID/D | Uniform win | 266631 | 7553 | 9.0640% | 132 | 0.03409148 |
| FEV 9 hospital_admissions_1D | CRB win | 266631 | 9180 | 11.0164% | 144 | 0.00495704 |
| TIME 1 CPHL/15T | CRB win | 266631 | 7501 | 9.0016% | 100 | 0.00913078 |
| TIME 2 CPHL/30T | CRB win | 266631 | 7917 | 9.5008% | 132 | 0.01508618 |
| TIME 3 CPHL/H | CRB win | 266631 | 7518 | 9.0220% | 418 | 0.00261109 |
| TIME 4 ECDC_COVID/D | CRB win | 266631 | 7069 | 8.4831% | 132 | 0.02375410 |

指标二损失口径：仅真实测试样本侧按任务 H 使用 480 点历史和 H 点真实目标计算 MSE 梯度；使用 TimerXL 96 点输出头的前 H 个预测，归一化仅依赖历史。安全历史侧不重算，直接使用同一 step-1000 检查点的既有梯度缓存。

### 指标二的余弦分数分布

按任务和方法，对所有安全历史锚点—真实测试样本两两余弦报告 P50、P90、P95、P99、最大值，并报告 cos>0 与 cos>=0.1 的配对比例。另外按每个真实测试样本取安全历史锚点中的最大 cos，报告测试样本最佳匹配值的平均和最大值。数值来自已保存的梯度，没有重算梯度。

两两配对统计的分母为安全历史锚点数乘以真实测试样本数（本批次每任务 100–40,500 对）。同一锚点会参与多个配对，因此这些分位数是描述性分布统计，不视为独立样本置信区间。每项任务仅有 2–9 个真实测试样本，因此测试样本最佳匹配值的高分位数不稳定，报告其均值和最大值。此后新增相似度分析必须包含分布统计；大规模配对需在实验文档中声明是全量精确分位数还是按固定规则抽样估计，并记录数量、随机种子和实际算法。

| Task | Method | Pair count | P50 | P90 | P95 | P99 | Max | Positive | cos >= 0.1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FEV 9 hospital_admissions_1D | CRB | 1152 | 0.010834 | 0.198855 | 0.257365 | 0.388612 | 0.569305 | 52.5% | 27.3% |
| FEV 9 hospital_admissions_1D | Uniform | 1152 | 0.009395 | 0.201206 | 0.254458 | 0.348050 | 0.619248 | 53.0% | 26.5% |
| TIME 1 CPHL/15T | CRB | 200 | 0.003821 | 0.142561 | 0.173722 | 0.231969 | 0.328059 | 53.0% | 14.5% |
| TIME 1 CPHL/15T | Uniform | 200 | 0.003930 | 0.117533 | 0.173413 | 0.208368 | 0.308987 | 50.5% | 12.5% |
| TIME 2 CPHL/30T | CRB | 264 | -0.004901 | 0.148476 | 0.220182 | 0.387366 | 0.456489 | 48.9% | 17.0% |
| TIME 2 CPHL/30T | Uniform | 264 | -0.009434 | 0.156222 | 0.216673 | 0.385690 | 0.456489 | 47.0% | 15.2% |
| TIME 3 CPHL/H | CRB | 1672 | 0.002196 | 0.156139 | 0.201210 | 0.288075 | 0.536552 | 50.8% | 18.8% |
| TIME 3 CPHL/H | Uniform | 1672 | -0.000749 | 0.157023 | 0.198629 | 0.292177 | 0.513484 | 49.8% | 18.9% |
| TIME 4 ECDC_COVID/D | CRB | 1188 | 0.018231 | 0.175268 | 0.226560 | 0.337262 | 0.509316 | 57.1% | 23.7% |
| TIME 4 ECDC_COVID/D | Uniform | 1188 | 0.027959 | 0.182204 | 0.244717 | 0.350152 | 0.482916 | 59.1% | 26.5% |
| TIME 6 epf_electricity_price/H | CRB | 40500 | 0.003769 | 0.087701 | 0.114698 | 0.177353 | 0.334714 | 52.5% | 7.4% |
| TIME 6 epf_electricity_price/H | Uniform | 40500 | 0.005470 | 0.089930 | 0.117976 | 0.181709 | 0.339312 | 53.6% | 7.8% |
| TIME 7 Commodity_Import/M | CRB | 250 | -0.011486 | 0.177083 | 0.234808 | 0.325868 | 0.398489 | 45.6% | 20.8% |
| TIME 7 Commodity_Import/M | Uniform | 250 | -0.009287 | 0.196155 | 0.247461 | 0.335832 | 0.378646 | 46.4% | 23.2% |
| TIME 8 Commodity_Production/M | CRB | 9 | -0.021656 | 0.072106 | 0.085432 | 0.096093 | 0.098758 | 33.3% | 0.0% |
| TIME 8 Commodity_Production/M | Uniform | 9 | -0.021656 | 0.072106 | 0.085432 | 0.096093 | 0.098758 | 33.3% | 0.0% |

### 梯度对齐与下游收益的任务级关系

以每个任务四个完整 RR arm 的 MASE 算术平均作为选样结果；以 full 与 rand01 中较低的 MASE 作为该任务较强基线。相对收益定义为 `(较强基线 MASE - 四 arm 平均 MASE) / 较强基线 MASE`，正数表示 RR 平均胜过两基线中更强者。另统计四 arm 中逐一同时胜过 full 和 rand01 的 arm 数。任务间对齐量分别为 CRB/Uniform 全源均值、四个 RR arm 的 top10% 对齐均值，以及 CRB/Uniform 安全历史—测试平均 cos 与两两余弦 P99 的均值。

| 任务 | 全源方向 cos 均值 | top10% cos 均值 | 安全历史—测试 cos 均值 | 同时胜过两基线的 arm | RR 平均相对较强基线收益 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FEV 9 hospital_admissions_1D | +0.000997 | 0.006184 | +0.007163 | 2/4 | -0.10% |
| TIME 1 CPHL/15T | +0.000591 | 0.008636 | +0.007747 | 0/4 | -12.09% |
| TIME 2 CPHL/30T | +0.001655 | 0.009232 | -0.007618 | 0/4 | -37.18% |
| TIME 3 CPHL/H | -0.000041 | 0.001811 | +0.000245 | 3/4 | +4.47% |
| TIME 4 ECDC_COVID/D | +0.000986 | 0.021821 | +0.019215 | 1/4 | -8.78% |
| TIME 6 epf_electricity_price/H | +0.000585 | 0.002601 | +0.005326 | 4/4 | +11.95% |
| TIME 7 Commodity_Import/M | -0.006606 | 0.006574 | -0.007829 | 0/4 | -5.78% |
| TIME 8 Commodity_Production/M | -0.000578 | 0.026808 | -0.040712 | 4/4 | +2.93% |

四 arm 平均 MASE 在 8 项任务中有 3 项胜过该任务的较强基线；单个 arm 有 14/32 项同时胜过 full 与 rand01。按任务等权计算的相对收益均值为 -5.57%。任务级结果只有 8 个观测，且每种训练配置为单 seed。以四 arm 平均相对较强基线收益为结果变量（正值代表 RR 更好），下表报告该收益与不同任务级对齐量的 Pearson 和 Spearman 相关；全源及 top10% 为上表已用的任务均值，安全历史—测试 P99 为两种方法配对分布 P99 的任务均值。相关系数只作描述性观察，不支持因果结论或稳定性推断。

| 任务级对齐量 | Pearson r | Spearman rho |
| --- | ---: | ---: |
| 全源平均 cos | -0.173 | -0.524 |
| top10% 平均 cos | -0.099 | -0.548 |
| 安全历史—真实测试平均 cos | -0.043 | -0.214 |
| 安全历史—真实测试配对 P99 | -0.554 | -0.595 |

对本批次的直接问题：RR 四 arm 平均 MASE 仅在 3/8 任务胜过 full 和 rand01 两个基线；任务级平均相对较强基线收益为 -5.57%。P99 与收益的排序相关在本批次呈负（rho=-0.595），不支持“历史—测试尾部对齐越高，RR 平均越容易胜基线”。安全历史—测试均值与收益的关系非常弱（rho=-0.214）。但 n=8 下这不是可推广的负向效应，也可能受少数任务与单 seed 影响。

逐项看，TIME 1、TIME 2、TIME 7 的四个 arm 都未同时胜过两条基线；TIME 6 和 TIME 8 的四个 arm 都胜过两条基线。TIME 6 在 top10% 对齐均值仅 0.002601 时仍有 4/4 胜出；TIME 4 的 top10% 对齐均值最高（0.021821），但 RR 平均仍比强基线差 8.78%。因此当前结果不支持“对齐越高，越容易胜过基线”的简单关系。低对齐与失败在部分任务上同时出现，但不能据此断定低对齐是失败原因。


### src P99 高匹配单元的时间与轨迹聚集

每个 src 单元的分数定义为该单元方向与该任务所有参考单元方向之间的最大 cos（best-reference match）。在每任务、方法和粒度内按该分数降序取正好 top 1% 单元；这量度是“至少与某个参考方向匹配”，不是 RR 选择中每个参考单元独立排序的分数。完全复用已缓存 source_units.npz 和 reference_units.npz 方向，未重算梯度。traj 粒度每方法和任务只有 96 条源轨迹单元，top 1% 因此是 1 条，可报最匹配轨迹，但不能检查轨迹内时间聚集。block 粒度用每个已选高分 block 所在轨迹、同轨迹上相邻高分 block 对数及最长连续运行长度量化聚集。

时间聚集的参考分布是每条轨迹内保持 P99 block 个数不变，在该轨迹的 block 位置中均匀随机重选 500 次（seed=42）。报告随机相邻对数均值和上尾 p 值。此重排只保持各轨迹高分 block 数，不保持相邻块分数之间由特征构造带来的依赖。block 方向由相邻的缓存锚点方向插值构建，相邻 block 共用边界锚点；因此相邻超额可作为分数时间聚集的描述，但不解释为独立样本的显著性证据。

| Task | Method | Trajectories with P99 blocks | Effective trajectories | Top 10 trajectory share | Adjacent P99 block pairs / random mean | Longest run | Randomization p |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FEV 9 hospital_admissions_1D | CRB | 28/96 | 7.3 | 88.4% | 441 / 198.9 | 12 | 0.002 |
| TIME 1 CPHL/15T | CRB | 36/96 | 5.8 | 82.8% | 210 / 52.2 | 6 | 0.002 |
| TIME 2 CPHL/30T | CRB | 27/96 | 5.3 | 91.0% | 213 / 36.2 | 7 | 0.002 |
| TIME 3 CPHL/H | CRB | 26/96 | 9.3 | 89.3% | 221 / 75.8 | 8 | 0.002 |
| TIME 4 ECDC_COVID/D | CRB | 23/96 | 7.0 | 95.0% | 178 / 33.5 | 6 | 0.002 |
| TIME 6 epf_electricity_price/H | CRB | 24/96 | 5.4 | 94.0% | 338 / 133.7 | 10 | 0.002 |
| TIME 7 Commodity_Import/M | CRB | 53/96 | 20.4 | 60.7% | 295 / 80.6 | 12 | 0.002 |
| TIME 8 Commodity_Production/M | CRB | 45/96 | 11.3 | 76.0% | 243 / 49.1 | 7 | 0.002 |
| FEV 9 hospital_admissions_1D | UNIFORM | 27/96 | 6.8 | 85.7% | 455 / 244.9 | 21 | 0.002 |
| TIME 1 CPHL/15T | UNIFORM | 29/96 | 5.2 | 85.0% | 275 / 72.1 | 8 | 0.002 |
| TIME 2 CPHL/30T | UNIFORM | 32/96 | 8.3 | 78.8% | 298 / 79.0 | 8 | 0.002 |
| TIME 3 CPHL/H | UNIFORM | 24/96 | 7.0 | 93.6% | 332 / 213.2 | 9 | 0.002 |
| TIME 4 ECDC_COVID/D | UNIFORM | 18/96 | 4.6 | 98.9% | 251 / 46.3 | 5 | 0.002 |
| TIME 6 epf_electricity_price/H | UNIFORM | 26/96 | 7.8 | 93.3% | 571 / 77.8 | 39 | 0.002 |
| TIME 7 Commodity_Import/M | UNIFORM | 50/96 | 20.6 | 61.9% | 342 / 100.3 | 13 | 0.002 |
| TIME 8 Commodity_Production/M | UNIFORM | 33/96 | 4.9 | 86.7% | 227 / 35.3 | 5 | 0.002 |

traj 粒度每个任务与方法的最高分源轨迹：

| Task | Method | Top trajectory | Best-reference cos |
| --- | --- | --- | ---: |
| FEV 9 hospital_admissions_1D | CRB | chronos/weatherbench_daily/T086145 | 0.5093 |
| TIME 1 CPHL/15T | CRB | chronos/wiki_daily_100k/T068491 | 0.5682 |
| TIME 2 CPHL/30T | CRB | chronos/monash_london_smart_meters/T000242 | 0.5377 |
| TIME 3 CPHL/H | CRB | chronos/monash_weather/T000866 | 0.4597 |
| TIME 4 ECDC_COVID/D | CRB | chronos/wind_farms_hourly/T000329 | 0.4380 |
| TIME 6 epf_electricity_price/H | CRB | chronos/monash_electricity_hourly/T000057 | 0.5901 |
| TIME 7 Commodity_Import/M | CRB | chronos/monash_fred_md/T000105 | 0.4136 |
| TIME 8 Commodity_Production/M | CRB | chronos/monash_temperature_rain/10917__t_mean | 0.1866 |
| FEV 9 hospital_admissions_1D | UNIFORM | chronos/weatherbench_daily/T086145 | 0.4200 |
| TIME 1 CPHL/15T | UNIFORM | chronos/wiki_daily_100k/T068491 | 0.5810 |
| TIME 2 CPHL/30T | UNIFORM | monash/australian_electricity_demand/T4 | 0.4395 |
| TIME 3 CPHL/H | UNIFORM | monash/weather/T1750 | 0.4450 |
| TIME 4 ECDC_COVID/D | UNIFORM | monash/oikolab_weather/T4 | 0.5123 |
| TIME 6 epf_electricity_price/H | UNIFORM | chronos/monash_electricity_hourly/T000057 | 0.6601 |
| TIME 7 Commodity_Import/M | UNIFORM | chronos/m5/FOODS_3_562_CA_2 | 0.3616 |
| TIME 8 Commodity_Production/M | UNIFORM | monash/weather/T1092 | 0.1806 |

输入与产物：src-96 梯度缓存、各批次 window_indices.npy和安全历史缓存；结果清单、梯度草图与 manifest 位于 outputs/selection/v1-crb-redo/sgwdev/metrics/。
