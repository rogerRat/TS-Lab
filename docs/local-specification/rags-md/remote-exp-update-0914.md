# exp 远程更新：重采样方法、模型参数与十任务范围

本文根据 `ssh exp:/root/TS3` 在 2026-09-14 的代码、配置和执行文档整理。这里的主机职责约定是：
`wsl` 只是用户的跳板机和 Agent 工作区；`ssh exp` 才是代码、配置、实验运行和结果的事实来源。
远程仓库当前 `HEAD` 为 `abc19dc`；工作区还包含未提交的文档、配置和实验套件更新，
因此这里记录的是远程当前工作状态，而不是只记录最近一次提交。

## 1. 方法口径已经改为全池重采样

RAGS 的输出不再是一个固定大小的候选子集，而是共同可行源池上的训练概率分布。
窗口长度为 $512+64=576$，池大小从 manifest 读取：

$$
N=\sum_i\left\lfloor\frac{L_i}{576}\right\rfloor .
$$

逐点滑动的合法窗口总数另记为 $A$；$N$ 是训练量换算用的窗口当量，不是子集大小。
训练前完成评分和概率构造，训练中只按固定分布有放回抽样，不重新读取参考数据或重新评分。

各方法保留原有评分单位：Random 按窗口，LESS 和轨迹级方法按轨迹，LP-Mirror 按窗口，
LTSV 保留块到点再到窗口的链条，RAGS 按 CRB 参考块与源块匹配。全池重采样只改变评分
结果如何交付给训练，不把所有方法改造成逐窗口评分。

### RAGS 的参考配比约束

对保留参考块 $j$，令 $n_j$ 为合法起点数，

$$
\pi_j=\frac{n_j}{\sum_r n_r}.
$$

令 $m_k$ 为源块 $k$ 的合法起点数，$s_{jk}$ 为已有 CRB 方向余弦。对每个参考行分别
执行正对齐门控和归一化：

$$
P(k\mid j)=
\frac{m_k\exp(s_{jk}/\theta)\mathbf{1}[s_{jk}>0]}
{\sum_l m_l\exp(s_{jl}/\theta)\mathbf{1}[s_{jl}>0]}.
$$

随后只做参考份额加权和：

$$
\beta_k=\sum_j\pi_jP(k\mid j),
\qquad \sum_k\pi_jP(k\mid j)=\pi_j,
\qquad \sum_k\beta_k=1.
$$

因此参考数据原有配比是硬约束。匹配分数只决定某一参考份额分给哪些源块，不能因为
某个参考更容易匹配、源块更多或相似度更高而改变该参考的总份额。共享源块的不同参考
贡献相加；不能先把所有参考--源块配对混在一起归一化，也不能混合后再截断或重加权。
若某参考行没有正对齐源块，程序报告该行及其 $\pi_j$ 并拒绝生成正式概率产物。
源块内部再按合法起点均匀抽样，所以属于块 $k$ 的窗口概率为 $\beta_k/m_k$。

远程实现对应 `src/rags/experiments/resampling.py:rags_marginal` 和
`src/rags/experiments/full_pool.py:_rags`。当前协议版本为
`full-pool-resampling-0914-v1`，采样器通过 CDF 或 alias 表表示同一个概率分布；
两种表示都要通过行质量守恒、总和为 1 和保存/加载一致性检查。

远程 `docs/gradient_assets.md` 记录的最终 exp 回归为 366 passed、2 skipped、6 warnings；
参数/任务探测记录另外注明 v2 相关套件 38 passed、probing 套件 18 passed。它们验证的是
执行路径、配置协议和探测流程，不等于十个任务上的正式源池效果已经完成。

## 2. 三个模型的冻结训练参数

以下参数来自远程 `configs/rags/shared_tasks_10.json`，用于安全历史开发探测后的共享任务
训练。三种模型各自使用自己的配方，不把某一模型的学习率、batch 或 Epoch 迁移给其他模型。
共同设置为 $C=512,H=64$、三种子 `{0,1,2}`、评估 batch 32、eps $10^{-8}$、
梯度裁剪 1、warmup 0，并把学习率总调度长度固定为 6 Epoch。

| 模型 | 学习率 | 停止 Epoch | 有效 batch / micro batch | weight decay | betas | 调度 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Chronos T5 Base | $10^{-3}$ | 2 | 32 / 32 | 0 | (0.9, 0.999) | linear 到 0 |
| TimeMoE 200M | $10^{-4}$ | 1 | 64 / 16 | 0.1 | (0.9, 0.95) | cosine 到初始值的 0.5 |
| TimesFM 2.5 | $3\times10^{-5}$ | 2.625 | 32 / 32 | 0.01 | (0.9, 0.999) | cosine 到 0 |

TimeMoE 的 bias/norm 参数不做 weight decay；四个 micro batch 累积为一次更新。
精度协议为 `bf16-model-gradient-adamw-v2`：模型、梯度和 AdamW 状态使用 BF16，
不表示所有中间规约都使用 BF16。参数结果是安全历史开发探测的冻结依据，不是测试集调参
结果，也不是正式源池实验已经完成的证明。

## 3. 已确定的十个共享任务

十个任务按用户批准顺序固定，三种模型共享同一列表。任务是数据集/频率/变量组的集合，
不是单实体或单窗口。`shared_tasks_10.json` 明确标记 `formal_exam_used=false`，所以它是
下一阶段开发范围和参数部署记录，不应被写成正式考试成绩或已经完成的源池主实验。

| 序号 | 任务 ID | 范围 | 当前证据/备注 |
| ---: | --- | --- | --- |
| 1 | `time_nem_demand_5t` | NEM 五地区电力需求，5 分钟 | TimeMoE +32.19%，TimesFM +20.00% |
| 2 | `time_azure_i_cpu_5t` | Azure I 类 492 实体平均 CPU，5 分钟 | 两模型均有改善；需审计实体集中度 |
| 3 | `time_darwin_water_15t` | 达尔文水质，7 实体×6 变量，15 分钟 | TimeMoE +10.34%，TimesFM +4.45% |
| 4 | `time_finland_15t` | 芬兰交通，原生单变量，15 分钟 | TimeMoE +14.76%，TimesFM +2.70% |
| 5 | `time_eweld_15t` | EWELD 十用户负荷通道，15 分钟 | TimeMoE +13.36%，TimesFM +0.40% |
| 6 | `quito_min` | Quito 十分钟，773 实体×5 指标 | TimeMoE +23.62%；需指标审计，TimesFM 待补测 |
| 7 | `time_cphl_h` | 叶绿素，2 条有效原生单变量，小时 | TimeMoE +4.65%；TimesFM 待单任务复核 |
| 8 | `time_nem_price_5t` | NEM 五地区电价，5 分钟 | TimeMoE +4.50%；与需求同源，TimesFM 待复核 |
| 9 | `time_solar_h` | 澳洲光伏，1 个有效通道，小时 | TimeMoE +2.81%，TimesFM +0.54%，弱改善 |
| 10 | `time_port_activity_d` | 港口进出口，99 实体×2 变量，日 | TimeMoE +0.35%，TimesFM −0.41%，弱收益/退化对照 |

上表的百分比是安全历史训练相对未训练模型的开发 MASE 改善，按三种子平均；不是正式
考试分数、梯度选样收益，也不是跨模型等更新步数比较。Chronos 尚未完成十任务逐项单任务
复核；TimesFM 尚待 `quito_min`、`time_cphl_h` 和 `time_nem_price_5t` 的单任务复核；
`quito_min` 和 Azure 任务仍需指标审计。Quito hour 因指标被单一序列过度支配而未纳入。

## 4. 当前边界

- `shared_tasks_10.json` 是开发冻结记录，不是可直接启动正式源池实验的 formal config。
- 远程仍保留旧的正式 full-pool 六任务配置；它不能覆盖本节的十任务开发决定，也不能把
  十任务的安全历史结果冒充正式考试结果。正式源池主实验须另有明确的配置迁移和验证记录。
- 远程代码和文档的最新核心证据为 `docs/experiments/probe_execution_20260914.md`、
  `docs/dev/ParameterTaskProbing.md`、`src/rags/experiments/resampling.py`、
  `src/rags/experiments/full_pool.py` 及 `configs/rags/shared_tasks_10.json`。
- 原始自动生成的 `shared_selection.json` 和历史探测产物保留作溯源，不覆盖人工批准的十任务顺序。
