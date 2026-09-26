# 2026-09-14 参数与共享任务探测

## 暂定拍板：三模型参数与十个共享任务

2026-09-14，用户明确批准按以下顺序定为10个任务，并暂定拍板三个模型的各自参数。
本节是当前决策，优先于下文初始8候选记录和旧程序自动产生的前三清单。
累计探测候选15个，从中选定10个，没有新增候选。三模型共用同一任务列表，未经用户再次决定不自行增删或重开调参。

机器可读记录：[shared_tasks_10.json](../../configs/rags/shared_tasks_10.json)。
执行checkout同步记录位于 `/root/TS3-probe/configs/rags/shared_tasks_10.json`。
此文件是范围/参数决策记录，不是可直接启动正式源池选样实验的配置；本次记录不启动新训练。

### 三模型各自参数

| 参数 | Chronos T5 Base | TimeMoE 200M | TimesFM 2.5 |
| --- | --- | --- | --- |
| 学习率 | 1e-3 | 1e-4 | 3e-5 |
| 停止点 E | 2 | 1 | 2.625 |
| 有效 batch / micro batch | 32 / 32 | 64 / 16 | 32 / 32 |
| 优化器 | AdamW | AdamW | AdamW |
| weight decay | 0 | 0.1（bias/norm不衰减） | 0.01 |
| betas | (0.9, 0.999) | (0.9, 0.95) | (0.9, 0.999) |
| 学习率调度 | linear | cosine | cosine |
| 最低学习率/初始学习率 | 0 | 0.5 | 0 |

共用设置：C512/H64，开发验证每序列8窗，seeds=[0,1,2]，评估batch32；eps=1e-8，
梯度裁剪max_grad_norm=1，warmup_ratio=0。原生损失、全参数训练；精度协议为
`bf16-model-gradient-adamw-v2`：参数/梯度/AdamW动量BF16，无FP32主权重；不等于所有中间运算都为BF16。

调度总长度始终固定 `schedule_epochs=6`，不能把上述停止点当成重新衰减到零的总长度。
这里 E 是有放回曝光预算：N=sum(floor(安全训练前缀长度/(512+64)))，更新步数=ceil(E*N/batch)。
不同任务相同E不等于相同更新步数，必须保留实际step/actual_epoch记录。

### 十个共享任务（用户批准顺序）

收益均为当前各模型固定配方下，安全历史训练相对基础模型的开发MASE下降百分比，3种子平均；
不是正式考试成绩，不是梯度选样收益，也不是等步数跨模型比较。

| 序号 | 任务ID | 范围 | TimeMoE | TimesFM | 选择说明 |
| --- | --- | --- | --- | --- | --- |
| 1 | time_nem_demand_5t | NEM五地区电力需求，5分钟 | +32.19% | +20.00% | 两模型三种子均改善 |
| 2 | time_azure_i_cpu_5t | Azure I类，492实体平均CPU，5分钟 | +17.92% | +4.34% | 两模型三种子均改善；保留带标记列的显式映射 |
| 3 | time_darwin_water_15t | 达尔文水质，7实体×6变量，15分钟 | +10.34% | +4.45% | 两模型三种子均改善 |
| 4 | time_finland_15t | 芬兰交通，原生单变量，15分钟 | +14.76% | +2.70% | 保留原生单变量任务，训练步数少 |
| 5 | time_eweld_15t | EWELD十用户负荷通道，15分钟 | +13.36% | +0.40% | TimesFM改善小，不能称强证据 |
| 6 | quito_min | Quito十分钟分辨率，773实体×5指标 | +23.62% | 未单任务补测 | 待TimesFM补测与指标审计 |
| 7 | time_cphl_h | 叶绿素，2条有效原生单变量，小时 | +4.65% | 未单任务补测 | 原生单变量，历史短；混合调参不等于单任务验证 |
| 8 | time_nem_price_5t | NEM五地区电价，5分钟 | +4.50% | 未单任务补测 | 与需求同源，不能视为独立数据来源 |
| 9 | time_solar_h | 澳洲光伏，1个有效通道，小时 | +2.81% | +0.54% | 弱改善任务 |
| 10 | time_port_activity_d | 港口进出口，99实体×2变量，日 | +0.35% | −0.41% | 非能源领域的弱收益/退化对照，不冒称两模型有效 |

“任务”是数据集/频率/变量组的集合，不是单实体单窗口。除明确标注者外，多变量拆通道或选单列不称原生单变量。
Quito hour未纳入：其基础分数被最大5条序列贡献约99.885%，种子方向分裂，指标待审计。
FEV候选可以使用，本轮已测，但未按本次批准顺序纳入这十项；不是今后禁止FEV。

### 暂定的含义与未完成事项

- 此处冻结的是下一阶段开发实验范围与参数，不宣称三个模型在十项上全部验证有效。
- Chronos尚未完成这十任务的逐项单任务复核；其原三任务混合调参最优均值+8.09%，有一个种子轻微退化。
- TimesFM尚待单任务复核Quito min、CPHL、NEM电价。相关任务已选入，不因未测就写成正收益。
- Quito min需指标审计；Azure TimeMoE基础分数最大5条约占28.3%，需检查改善是否集中于少数实体。
- 当前证据全部来自安全历史随机合法窗口训练：不做梯度选样，不从外部源池挑训练数据，不复用选样梯度代替训练反传。
- 外部源池随机/梯度选样对照和保留测试区间正式比较尚未开展，需要独立执行协议；不由本次拍板自动启动。

### 数据、版本与结果溯源

共同产物根：`/root/autodl-tmp/TS3/outputs/probe-20260914/`。

| 记录 | 相对产物根路径 |
| --- | --- |
| 三模型调参依据 | `probe-v2/parameters/<model>/result.json` |
| 首轮8任务TimeMoE结果 | `probe-v2/task_probe/result.json` |
| TimesFM四任务追加 | `timesfm-single-domain-v1/<task>/` |
| 新增7任务两模型结果 | `probe-expansion-v1/<model>/<task>/` |
| 初始8任务安全历史 | `probe-inputs-v2/` |
| Azure、水质、港口安全历史 | `probe-inputs-expansion-v1r2/` |

执行代码：初始探测7a952fd；FEV导出/扩展1f3f0a5；显式Azure列名映射修复8d3b957。
所有批次实际正常退出，扩展准备相关测试27 passed。初次扩展失败的输入目录/日志保留，不覆写。
旧 `probe-v2/shared_selection.json` 的NEM需求/Quito min/Quito hour自动前三清单仅供历史追溯，
当前人工批准的十任务以本节和 `shared_tasks_10.json` 为准；不伪改历史产物。

## 初始执行记录（历史）

用户已批准[8个候选范围](../dev/ProbeScopeV2.md)，总候选硬上限20。
搜索规则见[参数探测](../dev/ParameterTaskProbing.md)：E1/E2后最多4次追加，扩展与二分共用预算。

本轮运行快照为 `/root/TS3-probe` @ `7a952fd`，配置为该checkout下
`configs/rags/probing-v2.json`、`configs/rags/probe_inputs-v2.json`。
主仓库 `/root/TS3` 的文档提供索引，不表示探测源码已合并进主仓库当前分支。
进程显式设置 `PYTHONPATH=/root/TS3-probe/src`，三个模型按配置分别使用其环境。

2026-09-14 18:01，zcode完成8份输入manifest核对和plan后启动远端tmux `probe-v2-all`。
启动记录中的all进程为220161；此PID只作当时证据，恢复前重新核实进程和起始时间。

| 产物 | 位置 |
| --- | --- |
| 输入 | `/root/autodl-tmp/TS3/outputs/probe-20260914/probe-inputs-v2` |
| 新协议结果 | `/root/autodl-tmp/TS3/outputs/probe-20260914/probe-v2` |
| 日志 | `/root/autodl-tmp/TS3/outputs/probe-20260914/probe-v2-all.log` |
| zcode交接、实时STATUS、累计RESULTS、30分钟巡检 | WSL `/home/rogers/Projects/codex-zcode-handoff/probe-execution-20260914/` |

运行命令与权限以该交接目录的 `RELEASE_V2.md` 为准。主Codex开发、核查代码，zcode执行与巡检。
本页只记录入口与里程碑，实时数字写交接STATUS/RESULTS和实际run产物，避免多份进度副本。

旧 `probe-v1` 已停止并保留，不把其中单种子的Quito改善与v2的新范围/六轮衰减协议混合。
本次docs整理只修改主仓库文档，没有更改运行checkout或中断实验。
