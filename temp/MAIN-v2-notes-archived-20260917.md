<!-- MAIN-v2 analysis notes: retained across supervisor refresh -->
## ETA 与 CRB 预算减半估算

估算快照：2026-09-16 23:56（北京时间）。当时完成 3/513 cells，三模型各一个 original；TimesFM 已进入选集，其余两模型仍在首任务资产阶段。以下为该时点的剩余时间，非滚动倒计时。

| 模型 | 剩余训练评测 | 补 Ref 梯度 | 其余资产与选集 | 剩余合计（中心） | 粗略范围 | 中心完工时间（北京时间） |
|---|---:|---:|---:|---:|---:|---|
| chronos-base | 14.42 h | 2.58 h | 1.40 h | 18.40 h | 13.99–26.23 h | 2026-09-17 18:21 |
| timesfm-2.5 | 8.81 h | 1.11 h | 1.03 h | 10.95 h | 9.38–28.69 h | 2026-09-17 10:53 |
| timemoe-200m | 16.34 h | 2.47 h | 1.63 h | 20.44 h | 12.09–24.85 h | 2026-09-17 20:23 |
| **整批（三模型并行，取最大值）** | — | — | — | **20.44 h** | **13.99–28.69 h** | **2026-09-17 20:23** |

整批粗略完工窗口为 **9 月 17 日 13:56 至 9 月 18 日 04:38**。范围是工程情景外推，不是统计置信区间；未计故障重试、停机或额外作业。共享 GPU 竞争变化可能改变吞吐。

估算依据：

- 每模型剩余 162 个训练 cells；同任务集、同 epochs 的历史训练评测记录分别 44/106/43 个，单 cell 中位数 320.48/195.72/363.18 秒，范围采用 p10/p90。历史并发及在线评测负载有变化，因此范围较宽，尤其 TimesFM。
- 梯度计时仅匹配当前 eager、batch 4/4/8、worker 1/1/2 的执行字段，新增窗口平均耗时 0.07189/0.04453/0.06861 秒。TimesFM 首任务补 82,778 个窗口，资产实测 3,814 秒，其中梯度 3,686 秒。
- 后续源池 CRB 与首任务配置相同，复用源池全部已算锚点；没有把首任务约一小时的补算重复乘九。每任务仍计暖源池 CRB/Ref 处理开销。
- Ref 的 LESS 均匀位置与 CRB coverage 位置可确定，逐 `(dataset, fragment_id, physical_start)` 精确计数缓存缺口。未知 error/split 位置以其预算上限约束；中心值取已完成 v1 Ref 的 split 上限 90%，其余取 error+split 上限 90%。这部分是显式启发式，未运行未来任务来冒充实测。
- 六方法选集按首任务实测/保守预估外推，Ref 相关成本按窗口数平方根缩放；训练固定预算、选集缓存复用与原始评测计入估算，未计 LTSV/Quito（已移出矩阵）。

机器可读明细：exp `temp/crb-half-20260916/eta.json`；复现入口 `temp/eta_v2.py`。后续取得完整 v2 训练 cells 和 Azure Ref 实测后，应重新计算 ETA。

### 预算减半：CPU 配对重放

仅估算，**未修改正式 CRB 配置**。当前 2%/1%/1% 与减半 1%/0.5%/0.5% 配对比较；沿用真实 v2 `closest_zero_v2`、轨迹 p10 审计、全部块保留。逐窗真值来自已有独立窗口梯度的 4096 维 BF16 投影缓存，CPU 只读、0 模型调用、未使用考试集。

样本为历史 TimeMoE Ref 五任务的全部已完成 13 条轨迹，共 260,041 个合法窗口；实体 ID、长度、窗口起点及缓存行逐字段校验。Finland 只有一条轨迹。此样本不代表 Chronos/TimesFM，也不代表九任务正式源池选集。

| 任务 | 轨迹数 | 窗口数 | 当前逐窗重建余弦 | 减半后 | 绝对下降 |
|---|---:|---:|---:|---:|---:|
| NEM demand | 3 | 116,643 | 0.8920 | 0.8147 | 0.0773 |
| Azure CPU | 3 | 21,578 | 0.8689 | 0.8102 | 0.0587 |
| Darwin water | 3 | 34,360 | 0.8390 | 0.7707 | 0.0683 |
| Finland | 1 | 31,585 | 0.8297 | 0.7288 | 0.1009 |
| EWELD | 3 | 55,875 | 0.8651 | 0.7866 | 0.0785 |
| **按窗口加权合计** | **13** | **260,041** | **0.8697** | **0.7920** | **0.0777** |

主指标是原始锚点向量在每个合法物理起点的分段线性重建与真值的余弦均值；负值保留、零范数按 0 计，不用块常值替代此重建指标。**粗略判断：减半后梯度重建精度约 0.79，任务范围约 0.73–0.81，较当前下降约 0.06–0.10。** 这些是样本结果与外推依据，不是“预测准确率 79%”。

| 辅助指标 | 当前 | 减半 |
|---|---:|---:|
| 实际独立锚点数 | 10,401（4.000%） | 5,198（1.999%） |
| 逐窗余弦 ≥0.9 的窗口比例 | 50.58% | 27.66% |
| 平均逐窗相对 L2 误差（零真值不计） | 0.4744 | 0.6008 |
| 生产块代表向量对逐窗真值余弦均值 | 0.8412 | 0.7522 |

逐窗重建余弦相对下降约 8.93%，相对 L2 误差增加约 26.65%。分裂预算在本次实际耗尽，不能引用旧协议“split 未触发”的结果替代。

**下游 MASE 减半后的数值仍无法由这项 CPU 模拟可靠估出。** CRB 重建误差会影响块评分、采样概率与训练轨迹，三者不是线性关系；目前没有同配方的完整/减半训练对照，也尚无当前 v2 RAGE 完整训练结果可校准。不能把 8.93% 的梯度余弦降幅等同于 MASE 增加 8.93%。本次支持的结论是梯度近似有明显损失；最终预测精度的变化需要配对训练评测。

精度机器可读产物：exp `temp/crb-half-20260916/{protocol,rows,summary}.json`；复现：

```bash
cd /root/TS3
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
 PYTHONPATH=src /root/miniconda3/envs/ts3/bin/python -u temp/half_crb.py
```

所有重放与分析临时文件在 `temp/`；只保存小型 JSON，不复制梯度、训练产物或模型。文档附注由 `MAIN-v2-notes.md` 保存，新启动的 supervisor 会合并附注；本轮已有 supervisor 的内存代码由轻量文档维护进程补入附注，随 PID403347 退出，不重启正式 worker。


## 事故记录：azure 选集阶段 CUDA 非法访存（2026-09-17 02:07–03:20）

- 现象：三 worker 在进入 azure_i_cpu 选集阶段后接连"无声"死亡；supervisor 三次重试耗尽后退出（attempt 3, exit 1, 剩 453 cells）。
- 根因链：azure 选集（rage 参考侧 CRB probe）触发异步 CUDA illegal memory access（exit 1，爆点 selection_metrics.py:54 的 synchronize）→ 容器 GPU 走 NVIDIA MPS，非法访存打挂 MPS server → 其余 worker 以 Error 807 连锁死亡 → GPU 清空。
- 证据：CUDA_LAUNCH_BLOCKING=1 下同一路径稳定运行 15+ 分钟（此前 2–3 分钟内必崩）→ 判定为竞态型非法访存，非越界（越界在 blocking 下也会崩）。nem 任务选集全程无恙；azure 是首个大参考集任务（492 refs / 3.8M 点 / 池 334,945 blocks）。
- 头号嫌疑：commit 82761ee 新增的 CRBEngineMulti 跨片段批式 probe 协程管线（v2 新代码，v1 azure 走旧引擎无此问题）。
- 已落盘数据安全：60 cells（nem 三模型全臂 + azure 三模型 original）原子保存，无损失。
- 恢复措施：重启 MPS 守护进程并验证 GPU 正常；timesfm worker 以 CUDA_LAUNCH_BLOCKING=1 单独重跑 azure 选集（选集产物将缓存）；完成后同法处理 chronos/timemoe 的 azure 选集，再重启 supervisor 续跑（训练/评测阶段从未触发该错误）。
- 待办：在代码级定位并修复该竞态（对照 v1 旧引擎 diff）；修复验证前，azure 及后续大任务选集建议保持 blocking 模式或串行执行。

## 事故记录 2：chronos×azure×LESS 数值下溢（2026-09-17 05:49）

- 现象：chronos blocking worker 在 azure 选集第 3 臂（lessvg_t025）崩溃：ValueError: Temperature underflows value mass（resampling.py value_mass）。timesfm/timemoe 同格均正常；v1 从未运行过 chronos×azure×LESS，属未覆盖路径。
- 根因：value_mass 已做 max 平移，但要求所有参与单元质量严格非零。azure 参考近噪声（v1 异质性：ρ≈0.197），chronos 的 LESS sum-of-cosines 分数范围极大，部分轨迹相对 max 差距超过 ~180/T，exp 精确下溢为 0——数学上即 softmax 概率精确为 0，不影响采样分布。
- 处置：放宽该检查（部分单元零质量合法；全部退化在 max 平移下不可能）。**数值语义零变更**：任何原本能通过的运行，输出逐位不变；仅把"崩溃"改为"按数学极限继续"。备份 temp/resampling.py.bak-*；该变更未 git commit（provenance 冻结于启动时 commit），待用户裁定后补提交。
- 另：chronos 曾于 03:52 启动瞬间死于 MPS 瞬时 807 拒绝（闲置 1.5h 后 05:20 由巡检发现并重启）。

<!-- v2-patrol:start -->
## 巡检记录与解读（每 30 分钟自动更新）

- 巡检时间：2026-09-17T09:19:29+08:00（北京时间）；GPU util 100%、451.48/600.00 W、显存 93866/97887 MiB、59°C；workers 6/3
- 总进度：196/513 cells；⚠️ 进程异常

### 结果（按任务分节，每节三模型各一表；两列百分比分别为均值相对 Original 与 Random 的改善，正=更好；胜/n 为逐 seed 配对胜 random）

### time_nem_demand_5t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs Original | vs Random | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| lessvg_t025（LESS-main 轨迹主实现） | 3 | 1.169068 | 0.033417 | +32.33% | +15.43% | 3/3 |
| random（Random 槽格均匀基线） | 3 | 1.382337 | 0.103409 | +19.98% | +0.00% | — |
| tsds（TSDS 原生 OT） | 3 | 1.382384 | 0.084658 | +19.98% | -0.00% | 1/3 |
| rage_traj_gls（RAGE-Traj 轨迹消融臂） | 3 | 1.458938 | 0.050065 | +15.55% | -5.54% | 1/3 |
| rage_gateless（RAGE-main CRB 块主实现） | 3 | 1.502679 | 0.041795 | +13.01% | -8.71% | 0/3 |
| lpv_t025（LP-Mirror-main 窗口主实现） | 3 | 1.512232 | 0.067067 | +12.46% | -9.40% | 0/3 |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs Original | vs Random | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| lessvg_t025（LESS-main 轨迹主实现） | 3 | 0.670757 | 0.021367 | +23.94% | +9.51% | 3/3 |
| rage_traj_gls（RAGE-Traj 轨迹消融臂） | 3 | 0.715856 | 0.026068 | +18.83% | +3.42% | 2/3 |
| tsds（TSDS 原生 OT） | 3 | 0.721019 | 0.001882 | +18.24% | +2.73% | 3/3 |
| lpv_t025（LP-Mirror-main 窗口主实现） | 3 | 0.734408 | 0.002324 | +16.73% | +0.92% | 2/3 |
| rage_gateless（RAGE-main CRB 块主实现） | 3 | 0.736918 | 0.026703 | +16.44% | +0.58% | 2/3 |
| random（Random 槽格均匀基线） | 3 | 0.741243 | 0.016305 | +15.95% | +0.00% | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs Original | vs Random | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| random（Random 槽格均匀基线） | 3 | 1.219479 | 0.024238 | -14.44% | +0.00% | — |
| rage_traj_gls（RAGE-Traj 轨迹消融臂） | 3 | 1.229896 | 0.007915 | -15.42% | -0.85% | 1/3 |
| rage_gateless（RAGE-main CRB 块主实现） | 3 | 1.236595 | 0.024083 | -16.05% | -1.40% | 1/3 |
| lpv_t025（LP-Mirror-main 窗口主实现） | 3 | 1.251016 | 0.014249 | -17.40% | -2.59% | 1/3 |
| lessvg_t025（LESS-main 轨迹主实现） | 3 | 1.304216 | 0.033460 | -22.39% | -6.95% | 0/3 |
| tsds（TSDS 原生 OT） | 3 | 1.354949 | 0.033407 | -27.15% | -11.11% | 0/3 |

**解读**：

- chronos-base/time_nem_demand_5t：最优 lessvg_t025（LESS-main 轨迹主实现，MASE 1.1691），vs random +15.43%，配对胜 3/3。
- timemoe-200m/time_nem_demand_5t：最优 random（Random 槽格均匀基线，MASE 1.2195）。
- timesfm-2.5/time_nem_demand_5t：最优 lessvg_t025（LESS-main 轨迹主实现，MASE 0.6708），vs random +9.51%，配对胜 3/3。

<-- v2-patrol:end -->



## Global 版本：设计理由、决策与接入记录（2026-09-17）

### 动机

当前 RAGE 的 rage_marginal 对每个参考单位先形成一个条件源分布，再按参考单位的质量
pi_j = reference_count_j / sum(reference_count) 叠加。这个机制保留了参考窗口的比例，也让
不同参考轨迹都得到自己的质量；但它同时会强制一个没有可靠源匹配的参考轨迹把质量投向
“最不差”的候选。对多模态、需要覆盖多个对抗模式的参考集合，这种约束可能有收益；当参考
方向相近，或某个参考方向在源池中没有对应内容时，它会造成过度正则化和低质量强制选择。

LESS 的全局竞争提供了另一个合理的归纳偏置：所有参考和源单位共同参与评分，源单位按总的
参考对齐得分直接竞争。高对齐组合可以挤占低对齐组合的名额，从而优先保住真正能匹配的
队列；代价是弱匹配的参考模式可能被牺牲。这正是本次新增 Global 臂要测量的机制差异。

### 新增方法

新增两个正式训练臂：

- rage_global：保留原始 RAGE 的 CRB block 表示；对所有参考 block 的方向与每个源 block
  计算余弦，按参考 block 的合法窗口数做加权平均，得到每个源 block 的一个全局分数，再用
  一次全局 value_mass（温度 0.25，无硬门）在全部源 block 间竞争。
- rage_traj_global：保留 RAGE-Traj 的 CRB 全窗口线性重建轨迹表示；对所有参考轨迹方向
  与每个源轨迹计算同样的全局加权平均分数（无硬门），再一次性在全部源轨迹窗口间竞争。

Global 版本只改变质量聚合方式，表示、CRB 配置、源池、训练配方和温度保持对应原臂一致。
它不再调用逐参考的 rage_marginal，也不为某个参考轨迹保留独立的最终采样预算。源单位
仍按自身合法窗口数计入全局 softmax 的质量，最终概率总和为 1。

这里的“全局”采用参考窗口数加权平均，而不是简单把 CRB block 数相加：CRB v2 保留所有
合法窗口，block 数只是计算表示的分块方式；用窗口数能避免不同块大小改变参考总体的权重。
轨迹级 Global 也因此保持与原 RAGE 的窗口质量语义一致。Global 与 LESS 的共同点是源单位
只有一个全局排名/分数并直接竞争；LESS 的参考轨迹表示来自均匀采样窗口均值，本实验中的
Global 表示仍来自 CRB，以便隔离“表示”和“聚合机制”两个因素。

### Seed 缩减

正式 v2 矩阵改为只运行 seed=0，以优先得到 Global 与原臂的直接方法对照并缩短训练评测
时间。方法选择、参考资产和 CRB 计算本身是确定性的；Seed 只影响训练抽样/随机状态。新的
main_v2.json 为八臂单 Seed 配置，旧 run 中已完成的三 Seed 结果不混入新矩阵。

### CRB 语义保持

本次改动没有减少 CRB 保留窗口。closest_zero_v2 仍保留所有合法窗口，稀疏预算只控制独立
梯度锚点与误差/分裂探测，其他窗口由锚点线性重建。Global 版本改变的是重建表示进入评分后的
概率聚合，不是 CRB 支持集合。

### 验证

- tests/test_full_pool_resampling.py、tests/test_full_pool_control.py、
  tests/test_full_pool_run.py：18 passed。
- resampling.py 与两个实验入口通过 py_compile。
- 合成相反参考方向测试确认 Global 与逐参考 RAGE 产生不同分布，且各自概率总和为 1。
- main_v2.json 在实验环境中通过 validate_config，并确认八臂顺序与 seeds=[0]。

### 运行接入

主实验入口已加入 rage_global 与 rage_traj_global，主控 supervisor 的文档摘要也更新为
八臂单 Seed。新 run 应使用独立输出目录，避免覆盖已有三 Seed 的旧 v2 run；输入与已验证的
共享数据可以复用，选择缓存需按新的臂名生成。
