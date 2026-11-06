import json
from pathlib import Path

root = Path('/root/TS3')
output = root / 'temp/representation-error-20260917'
summary = json.loads((output / 'summary.json').read_text())
metrics = summary['pooled']['metrics']
labels = dict(linear='同锚点逐窗线性插值', production_block='生产块代表：两端均值常值',
              oracle_block='真值块均值（诊断下界）', crb_trajectory='CRB 重建曲线的轨迹均值',
              oracle_trajectory='真值轨迹均值（诊断下界）')
report = '''## 块平均是否抹掉 CRB 重建收益：配对诊断（2026-09-17）

### 假说与核实

`CRBEngine._make_block` 的块代表为 `c_b=(g_left+g_right)/2`，即线性重建曲线的几何中点估计，
不是额外计算的中点真梯度。最终块级 RAGE 匹配的是归一化 `c_b`，同一块的全部源窗口共享单位质量，
训练在块内均匀抽样；块内斜率没有进入匹配。端点参与了 `c_b`，但端点差向量确实被压缩掉。
半开区间的实际离散窗口均值与几何中点略有差别。RAGE-Traj 的 `aggregate_retained` 则精确平均
块归属下的逐窗原始线性插值，然后归一化，并非逐块单位方向平均。

需要区分两个完全不同的“误差积分”：

- 局部累计平方误差：`sum_t ||estimate_t-g_t||² / sum_t ||g_t||²`。这个量保留每个窗口的差异。
- 总积分/平均梯度误差：`||mean_t(estimate_t)-mean_t(g_t)|| / ||mean_t(g_t)||`。
  正负残差能在这个量中抵消；轨迹均值由线性曲线平均得到，两者总积分误差必然相同。

例如一段梯度为 `g(u)=(1,1-2u), u∈[0,1]`，两端均值是 `(1,0)`。线性插值精确，
块常值的局部平方误差积分为 `1/3`，但两者总积分误差都是 0。因此平均梯度精确并不说明局部重建精确。

### 实验

对历史缓存中的 TimeMoE 五任务、13 条已完成参考轨迹、260,041 个合法窗口，用同一套
生产 v2 CRB 2%/1%/1% 锚点进行 CPU 配对重放。真值为独立逐窗梯度的 4096 维 BF16 投影。
比较同锚点线性插值、生产块常值、CRB 轨迹常值，以及真值块/轨迹均值两个 oracle。
没有模型调用，没有新增梯度，没有使用考试集，也没有更改主实验矩阵。

平方误差按全体真值梯度能量归一化；逐窗余弦按窗口数加权；平均梯度相对误差及其方向余弦按轨迹宏平均。

| 表示 | 归一化累计平方误差 | 逐窗余弦 | 平均梯度相对误差 | 平均梯度方向余弦 |
|---|---:|---:|---:|---:|
'''
for name, label in labels.items():
    m = metrics[name]
    report += (f"| {label} | {m['normalized_squared_error']:.4f} | {m['mean_cosine']:.4f} | "
               f"{m['trajectory_macro_integral_relative_error']:.2%} | {m['trajectory_macro_mean_direction_cosine']:.6f} |\n")
report += '''
| 任务 | 线性插值平方误差 | 生产块平方误差 | 真值轨迹常值平方误差 |
|---|---:|---:|---:|
'''
for task, task_summary in summary['tasks'].items():
    m = task_summary['metrics']
    report += (f"| {task} | {m['linear']['normalized_squared_error']:.4f} | "
               f"{m['production_block']['normalized_squared_error']:.4f} | "
               f"{m['oracle_trajectory']['normalized_squared_error']:.4f} |\n")
report += '''
### 结论与限制

1. “块级没有比轨迹平均降低多少误差”在这个样本上不成立：生产块误差 0.2991，
   真值轨迹常值误差 0.8377，降低 64.29%；13/13 条轨迹均有降低。块内平均仍然保留了块间变化。
2. “块均值丢弃了线性插值的局部收益”成立：同锚点线性插值从 0.2991 降到 0.2574，
   比生产块常值低 13.97%；逐窗余弦从 0.8412 提高到 0.8697。
3. 不能把“平均”本身等同于根因：真值块均值误差只有 0.2009，比线性插值还低。
   固定分块下，真值块均值是最优平方误差常值；生产端点均值与它的差距为 0.0982，
   占生产块误差 32.83%。这混合了端点代表偏差、梯度曲率/噪声等，不能全归因于丢掉线性斜率。
4. 总积分指标无法解释局部结构损失：线性插值与 CRB 轨迹常值的平均梯度相对误差都为 5.52%，
   方向余弦都为 0.998161，但两者逐窗平方误差分别为 0.2574 与 0.8380。
   样本里的全局平均方向估计已经准确，RAGE-Traj 的局部信息损失主要是表示压缩，而非平均估计不准。
5. 这些结果仅针对历史 TimeMoE 参考样本的梯度近似，不证明下游 MASE 的因果关系，
   也不能外推到 Chronos/TimesFM 或完整正式源池。更低逐窗误差不一定等于更好的下游训练选择。

### 与方法表现的下一步对应

优先对同一正式任务、同一源支持、同一锚点比较现有块常值与逐窗线性方向的对齐分数、
softmax 分布的总变差和概率加权期望对齐。其他温度、门、参考质量聚合方式、源质量先验均固定。
只需在相同锚点上计算窗口估计，不增加梯度预算；未归一化点积可以利用线性结构解析聚合，
但逐窗余弦、逐窗指数权重是非线性的，不能用端点均值代替。
若概率分布变化明显，再在一个代表任务上做 seed0 配对训练，对比同预算最终 MASE；
若只提高重建精度却没有改变有效选集或 MASE，就不能把块平均认定为性能差距的主因。

若目标本身是平均参考损失的一阶下降，平均参考梯度本来就是正确的一阶目标；
需要保留局部梯度的理由是保留多个模式、避免归一化与 softmax 前的抵消，而非“任何平均都不对”。

产物：`temp/representation-error-20260917/{protocol,rows,summary}.json`；
复现脚本：`temp/replay_gradient_representations.py`（只读 CPU 重放）。
'''
(output / 'report.md').write_text(report)
marker = report.splitlines()[0]
for path in (root/'docs/MAIN-v2.md',):
    text = path.read_text()
    if marker not in text:
        end = '<!-- v2-notes:end -->'
        if end in text:
            text = text.replace(end, report + '\n\n' + end, 1)
        else:
            text = text.rstrip() + '\n\n' + report
        path.write_text(text)

incident = '''
## Global 正式入口分派修正（2026-09-17）

审计发现新增 Global 臂先被 `TEMPERATURE_ARM_BASE` 映射为原 RAGE 核心，而入口先判断 base，
导致 Global 的 name 分支不可达。先前概率总和与合成核心测试未覆盖正式入口，不能证明实际实验运行的是 Global。
已将 Global name 分支移到原 base 分支之前，新增两项正式入口分派测试；另用解析线性聚合
`sources @ (reference_weights @ refs)` 代替完整参考×源矩阵，并用显式矩阵结果做等价测试。
四组相关测试 22 passed。两臂仍无硬门，轨迹近零源处理与原 RAGE-Traj 一致。
已通知主控 Claude 立即隔离错误 Global 选集和潜在训练产物并重新生成，原六臂产物无需因本问题作废；
重新运行时必须检查 scoring_unit 与 reference_audit 的全局聚合语义。
'''
for path in (root/'docs/MAIN-v2.md',):
    text = path.read_text()
    if '## Global 正式入口分派修正（2026-09-17）' not in text:
        end = '<!-- v2-notes:end -->'
        if end in text:
            text = text.replace(end, incident + '\n\n' + end, 1)
        else:
            text = text.rstrip()+'\n\n'+incident
        path.write_text(text)
print('Representation report and dispatch tests recorded inside MAIN-v2 preserved notes')
