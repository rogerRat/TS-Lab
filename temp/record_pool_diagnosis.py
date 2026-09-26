from pathlib import Path

path = Path('/root/TS3/docs/MAIN-v2.md')
content = path.read_text()
heading = '## 源池规模、覆盖与有效抽样诊断（2026-09-17）'
note = '''

## 源池规模、覆盖与有效抽样诊断（2026-09-17）

用户怀疑效果不佳是否源于可行源池过小。本次只读 CPU 审计，没有替换正式源池
或增加主实验训练。脚本 temp/pool_audit.py，结果 temp/pool-audit-20260917.json；
来源组成 temp/pool-composition-20260917.json。

实际候选池：1082 完整轨迹、9,674,927 时间点、9,052,777 stride1 合法重叠窗口，
N=16,284 个非重叠 576 点窗口等价量。完整源索引 840,209 条轨迹、
6,334,649,697 个索引点；剔除 dropped 和长度不足576后为742,312条、
6,320,130,088点，尚未逐条核实有限值和非恒定。实际候选池约为总索引
0.13% 轨迹、0.15% 点量，绝不是把整个 profile-10b 交给选择器。
精确池构造从固定 seed 的轨迹随机顺序中寻找 N 的完整轨迹子集，1253条被检查、
1083条合格、1082条选中；未按任务领域保证覆盖。

来源点量：WeatherBench daily 4,806,682 + weekly672,014 =56.63%；
solar 1,261,440 + solar_1h70,080 =13.76%；Quito hour约16.51%、
minute约7.93%；UTSD约2.51%。Quito/UTSD内部细分领域未在本次核实，
不能由这些顶层标签断言没有CPU/水务可迁移模式。气象点量占优和候选轨迹少
使覆盖不足成为合理怀疑，不构成“扩大池一定提升效果”的证明。

训练曝光（每个 task/arm，seed0）：TimeMoE255步×64=16,320次窗口抽样，
Chronos1018步×32=32,576次，TimesFM1336步×32=42,752次。
所谓epoch由N=16,284定义，不是遍历905万个重叠窗口；约905万窗口不能当独立样本。
后训练没有与预训练点量或参数量比例绑定的统一门槛。Time-MoE论文第4.2节
在六个目标任务训练集上只微调1epoch，表9训练规模涵盖ETT/Weather/GlobalTemp；
应以目标覆盖、独立实体、质量、训练曝光和验证曲线判断，而非预训练规模。
论文 https://arxiv.org/abs/2409.16040 ，已读取PDF第4.2节和表9；
LESS https://arxiv.org/abs/2402.04333 的5%数据结论不是跨任务的绝对数据量标准。

已完成NEM的分布证据：RAGS-gateless块级窗口概率ESS/全池窗口数，Chronos99.49%、
TimesFM99.72%、TimeMoE93.51%；非正梯度匹配的joint mass分别40.55%、44.88%、
30.11%。同一候选池LESS窗口ESS比例分别2.32%、9.95%、1.47%。Chronos
NEM MASE：LESS1.1305、random1.4139、RAGS1.5479、RAGS-Global1.3594；
TimesFM LESS0.6727、random0.7266、RAGS0.7264。TimeMoE NEM则LESS1.3342
不及random1.1918和RAGS1.2181，不能泛化为“LESS集中所以总是更好”。
TimeMoE Azure LESS轨迹概率ESS约1.00005、预期精确窗口起点重复约40.88%；
RAGS块级ESS比例98.78%，与选择稀释及另一方法的过度集中同时存在相符。
ESS定义1/sum(q_start²)，轨迹ESS定义1/sum(p_trajectory²)，是概率集中度，
不是独立信息量；本次预期重复只统计相同窗口起点，不含高度重叠的不同起点。
非正joint mass既可能来自弱来源，也可能来自参考方向冲突/表示误差，不能视为
“未覆盖的参考比例”；Global平均评分很弱也不能替代每参考最佳匹配的分析。

建议先做小型、独立新run的单seed因果诊断，保留主实验结果可比性：
1. 原池与嵌套4倍候选池，固定曝光次数、有效batch、学习率序列和评估口径。
   现代码budget_windows强制等于pool_n，直接扩池并保持epoch会同时增加训练步数，
   因而不能区分候选覆盖和额外训练。实验须显式分离candidate规模和曝光预算。
2. 原池与同点量/等价量的领域更均衡池，固定训练曝光；区分来源构成和总量。
3. 固定原池，只比较曝光B与2B，固定其他条件；识别训练不足与更多更新导致的退化。
先选NEM及Azure两个已观察到不同选择表现的任务，比较random/LESS/RAGS-Global，
按参考权重审计最佳/前k源对齐及其提升，结合验证集结果判断（不以测试集选超参）。
若扩池增加最佳匹配和目标性能，支持覆盖不足；只增曝光有效则支持训练不足；
仅均衡来源有效支持构成偏置。当前更直接的证据是组成偏置与RAGS选择稀释，
尚不能把方法效果不好主要归因于纯粹的候选数量不足。
'''
marker = '<!-- v2-notes:end -->'
if marker not in content:
    raise RuntimeError('Missing persistent notes marker')
if heading not in content:
    path.write_text(content.replace(marker, note + '\n' + marker))
else:
    path.write_text(content.replace('30.14%。同一候选池LESS窗口ESS比例',
                                    '30.11%。同一候选池LESS窗口ESS比例'))
print('Recorded pool provenance, distributions, exposure and causal experiment proposal')
