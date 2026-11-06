from pathlib import Path
p=Path("scripts/exp/main_v2_supervisor.py")
s=p.read_text()
s=s.replace("'- 继承最终 v1 的全部裁定：九任务、三模型、六训练臂、seeds 0/1/2，统一数值 softmax T=0.25、无门；Quito/LTSV 延续移除。',",
"'- 继承最终 v1 的全部裁定：九任务、三模型；八训练臂（含 rage_global 与 rage_traj_global），只跑 seed 0，统一数值 softmax T=0.25；Quito/LTSV 延续移除。',")
p.write_text(s)

p=Path("docs/MAIN-v2-notes.md")
with p.open("a") as f:
    f.write('''


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
  一次全局 value_mass（温度 0.25，正对齐门）在全部源 block 间竞争。
- rage_traj_global：保留 RAGE-Traj 的 CRB 全窗口线性重建轨迹表示；对所有参考轨迹方向
  与每个源轨迹计算同样的全局加权平均分数，再一次性在全部源轨迹窗口间竞争。

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
''')
