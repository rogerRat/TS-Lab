# main-crb-v0 revised resample design constraints

- 记录时间：2026-11-16 CST
- 批次背景：用户在 main-crb-v0 / q16-src256-eight-task-resample 与 main V7 方法对照后，指定后续新版的设计约束。
- 本记录描述新版目标，不改写当前批次的实验身份、代码、配置、结果或运行状态。

## 用户已明确的新版约束

- 抽样对象不再是单个窗口；分别以 block 或 trajectory 作为排名和抽样对象。
- 对象概率由固定的 V5 rank-volume 映射生成。source 对象概率不再乘对象所含合法窗口数。
- 每次先按对象概率抽取 block 或 trajectory，再在被抽中的对象内部均匀抽取合法窗口。对象总概率不因对象长度而改变。
- Reference 与 source 两侧都不按合法窗口数做长度加权。
- Cover 使用旧版最长区间优先的中点二分方式，不用均匀 linspace cover。
- CRB 预算比例固定为 cover:error:split = 1:0:1；不运行 error 阶段，只运行 cover 与 split。

## Reference 当前实现核对

- 当前窗口评分路径使用较密的 q8 reference block directions；各 reference block 按其拥有的合法起点数占该轨迹合法起点数的比例加权，再对 reference trajectories 等权。因此它不是跨 reference blocks 的简单均值，且包含 block 内长度权重。
- 当前轨迹评分路径从各 reference trajectory 的 q8 anchors 计算轨迹表示：轨迹内先平均 raw projected anchor gradients，再 L2 normalize；然后对 reference trajectory vectors 做等权算术平均。这个路径没有按轨迹合法窗口数加权。
- 因此新版采用何种 reference mean 仍需确定到向量层级和归一化顺序。一个与无长度加权相符的实现候选是：每条 ref 轨迹用 q8 dense anchors 求一个均值向量，逐轨迹归一化，再对 ref 轨迹等权平均；不能把所有 anchor 混在一起直接平均，否则长轨迹因 anchor 更多而会获得更大隐式权重。

## 现有批次边界

- 当前 q16-src256-eight-task-resample 已运行/记录的逐窗口概率、均匀 cover 和相应 MASE 仍属于原批次协议。
- 若实现上述新版，须另设批次与配置身份；不覆盖或重解释当前批次的缓存、概率载荷和结果。
- 尚未修改代码、参数或队列，也未检查当前实时进程。
