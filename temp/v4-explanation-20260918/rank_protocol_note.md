> 已被用户最新决策部分替代：源侧不做轨迹双层平权。最新源数据量先验、排名协议及落实结果见docs/experiments/rank_decisions_20260918.md；本文保留为此前草案，不能作为当前实验执行定义。

# 双层权重澄清与排名统一强度协议草案（2026-09-18）

本轮只核对实现、读取已有评分、补充排序诊断；不修改正式代码或实验协议，不启动正式任务。

## 数值澄清
TV=.705对应Chronos NEM LESS/Traj-max实际分布，不是Finland。Chronos Finland TV=.282，TimesFM Finland TV=.209。TV=1-Σmin(p,q)，不是排名相关系数。低强度时即使排序不同也可得到小TV。
LESS/Traj-max的1/N_ref缩放完全不改变排序，但锚点改变会影响排序。现有概率温度匹配不是因果训练消融。
排序诊断见less_traj_rank_audit.json：Chronos NEM Spearman .918/top10%重叠91.7%，Finland .924/79.8%；TimesFM NEM .713/67.0%，Finland .600/39.4%。

## 当前实现的两侧权重
src/rags/experiments/full_pool.py:_benchmark_block_marginal参考权重alpha_rb=(1/R_ref)(L_rb/L_r)，每条参考轨迹意见权重总和1/R_ref。
src/rags/experiments/reference_opinions.py全局路径对源scores直接value_mass；源端没有counts或轨迹基准质量。线性全局路径rags_benchmark_marginal亦为equal_block。_rags把该beta直接作为源块概率交给ResampledWindows。
因此参考双层意见平权已落实，源块全局Softmax没有保证源轨迹基准平权。两条源轨迹分别1/2个同分块，源轨迹质量为1/3、2/3。
若源端也需要轨迹平权，应定义基准q_tb=(1/N_source)(L_tb/L_t)，并使用p_tb ∝ q_tb exp(S_tb/T)。平分数时源轨迹总质量才是1/N_source；评分差异允许选后轨迹质量不同。若要求任何评分下选后轨迹质量都固定1/N_source，Traj方法就不能在轨迹间选择。

## 模长风险的解释边界
均匀源锚点361856个：TimesFM top1%占Σ||g||的99.62%，最大投影模长5.96e14。模长总和不是向量和、平方能量或实际更新量。
补充2100个控制+最大模长样本，log梯度模长与log输入std相关.946。极大缓存梯度源窗口标准差约450万。原生MSE使单位/幅度可能影响模长，尚未通过归一化干预确认全部原因。
原始均值、单位化后均值、批裁剪后梯度期望、AdamW更新不同。当前训练先求批平均再clip_grad_norm_至1；全局范数裁剪保留批方向，不能消除一个大梯度窗口对批方向的主导，也不等于逐窗口裁剪。E[clip(g_batch)]不等于clip(E[g_batch])。
MASE评估有序列尺度归一化，原生raw-MSE梯度不必等同平均MASE改善方向。原始均值方向诊断在TimesFM须结合模长分层、去掉极端尾部的敏感性、真实批梯度/实际更新、目标参考loss变化。
不据此判定缓存错误、训练必然失稳或需立即更改当前协议。

## 推荐的排名消融定义
先固定共同源基准测度。若沿用轨迹平权设计，合法窗w属于轨迹t时q_w=1/(N_source L_t)。Traj单位q_t=1/N_source；Block单位q_tb=L_tb/(N_source L_t)。
对原始方法最终分数按该q计算加权分位u（分数越高u越高；真并列使用整组加权中位分位，不按ID制造次序）。
令p_j(beta)=q_j exp(beta*u_j)/Σq_k exp(beta*u_k)。原始分数只用于排序，分数尺度/间距被丢弃；beta控制选择强度。
为避免单位数和块长度造成比较偏差，按共同q校准目标rho=1/Σ_j(p_j²/q_j)，而不是分别匹配块/轨迹的原生单位ESS。rho=1表示p=q，越小越集中。每方法求beta达到同一预先固定rho，并报告KL(p||q)、高排名分位累计质量、轨迹质量与真实窗口ESS等补充指标。单一rho相同不保证整个概率形状相同。
若更强调固定分位概率形状，可令f_beta(u)=beta*exp(beta*u)/(exp(beta)-1)，对每个排序区间I_j（宽度q_j）积分分配p_j=∫I_j f_beta(u)du；并列组先整体积分再按组内q比例分配。粗粒度Traj与并列会平滑分位概率，因此两臂完全相同的逐窗口密度形状未必可实现。
此实验回答“在同等预算、共同先验和共同强度下，哪个方法排序更有用”，不完全等同于“谁的概率校准更好”。微小、不稳定的分差也可能被排名转换放大；保留真并列，并检查排序稳定性。
若当前Block从equal_block改为共同q，同时发生源先验变化；需单独标记并保留原先验对照，不能将全部变化归因于强度剥离。优先在LESS与Traj-max共享的1082源轨迹上做排名映射，先隔离锚点排序；Block/Traj再使用显式共同q协议。

