# v1-rage-ads 平均名次与零概率支持审计

- 检查/操作时间：2026-11-14 14:12–14:14 CST。
- 实验代号 / 批次：`v1-rage-ads` / `rolling-eight-task`。
- 平均名次口径：任务内按 MASE 升序排名，名次 1 为最好；八个任务等权平均，无并列。
- 平均名次：四臂比较时 fullRand/rage/ancUni/Traj 分别为 2.750/1.625/2.625/3.000；只比较 rage/ancUni/Traj 三个数据选择臂时分别为 1.375/2.250/2.375。
- 零概率定义：某个完整 src-96 合法窗口起点在最终 sampler 概率向量中概率严格等于零的比例。每任务总体 2,666,312 个起点；八任务合并分母 21,330,496。此值表示候选总体中无抽样支持的窗口份额，不是有限次抽样命中比例。
- 精确合并结果：fullRand 0/21,330,496=0.00%；rage 12,605,710/21,330,496=59.10%（任务范围56.25%–76.50%）；ancUni 16,841,624/21,330,496=78.96%（78.14%–84.63%）；Traj 11,480,931/21,330,496=53.82%（4.91%–94.69%）。
- 计算方法：ADS两臂读取最终概率descriptor并用训练同一 `SourceProbabilitySampler` 复原精确正支持；Traj按未选轨迹及零概率轨迹概率汇总合法窗口数；fullRand按全池均匀分布记零。没有抽样近似、哈希或整池稠密概率数组。
- 产物：`outputs/selection/v1-rage-ads/rolling/audit/probability-support/probability-support.jsonl`、`summary.json`、`.complete`；复现脚本 `scripts/exp/v1-rage-ads/audit_probability_support.py`；运行日志 `logs/v1-rage-ads/ads-probability-support_20261114141222.log`。
- 汇总解释：主臂rage约有40.90%的source窗口获得正概率支持；ancUni约21.04%。这是算法按正 anchor 邻域分配概率后的稀疏支持结果，说明大量候选窗口不能被该臂采到；是否因此影响下游效果不能只凭该比例判断。
