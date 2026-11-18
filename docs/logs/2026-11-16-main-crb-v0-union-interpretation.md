# main-crb-v0 并集结果解释巡检

- 检查时间：2026-11-16 CST；用户询问 CRB/UNI 筛选集与 ref 联训为何未普遍超过 REFT。
- 核查证据：训练脚本 src/rage/experiments/train_eval.py 使用 numpy rng.integers 对窗口行有放回均匀采样；batch=32、steps=4,000 对应 128,000 次抽样。并集 metadata.json 显示 selected source 每任务 638,202 行、ref 61–259,085 行。
- 计算结论：并集里的 ref 期望抽样比例为 0.0096%–28.8743%，对应约 12–36,959 次；单独 REFT 的 128,000 次抽样全来自 ref。这支持“并集训练稀释 ref 暴露量”机制，不等同于已证明该机制造成指标差异。
- 相似度证据：selection.json 记录 top 10% max-cosine threshold 约 0.1523–0.3984，但没保存全体窗口 score 分布。不能据此断言入选 source 与 ref 相似度过低。
- 其他尚未区分的因素：固定更新预算、稀疏探针方向/最大值聚合、任务域或梯度冲突、单 seed 不确定性以及可能的重复窗口。
- 逐任务证据和限制已追加到 docs/exp/main-crb-v0/results.md；未启动新实验。
