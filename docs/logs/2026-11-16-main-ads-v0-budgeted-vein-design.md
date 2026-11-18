# main-ads-v0 后续算法设计记录

- 操作时间：2026-11-16 CST。
- 实验代号：main-ads-v0。
- 操作：将预算约束的阶段平衡 src 精算算法写入实验文档，方案代号 ads-budgeted-vein。
- 状态：仅设计草案；未实现、未启动，没有新增实验结果。
- 记录内容：说明高分配对端点 OR 合并造成的 ref 数量相关覆盖饱和；提出均布粗探索、ref 阶段等权校准、src 连续块边际收益选区、固定新增 src 锚点预算、块内均匀加密及预算匹配 uni 对照。
- 未确定事项：ref 阶段数、src 块宽、阶段内归一与块聚合、区域数 K、新增预算 B_add、探索比例 epsilon、softmax 温度 beta、块内间隔和验证指标；均标记待定。
- 更新文档：docs/exp/main-ads-v0/exp.md、docs/exp/main-ads-v0/results.md、docs/EXP.md、docs/RESULTS.md。
- 未执行算法代码或训练/评分计算。
