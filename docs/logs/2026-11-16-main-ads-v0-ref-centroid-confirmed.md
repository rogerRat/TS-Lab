# main-ads-v0 ref 平均方向评分确认记录

- 操作时间：2026-11-16 CST。
- 实验代号与计划批次：main-ads-v0 / ads-budgeted-vein。
- 操作：根据用户确认，记录归一化 ref 平均方向作为 src 评分定义。
- 已确认公式：对每个 ref 梯度草图单独作 L2 归一化后求均值方向 r_bar；src 分数为归一化 src 与 r_bar 的内积，等价于 src 对所有 ref 锚点余弦的算术平均。
- 设计目的：避免“任一 ref 配对高分即激活”造成的 src 端点并集饱和；后续按 src 锚点/连续块直接分配固定精算预算。
- 已记录的限制：单一平均方向会弱化阶段专属匹配；以 rho=||r_bar|| 诊断 ref 方向抵消，低 rho 时的阶段化回退尚待确认。
- 待定事项：src 块宽、块分数汇总、固定预算 B_add/K、探索比例 epsilon、q 对应块内间隔、rho 阈值及回退配置。
- 状态：仅更新文档；未实现、未运行，没有新增结果。
- 更新文档：docs/exp/main-ads-v0/exp.md、docs/exp/main-ads-v0/results.md、docs/EXP.md、docs/RESULTS.md。
