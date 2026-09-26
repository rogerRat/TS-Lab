# CPUStuckArrow：Arrow 数据读取与动态采样开发约束

> 历史归档：保留当时的设计、操作或实验证据，不作为当前执行指令。当前入口见 `docs/README.md`。

## 背景

2026-08-11 的 `d-ot` 训练在进入模型构建前长时间占满 CPU，GPU 尚未被使用。日志停在 `Loading source Feature Bundle`，进程 RSS 达到约 27 GB。根因不是 Sinkhorn 本身，而是启动阶段将大型嵌套 Arrow 表一次性转换为 Python 对象，并随后重复包装为多层 Python 计算对象。

`d-kde` 与 `d-ot` 共用这条启动路径，因此不能只修复 OT 求解器。两种 dynamic 策略还曾经把 source/reference pair 做全量两两计算，造成额外的 CPU 和内存放大。

## 强制约束

### Arrow 读取

- 禁止对大型或嵌套 Arrow 文件使用 `RecordBatch.to_pylist()` 后再保存完整列表。
- 需要处理大量记录时，使用 `iter_arrow_rows()` 或等价的批次内逐行转换；单行对象处理完成后必须允许释放。
- 如果需要校验 ID 集合，使用轻量 ID 两遍扫描；不要同时保留完整 Arrow 行、ID 字典和计算对象。
- 不得为了方便查找而复制完整嵌套 `segments`。应优先按 Arrow 原顺序流式构建，或使用紧凑的 NumPy/Arrow 表示。
- 新增 Arrow 读取代码时必须说明峰值内存随记录数的增长方式，并覆盖大于单个 batch 的测试场景。

### Python 对象与 NumPy

- 不要把每个 segment/pair 先建成 Python dataclass，再在求解前重新堆叠成 NumPy 数组，除非数据规模有明确上限。
- 不要在热路径的 property 中反复执行 `np.asarray()`、减法或拼接。特征、状态、转移量应一次构建并缓存。
- 任何临时数组都应限制在 block 范围内；不得保存每个 reference 的完整 affinity 数组列表。

### Dynamic KDE/OT

- 禁止直接创建 `source_count × reference_count` 的全量距离矩阵。
- 近邻筛选必须使用近邻索引或分块 top-k；`neighbor_count` 必须真正限制候选计算规模。
- KDE 应按 reference block 或逐 reference 累加结果，不能把所有 reference 的 source-sized 中间数组保存在 `affinity` 列表。
- OT 的 cost matrix 仍须有明确的候选规模和内存预算；不能因为前面使用了近邻索引就默认完整 cost matrix 安全。当前超过预算时必须在分配前失败并报告规模。
- terminal OT/KDE 也必须遵守同样的近邻和分块约束，不能先生成全量距离矩阵再筛选。

## 必须保留的可观测性

启动阶段至少记录：

- Arrow record 数、有效轨迹数和 segment 数；
- dynamic pair 数量；
- 每个 reference 的候选数量；
- 实际求解矩阵形状、估计字节数和耗时；
- 进入 adapter/model 构建的明确日志。

长于 60 秒的阶段必须有进度日志。任何可能超过配置内存预算的阶段应在分配大矩阵前失败，并报告规模和建议配置。

## 测试门槛

- 单元测试必须验证 Arrow 读取结果与旧接口一致，包括 ID 重复、ID 缺失、丢弃记录和 segment 几何校验。
- dynamic KDE/OT 测试必须覆盖无支持、近邻筛选、terminal segment 和质量归一化。
- 至少添加一个回归测试，确保求解器不再构造全量 source/reference 状态距离矩阵；测试应使用 monkeypatch 或受控数组检查分块/近邻路径，而不是依赖机器内存大小。
- 运行真实规模前，先在小 catalog 上验证日志能到达 adapter/model 阶段；禁止用真实大任务作为首次回归测试。

## Review checklist

提交涉及 Arrow、Feature Bundle 或 sampling 的代码前，逐项确认：

1. 是否存在全量 `to_pylist()`、全量 `list`、`dict` 或嵌套对象复制？
2. 是否存在 source/reference 两两广播、`argsort` 或隐式 dense 矩阵？
3. 是否会重复计算同一 feature/state/delta？
4. 峰值内存和复杂度是否有显式估算？
5. CPU-only 的启动阶段是否有进度和超时可观测性？
6. `d-kde` 和 `d-ot` 是否同时验证，而不是只验证一个 solver？

## 当前修复原则

当前实现使用逐行 Arrow 迭代器避免完整 Python 行列表，并让 dynamic KDE 不再保存所有 reference 的 affinity 数组；dynamic OT 使用近邻索引替代全量状态距离矩阵。后续仍应继续把 terminal 求解和 OT cost matrix 改为有预算的分块实现。
