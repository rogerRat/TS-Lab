# main-reconst-v0 实验设计

## 基础信息

| 字段 | 内容 |
|---|---|
| 实验代号 | `main-reconst-v0` |
| 目标 | 复用 `main-ads-v0` 已缓存的完整轨迹梯度，回放比较「最长区间二分」与「预算严格匹配的均匀锚点」在同等锚点数下的线性插值重建质量 |
| 假设 | 在同等锚点预算下，反复二分当前最宽区间能比严格等距布点更贴近真实的梯度变化，从而提高重建方向余弦、降低相对 L2 误差 |
| 文档创建时间 | 2026-11-16 CST |
| 最后修改时间 | 2026-11-16 CST |
| 实际实验启动时间 | 2026-11-16 12:28:34 CST |
| 当前状态 | `bisect-vs-uniform` 批次完成；两条轨迹 × 六档 q × 两种锚点布局全部生成，`.complete` 已核验 |
| 继承实验 | `main-ads-v0`（只读复用训练检查点与梯度缓存），角色划分同 `main-crb-v0` |

## 模型与训练

- 本实验不做任何训练、不做任何梯度反向传播，也不写入 `main-ads-v0` 的产物。
- 复用检查点：`outputs/train/main-ads-v0/checkpoint/adapter/`（TimerXL 84M，LoRA `q_proj`/`v_proj`，r=8、alpha=16、dropout=0，2,000 步，seed 0）。
- 复用梯度缓存：`outputs/grad_cache/main-ads-v0/traj-replay/`，TimerXL 原生 V6 单窗口损失对全部可训练 LoRA 参数的梯度，Count-Sketch 投影 4,096 维、seed 0，float16 存储，逐窗口独立计算。
- 窗口：480 历史点 + 96 预测点，总长 576，stride=1。
- 全部写入落在 `outputs/selection/main-reconst-v0/`，日志落在 `logs/tools/replay/` 与 `logs/main-reconst-v0_console.log`。

## 数据处理

| 字段 | 内容 |
|---|---|
| 轨迹 ordinal 0（长） | `monash_australian_electricity / T000004`，230,161 个合法窗口起点 |
| 轨迹 ordinal 1（短） | `monash_fred_md / T000066`，153 个合法窗口起点 |
| 合法起点定义 | `TimeSeriesFragment.valid_starts(context=480, prediction=96, stride=1)`，1-based 物理起点，缓存行序即 0-based 顺序 |
| 数据校验 | 只校验行数、形状、有限性与 `.complete` 标记；不做哈希级校验 |

## 锚点预算与两种布局

- 锚点预算 `K = max(2, ceil(N / q))`，两端点必然计入，两种方法锚点数完全相同。
- 布局 A：最长区间二分（`bisect`）
  - 从两端点开始，反复取当前最宽的相邻锚点空隙，在其整数中点加入锚点，直到达到 K。
  - 并列时先取更宽空隙，再取更靠左空隙，完全确定性、不使用随机数。
  - 实现复用 `rage.crb.CRBEngine` 的 cover 阶段：`cover_density = K/N`，`error_density = 0`、`split_density = 0`、`detection = "none"`，因此只执行纯二分覆盖。
- 布局 B：预算匹配均匀锚点（`uniform`）
  - `np.rint(np.linspace(0, N-1, K))`，即 `rage.experiments.v7_scores.uniform_anchors` 的冻结口径，与 `main-crb-v0` 的 `uni-budget-matched` 完全一致。

## 重建与评估

- 重建规则：`rage.window_reconstruction.interpolate_window_gradients`，相邻真实锚点之间逐窗口线性插值，锚点位置精确返回真实梯度，从不外推。
- 逐窗口指标：
  - 相对 L2 = `||estimate - truth|| / ||truth||`；
  - 方向余弦 = `cos(estimate, truth)`，在投影 4,096 维空间内计算。
- 汇总指标：
  - 全部窗口平均相对 L2、平均余弦；
  - 剔除锚点后（`non_anchor`）的平均相对 L2、平均余弦；
  - 余弦 ≥ 0.9 的窗口占比；
  - 余弦中位数、p10、最小值、最大值；
  - 锚点间距范围、均值与变异系数（用于确认布局差异）。
- 主判据：同一 q、同一锚点数下 `bisect` 与 `uniform` 的等权全窗口平均余弦与平均相对 L2；并给出逐窗口余弦胜负计数作为稳健性参考。
- 无"预算匹配"以外的对照；两种方法锚点数严格一致，因此矩阵/计算预算相同。

## 执行细节

- 入口（配置驱动，复用 `scripts/tools`）：`python -u -m scripts.tools.run --config configs/experiments/main-reconst-v0/run.yaml`
  - 阶段 `replay` 调用 `scripts/tools/replay.py`；核心逻辑在 `src/rage/replay_cache.py`。
  - 直接运行单阶段可用 `python -u -m scripts.tools.replay --config configs/experiments/main-reconst-v0/replay.yaml`。
- 说明：在该环境中从 `scripts/tools/` 目录直接执行会因同名 `select` 遮蔽标准库 `select` 而失败，因此统一使用 `-m` 模块形式。
- 配置文件：`configs/experiments/main-reconst-v0/replay.yaml`、`configs/experiments/main-reconst-v0/run.yaml`。
- Conda 环境：`rage-timerxl`（只用到 numpy 与内存映射，无 GPU 计算）。
- 参数：`q_values = [4, 8, 16, 32, 64, 128]`、`chunk_rows = 4096`、`cosine_ge = 0.9`、`save_window_metrics = true`、`plot = true`。
- tmux 会话：`main_reconst_v0`；控制台日志 `logs/main-reconst-v0_console.log`。
- 输出：`outputs/selection/main-reconst-v0/bisect-vs-uniform/`。
- 实测总耗时 12.2 分钟（纯回放，未做梯度计算）。

## 批次管理

| 批次 | 内容 | 状态 | 主要参数 | 输出 |
|---|---|---|---|---|
| `bisect-vs-uniform` | 两条轨迹 × q=4/8/16/32/64/128 × 两种锚点布局的线性插值回放重建 | 完成 | K=max(2,ceil(N/q))；两端点计入；chunk_rows=4096；cosine_ge=0.9 | `outputs/selection/main-reconst-v0/bisect-vs-uniform/` |
| `downstream` | 下游训练/评估 | 未启动 | 未定义 | 未生成 |

## 决策记录

| 项目 | 设置 | 状态 |
|---|---|---|
| 「最大二分」定义 | 单轨迹最长区间二分（CRB cover 阶段），非二分图匹配 | 已由用户确认 |
| 评估范围 | ordinal 0 与 ordinal 1 两条轨迹 | 已由用户确认 |
| 预算档位 | q=4/8/16/32/64/128 | 已由用户确认 |
| 锚点预算 | K=max(2, ceil(N/q))，两端点计入，两方法相同 | 已确定 |
| 均匀口径 | `np.rint(linspace(0, N-1, K))`，与 main-crb-v0 预算匹配 uni 相同 | 已确定 |
| 重建规则 | 相邻锚点线性插值，锚点精确、不外推 | 已确定 |
| 是否重算梯度 | 否，纯回放，只读复用 main-ads-v0 缓存 | 已由用户确认 |
| 工具归属 | 复用 `scripts/tools`：新增 `replay.py` 阶段入口，注册进 `run.py` | 已由用户要求 |
| 下游任务 | 本批次不执行 | 未确认（待用户指示） |

## 局限

- 余弦与相对 L2 都在 Count-Sketch 4,096 维投影空间内度量，绝对值不代表完整参数空间。
- 两条轨迹只覆盖一个模型状态；短轨迹仅 153 个窗口，q≥32 时仅剩 2–5 个锚点，统计波动大。
- 本批次只做重建回放对比，未进入下游训练/评估，不能据此断言下游指标收益。
