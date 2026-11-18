# 2026-11-16 main-reconst-v0 二分锚点回放重建对比

## 检查时间与范围

- 检查时间：2026-11-16 12:28–12:54 CST
- 实验代号：`main-reconst-v0`
- 批次：`bisect-vs-uniform`
- 关联实验：`main-ads-v0`（只读继承训练检查点与两条完整轨迹梯度缓存）

## 需求与已确认决策

- 需求：基于已有的完整轨迹梯度，对比「最大二分」与「均匀锚点」的重建效果。
- 用户确认一：「最大二分」指单轨迹的最长区间二分（即 CRB cover 阶段的规则），不是二分图最大权匹配。
- 用户确认二：评估两条轨迹（ordinal 0 长轨迹、ordinal 1 短轨迹），预算档位 q=4/8/16/32/64/128。
- 用户后续要求：只做重放、不重算梯度；并尽可能复用 `scripts/tools` 下的脚本。

## 实现方式（复用优先）

- 未新增实验专属脚本目录；改为复用现有工具链：
  - 新增 `src/rage/replay_cache.py`：缓存只读读取器、锚点预算、布局构建、线性插值重建与配对比较。
  - 新增 `scripts/tools/replay.py`：配置驱动的回放阶段入口，与既有 `warmup/grad_cache/select/train/eval` 并列。
  - 注册进 `scripts/tools/run.py` 的 `TOOLS` 表，并把子阶段调用改为 `python -m` 模块形式，避免 `scripts/tools/select.py` 遮蔽标准库 `select` 的既有问题。
  - 复用 `rage.crb.CRBEngine` 的 cover 阶段做最长区间二分（`error_density=0`、`split_density=0`、`detection=none`）。
  - 复用 `rage.experiments.v7_scores.uniform_anchors` 做预算匹配均匀锚点，与 `main-crb-v0` 的 `uni-budget-matched` 口径一致。
  - 复用 `rage.window_reconstruction.interpolate_window_gradients` 做线性插值重建。
- 新增配置：`configs/experiments/main-reconst-v0/replay.yaml` 与 `run.yaml`。
- 新增测试：`tests/test_replay_cache.py`（10 项），与既有 `test_crb_select.py`、`test_window_reconstruction_protocol.py`、`test_crb.py` 一并通过（共 41 项）。

## 执行命令

```
source /root/miniconda3/etc/profile.d/conda.sh && conda activate rage-timerxl
cd /root/TS-Lab
python -m pytest tests/test_replay_cache.py tests/test_crb_select.py tests/test_window_reconstruction_protocol.py tests/test_crb.py -q
python -u -m scripts.tools.run --config configs/experiments/main-reconst-v0/run.yaml
```

- tmux 会话：`main_reconst_v0`。
- 控制台日志：`logs/main-reconst-v0_console.log`（跑完后清理）。
- 工具日志：`logs/tools/replay/main-reconst-v0_20261116124124.log`。
- 首轮（12:28–12:39）因 SVG 中 `width="100%%"` 转义错误重跑；修正后 12:41–12:53 完成，耗时 12.2 分钟。

## 结果摘要

- 24 个布局（2 轨迹 × 6 档 q × 2 方法）全部完成，`.complete` 与 `replay_metrics.json` 已核验；本批次未做任何梯度计算。
- 长轨迹（230,161 窗口）：

| q | 平均余弦 二分/均匀 | 平均相对 L2 二分/均匀 | 余弦差（二分−均匀） |
|---:|---|---|---|
| 4 | 0.96551 / 0.97443 | 0.20969 / 0.19167 | −0.0089 |
| 8 | 0.89122 / 0.89844 | 0.40440 / 0.40465 | −0.0072 |
| 16 | 0.76033 / 0.76237 | 0.62376 / 0.63838 | −0.0020 |
| 32 | 0.59687 / 0.60147 | 0.83845 / 0.82849 | −0.0046 |
| 64 | 0.44978 / 0.46887 | 1.01702 / 0.99125 | −0.0191 |
| 128 | 0.30067 / 0.30634 | 1.18500 / 1.17318 | −0.0057 |

- 短轨迹（153 窗口）结论不一致且样本过小：q=4 二分余弦略高（+0.0032），q=8 均匀明显更好（−0.0161），q=16 二分相对 L2 更好（−0.0740）但窗口胜场更少，q≥32 两法锚点相同、指标完全相同。
- 布局核验：二分 gap 在 `[q, 2q-2]` 内变化、CV≈0.30；均匀严格等距（CV≈0.000）。两者锚点数与等效间隔完全一致，差异只来自位置分布。
- 判读：在统计最稳定的一条长轨迹上，同等预算的严格均匀锚点在全部六档 q 上平均余弦更高；最长区间二分没有稳定重建优势。这与历史 `v0-crb-reconst-*`、`v0-crb-mech-test` 的结论方向一致。

## 修改的代码/脚本/配置

- 新增：`src/rage/replay_cache.py`、`scripts/tools/replay.py`、`tests/test_replay_cache.py`、`configs/experiments/main-reconst-v0/replay.yaml`、`configs/experiments/main-reconst-v0/run.yaml`。
- 修改：`scripts/tools/run.py`（新增 `replay` 阶段并改为 `-m` 模块调用）。
- 新增文档：`docs/exp/main-reconst-v0/exp.md`、`docs/exp/main-reconst-v0/results.md`、本日志。
- 更新：`docs/EXP.md`、`docs/RESULTS.md`、`docs/SCRIPT_TOOLS.md`。
- 未修改 `main-ads-v0` 的任何代码、训练产物或梯度缓存；对其只读。
- 清理：本实验过程中删除过首轮误建的 `scripts/exp/main-reconst-v0/`、`src/rage/bisect_anchors.py`、`tests/test_bisect_anchors.py` 及其 smoke 产物与早先日志（用户要求改为复用 `scripts/tools`）。

## 未解决问题与下一步待办

- 下游训练/评估未执行：本批次只做重建回放，未验证下游指标。
- 只有两条轨迹、一个模型状态；短轨迹 q≥32 退化为两端点，统计意义有限。
- 可选的后续尺度：更多轨迹、更细 q、与 CRB 完整三阶段（含 error/split）的同预算对照，待用户指示。
