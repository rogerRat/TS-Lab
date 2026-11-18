# 2026-11-16 main-crb-v0 建立与 crb-histgram 批次

## 检查时间与范围

- 检查时间：2026-11-16 02:29–02:36 CST
- 实验代号：`main-crb-v0`
- 批次：`crb-histgram`
- 关联实验：`main-ads-v0`（只读继承训练检查点与两条轨迹梯度缓存）

## 任务状态与进度

- 需求：仿照 `main-ads-v0` 新建 `main-crb-v0`，仅把核心选择算法替换为 CRB，即先按预算 q_cov 均布（cover），再对所有区间按端点梯度方向抵消强度排序（`kappa = ||g_l+g_r||/(||g_l||+||g_r||)`，越小越抵消），按预算 q_can 分割，默认 q_can=q_cov，总预算 q 一般为采样间隔 32。
- 已与用户确认的两项决策：
  1. 预算语义：q 为最终等效采样间隔，各阶段合计锚点数约 N/q。
  2. 复用 main-ads-v0 已训练 LoRA 检查点与梯度缓存，只读，不重新训练/重算梯度。
- 已与用户确认的第三项决策（避免退化）：若 cover 与 split 严格各占一半，split 预算恰等于区间数，会把每个区间各分割一次，退化成均匀 q 网格，抵消排序失效。因此取 cover:split = 2:1（`cover_spacing_ratio=1.5`、`split_fraction=0.5`），保证排序真正决定哪些区间被加密，同时总等效间隔仍为 q。
- 结果：`crb-histgram` 批次 q=8/16/32/64 全部完成，耗时约 21 秒，GPU 0 占用（无需梯度计算），`.complete` 已核验。

## 新增与修改内容

- 新增核心模块 `src/rage/crb_select.py`：
  - `CRBSplitConfig`（q、cover_spacing_ratio、split_fraction、cover_spacing、split_spacing、split_budget）。
  - `interval_kappa`（相对抵消分数，零分母返回 1，尺度不变）。
  - `select_crb_positions`（cover 均布 → 区间 kappa 排序 → 上限预算 split；并列先更宽、再更左；输出 cover/split/final 锚点与逐区间记录）。
  - `selection_audit`（JSON 安全审计，含逐区间 kappa 排名与被分割/保留区间平均 kappa）。
- 新增实验入口 `scripts/exp/main-crb-v0/run_main_crb_v0.py`：
  - 只读加载 `outputs/grad_cache/main-ads-v0/traj-replay/`。
  - 对每档 q 生成 CRB 与逐侧预算匹配 uni 的 BF16 余弦矩阵（uint8 256 桶）、100/256 桶统计、精确分数统计、锚点位置文件、SVG 图与 `cosine_histograms.json`。
- 新增测试 `tests/test_crb_select.py`：18 项全部通过（`python -m pytest tests/test_crb_select.py -q`）。
- 新增文档 `docs/exp/main-crb-v0/exp.md`、`docs/exp/main-crb-v0/results.md`。
- 更新 `docs/EXP.md`、`docs/RESULTS.md`：新增实验索引行与批次结果行，最后修改日期改为 2026-11-16。

## 关键结果摘要（q=32，预算严格匹配）

| 指标 | CRB | uni 匹配 |
|---|---:|---:|
| 配对数 | 57,560 | 57,560 |
| src 锚点 | 7,195 | 7,195 |
| ref 锚点 | 8 | 8 |
| 均值 | -0.00220 | -0.01157 |
| 标准差 | 0.05962 | 0.06204 |
| 负值占比 | 0.5190 | 0.5851 |
| >0.05 占比 | 0.1809 | 0.1609 |

- 四档 q 的 CRB 均比匹配 uni 更靠近 0 且更正、负值占比更低、>0.05 占比更高或持平、标准差更小；q 越大差异越明显，q=64 时 CRB 均值转正。
- 所有档位被分割区间平均 kappa 均低于保留区间，验证排序确实选中抵消更强的区间。

## 修改的代码/脚本/配置

- 新增：`src/rage/crb_select.py`、`scripts/exp/main-crb-v0/run_main_crb_v0.py`、`tests/test_crb_select.py`。
- 新增文档：`docs/exp/main-crb-v0/exp.md`、`docs/exp/main-crb-v0/results.md`、本日志。
- 修改：`docs/EXP.md`、`docs/RESULTS.md`。
- 未修改 `main-ads-v0` 的任何代码、训练产物或梯度缓存；对其只读。

## 执行命令

```
source /root/miniconda3/etc/profile.d/conda.sh && conda activate rage-timerxl
cd /root/TS-Lab
python -m pytest tests/test_crb_select.py -q
python -u scripts/exp/main-crb-v0/run_main_crb_v0.py
```

- tmux：`main_crb_v0`、`main_crb_v0b`。
- 日志：`logs/main-crb-v0/crb-histgram-bf16-u8_*.log`、`logs/main-crb-v0/run_console*.log`。

## 未解决问题与下一步待办

- 下游训练/评估未执行：本批次只做选择与直方图对比，未验证下游指标。若需要，需继承 main-ads-v0 的 TIME1 等任务另起批次。
- ref 轨迹仅 153 个合法窗口，q≥32 时 ref 锚点仅 5–8 个，ref 侧统计波动较大；结论主要依赖 source 侧锚点。
- 可选的后续尺度：更细 q（如 4）或替换 `cover_spacing_ratio`/`split_fraction` 的敏感性分析，待用户指示。
