# docs 目录重排操作日志

时间：2026-11-13 CST
操作：按 AGENTS.md 文档规范，将 `docs/` 重排为 `docs/EXP.md`、`docs/RESULTS.md`、`docs/exp/<实验代号>/{exp.md,results.md}` 结构。
约束：仅做文件移动、改名与相对链接/路径字符串的机械替换，不改动任何实验设置、数值、表格、结论与适用边界。

## 1. 备份与提交

- 重排前完整备份：`archive/docs-reorg-backup/docs-docs_legacy-20261113.tar.gz`（实际位于数据盘 `/root/autodl-tmp/rage/archive/`，含 docs 与 docs_legacy 共 146 个条目）。
- 重排前快照提交：`596f114` `docs: snapshot docs and docs_legacy before experiment doc reorganization`，包含当时 docs 与 docs_legacy 的全部状态（145 个路径）。
- 重排脚本：`scripts/temp/reorg_docs_20261113.py`（不纳入 git）；机器可读移动清单与校验结果：`scripts/temp/reorg_docs_report.json`。

## 2. 移动规则（共 78 个文件，git 全部识别为 rename）

### 2.1 索引

- `docs/exp/EXP.md` -> `docs/EXP.md`
- `docs/RESULTS.md` 原地保留，仅改写链接。

### 2.2 实验设计文档（32 个）

`docs/exp/<代号>.md` -> `docs/exp/<代号>/exp.md`，覆盖全部 32 个实验代号（含 `v1-rage-trev`、`rage-smoke-static-cycle` 等）。

### 2.3 同名结果文档（26 个）

与实验代号精确同名的 `docs/results/<代号>.md` -> `docs/exp/<代号>/results.md`。

### 2.4 孤立结果独立文件夹（5 个，只有 results.md，无 exp.md）

| 原文件 | 新位置 |
|---|---|
| `docs/results/v0-crb-cos-var.md` | `docs/exp/v0-crb-cos-var/results.md` |
| `docs/results/v0-crb-reconst-fine.md` | `docs/exp/v0-crb-reconst-fine/results.md` |
| `docs/results/v0-chronostiny-timerxl-20261112.md` | `docs/exp/v0-chronostiny-timerxl-20261112/results.md` |
| `docs/results/v0-consolidated-summary-20261112.md` | `docs/exp/v0-consolidated-summary-20261112/results.md` |
| `docs/results/src-8192-source-coverage.md` | `docs/exp/src-8192-source-coverage/results.md` |

### 2.5 补充结果（11 个，保留原文件名）

| 原文件 | 新位置 |
|---|---|
| `v0-active-value-rank-diagnostics.md` | `docs/exp/v0-active-value-rank/` |
| `v0-crb-final-test-timerXL-64tr-spacing-q-dir.md` | `docs/exp/v0-crb-final-test-timerXL-64tr/` |
| `v0-timerXL-grad-4-lm-head-change-velocity.md` | `docs/exp/v0-timerXL-grad-4-lm-head/` |
| `v1-crb-redo-timerxl-baselines.md` | `docs/exp/v1-crb-redo/` |
| `v1-rage-trev-seed0-final.md` | `docs/exp/v1-rage-trev/` |
| `v1-rage-trev-timerxl-seed0.md` | `docs/exp/v1-rage-trev/` |
| `v1-rage-trev-timesfm-seed0.md` | `docs/exp/v1-rage-trev/` |
| `v0-timer-XL-grad-4-two-random-16-blocks.md` | `docs/exp/v0-timer-XL-grad-4-change-velocity/` |
| `v0-timerxl-grad64-pca.md` | `docs/exp/v0-timer-XL-grad-64-change-velocity/` |
| `64-large-var-k-error-composition.md` | `docs/exp/v0-crb-mech-test/` |
| `64-large-var-k-error-composition-full-tables.md` | `docs/exp/v0-crb-mech-test/` |

### 2.6 模板（3 个 dotfile，保留原名）

`.v0-timerxl-grad64-index-template.md`、`.v0-timerxl-grad64-log-template.md`、`.v0-timerxl-grad64-report-template.md` -> `docs/exp/v0-timer-XL-grad-64-change-velocity/`。

### 2.7 未移动项

- `docs/logs/` 全部文件冻结，一字未改（历史日志中的旧路径作为历史记录保留）。
- `docs/lora-support.md`、`docs/third_parties/` 保持原位。
- 空目录 `docs/results/` 已删除。
- `docs_legacy/` 在本次重排后整体删除（见提交 2），删除前内容已由备份与提交 1 完整保存。
- 删除导致 16 个 exp.md 中的历史原文链接 `../../../docs_legacy/EXP.md` 成为悬空链接（重排时路径改写正确，目标在删除前存在）。涉及：`rage-smoke-static-cycle`、`v0-active-value-rank`、`v0-crb-reconst-test`、`v0-crb-var`、`v0-directional-marginal-gain`、`v0-grad-mech-l2-norm`、`v0-grad-mech-test`、`v0-grad-traj-level-norm-gap`、`v0-gradient-proxy-cost`、`v0-gradient-proxy-test`、`v0-gradient-variance-decomposition`、`v0-proxy-dp-segmentation`、`v0-uni-data-evaluation`、`v0-uni-reconst-scale`、`v0-vage`、`v0-var-proxy-estimate`。目标可通过 `git show 596f114:docs_legacy/EXP.md` 或备份 tarball 恢复。

## 3. 链接与路径字符串改写规则

- Markdown 相对链接按旧位置解析后映射到新位置，重新计算相对路径，锚点保留；http(s)、outputs/、data/ 等非文档链接不动。
- 深度增加一个层级后，`../../docs_legacy/` 与 `../../archive/` 前缀机械补为 `../../../`。
- 仓库相对路径字符串 `docs/exp/...`、`docs/results/...` 按同一移动映射替换。
- 59 个文件被改写；git diff 核验全部增删行均为路径行，无任何正文、数值或表格行变化。
- 手工补改 1 处：`.v0-timerxl-grad64-index-template.md` 内 `](results/v0-timer-XL-grad-64-change-velocity.md)` 原为相对 docs 根的链接，移动后改为 `](results.md)`。

## 4. 未机械映射、保留原文的 5 处描述性路径

以下为占位符或目录级泛指，无法映射到具体文件，按确认意见保留原文，列为后续待办：

1. `docs/EXP.md` 第 7 行：`docs/results/<实验代号>.md` 占位符。
2. `docs/EXP.md` 第 9 行：`docs/results/...` 目录级说明（同句还提到旧 `results/` 目录）。
3. `docs/RESULTS.md` 第 6 行：`docs/results/<实验代号>.md` 占位符。
4. `docs/exp/v0-active-value-rank/exp.md` 第 7 行：`结果和局限见 docs/results/ 与 RESULTS.md`。
5. `docs/exp/v0-consolidated-summary-20261112/results.md` 第 34 行：`所有已登记的 v0 专题报告均保留在 docs/results/`。

## 5. 校验结果

- 文件总数：重排前后 docs 下均为 91 个文件，无丢失、无新增内容文件。
- 目录结构：37 个实验文件夹（32 个含 exp.md，31 个含 results.md，其中 5 个孤立文件夹仅含 results.md）。
- 链接校验：82 个非 logs Markdown 文件中的 208 个相对链接，在删除 docs_legacy 前全部解析到存在的目标，0 断链；删除 docs_legacy 后仅剩第 2.7 节记录的 16 个历史原文悬空链接，其余 192 个链接仍然有效。
- 旧形式残留：非 logs 文档中 `](results/`、`](../results/`、`](../exp/` 残留为 0。
- git 识别：78 个移动全部为 rename（19 个纯 rename，59 个 rename 加路径改写），`docs/RESULTS.md` 为原地修改。

## 6. 未解决问题与待办

- 16 个指向已删除 `docs_legacy/EXP.md` 的悬空历史链接（清单见 2.7）：需用户决定是改为指向 git 历史/备份，还是移除链接保留文字。
- 第 4 节列出的 5 处占位符/目录级描述需后续按新结构人工改写正文（本次不允许内容变更，未处理）。
- `docs/lora-support.md`、`docs/third_parties/` 在 AGENTS.md 规范树中没有对应位置，本次保持原位，待用户决定归属。
- 根目录 `TEMP.md` 含 26 处 `docs/results/...` 旧路径引用，不在本次 docs 重排范围内，未修改。
