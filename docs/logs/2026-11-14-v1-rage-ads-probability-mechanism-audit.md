# v1-rage-ads 概率支持机制复核

- 检查时间：2026-11-14 14:25 CST。
- 实验代号 / 批次：`v1-rage-ads` / `rolling-eight-task`。
- 核对文件：`scripts/exp/v1-rage-ads/run_selection.py`、`scripts/exp/v1-rage-ads/train_eval.py`、八任务各臂 `selection.json` / `sampling.json` / ADS `probability.json`，及 `configs/data/src-96.yaml`。
- ADS 配置核实：RAGE 与 ancUni 每任务共用 q=16、eta=1 及相同粗锚点；RAGE 为 R=1，ancUni 为 R=0。因此两者不是总梯度计算量匹配的对照。八任务粗锚点合计 source 666,640、reference 8,700，均复用热身缓存。RAGE 额外一轮细化名义目标为 source 607,690、reference 8,595；实际新计算 source 591,351、reference 8,595。ancUni 没有细化梯度计算。最终 descriptor source 锚点位置合计 RAGE 1,277,117（正分锚点 1,247,034）、ancUni 666,640（正分锚点 641,903）。
- Traj 规则核实：按全局 top-ρ 轨迹方向配对选出 source 轨迹，然后均匀抽选中轨迹并在轨迹内部均匀抽合法窗口。未选轨迹的窗口概率为零；选中轨迹的窗口概率均为正。选择轨迹条数与窗口支持比例不同，后者按各轨迹合法窗口数加权。
- Traj 实际支持：TIME 2 为 80/96 条轨迹，窗口零概率 4.91%；TIME 8 为 20/96 条轨迹，窗口零概率 94.69%。八任务累计选中 406/768 个“任务×轨迹”位置，正支持 9,849,565/21,330,496 个窗口起点。
- 结果解释和对照口径已补入 `docs/exp/v1-rage-ads/results.md` 与 `docs/exp/v1-rage-ads/exp.md`；原始概率支持审计产物未改写。
- 任务状态：实验完成；本次为审计解释与文档补充，无训练、选择或评估重跑。
