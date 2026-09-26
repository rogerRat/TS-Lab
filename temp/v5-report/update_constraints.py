"""Record current result-order and seed/task execution constraints."""
from datetime import datetime, timezone, timedelta
from pathlib import Path

section = '''
## 当前正式实验：结果排序与执行优先级

- 所有结果表必须按相对于 rand 的改善率从好到差降序排序，包括任务模型表、末尾汇总表和新增结果表。改善率为 100×(1−方法 MASE 均值/rand MASE 均值)，正值更好；rand 为 0%，orig 也使用相同排序键。没有结果或没有 rand 基准的行放末尾；同改善率稳定排列。禁止按绝对 MASE、方法名或完成时间替代。
- 正式实验优先级为种子 → 下游任务 → 该任务的全部方法 cell。当前种子的全部模型×任务×方法完成训练和评测后才进入下一种子；固定 seed0 → seed1 → seed2，不能跨种子抢跑。轮内两模型并行，每个模型按配置任务顺序先完成当前下游任务的全部方法，再推进下一任务；两模型独立按任务推进，整轮种子屏障覆盖两模型。已完成 cell 不重跑；orig 只测一次，后续种子引用。
- 当前结果文档为 docs/MAIN.md，巡检后执行 `/root/miniconda3/envs/ts3/bin/python temp/v5-report/main_report.py` 刷新。生成器必须保留以上约束，并核验所有任务模型表及末尾汇总表的降序。
'''
p = Path('AGENTS.md')
content = p.read_text()
if '## 当前正式实验：结果排序与执行优先级' not in content:
    with p.open('a') as handle:
        handle.write(section)
stamp = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
with Path('docs/Progress.md').open('a') as handle:
    handle.write(f'\n### {stamp} User-frozen result ordering and execution priority\n- Added binding constraints to AGENTS.md and the MAIN generator: every result table sorted descending improvement versus rand (including orig, missing baseline/results last), with global summary sort and mean/count checks. Execution priority is seed, then downstream task, then all its method cells within each parallel model worker. Audited seed_rounds.run_rounds whole-matrix barrier and full_pool_worker dataset outer loop plus train_dataset inner arm loop: current execution already conforms to this priority; models advance independently within the same seed. Existing upfront orig baseline evaluations retained, never repeated. No new global cross-model task barrier was introduced; seed barrier covers both models. No experiment restart, scientific protocol/training change or source-code modification. Refresh MAIN and verify tables next; experiment ongoing.\n')
print('Constraints recorded; scientific code and live workers unchanged.')
