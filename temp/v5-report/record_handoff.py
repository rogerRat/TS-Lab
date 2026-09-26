"""Record user-directed ownership change, without changing experiment state."""
from datetime import datetime,timezone,timedelta
from pathlib import Path
import json

stamp=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
section='''## 2026-09-18 巡检责任交接：WSL tmux moni

用户明确要求本机停止巡检，由 SSH `wsl` 主机上的 `tmux moni` 中 Codex 接管巡检与总结。模型固定为 `gpt-5.6-luna`，推理强度 `high`。本机 heartbeat `v5` 已暂停，不重新启用。WSL 工作目录 `/home/rogers/temp/v5-monitor`；完整交接材料 `HANDOFF.md`，巡检记录 `monitor.log`，最新总结 `latest-summary.md`，接管确认 `ACK.json`，当前状态 `STATUS.json` / `monitor.json`。同一 Codex 任务每十分钟续跑，全部 666 独立 cell 和三轮 seed 完成且报告核验后停止。

此交接只改变监督责任，不停止或重启实验。冻结科学版本、种子整轮屏障、轮内任务优先级、两模型并行、全部结果表按相对 rand 改善降序及产物保留边界不变。已有原生 checkpoint 保留协议不因交接改变。WSL 负责人每次巡检更新 MAIN 和 Progress，故障先定位，在既有授权内恢复并避免重复 cell，必要时报告用户。当前实验仍在运行，首轮实时检查及确认由接管方负责。
'''
main=Path('docs/MAIN.md')
content=main.read_text()
marker='<!-- v5-notes:end -->'
assert marker in content
content=content.replace(marker,section+'\n'+marker,1)
main.write_text(content)
with Path('AGENTS.md').open('a') as handle:
    handle.write('\n## 当前巡检负责人（用户最新决定）\n\n巡检与总结已移交 SSH `wsl` 主机的 `tmux moni`，由 Codex `gpt-5.6-luna` / `high` 持续负责。本机 Codex heartbeat `v5` 已暂停。完整交接见 WSL `/home/rogers/temp/v5-monitor/HANDOFF.md` 和 MAIN 文末责任交接。不得依据此前本机本人监督的记录重启本机巡检；实验容器 supervisor 保持运行。\n')
with Path('docs/Progress.md').open('a') as handle:
    handle.write(f'\n### {stamp} User-directed V5 monitoring handoff to WSL\n- Paused local heartbeat v5. Confirmed WSL Codex CLI0.154.0, cached supported model gpt-5.6-luna/high, idle tmux moni and direct SSH to experiment container. Launched single monitoring wrapper PID17013 and Codex PID17015 in moni, session 01a0b4e7-1228-75c1-a564-b4a3784688e3. Complete HANDOFF.md delivered, with frozen science/execution/report/cache/recovery/retention/test/partial-result history. First patrol and ACK pending at this record. Updated MAIN/AGENTS ownership; same Codex task resumed every10min via wrapper, supervision ends after full completion. Scientific6d3ead6 and experiment supervisor148026 unchanged, no experiment process stopped/restarted, no cell repeated or manual deletion. Next: confirm recipient first live patrol and ACK before ending transfer.\n')
print('Monitoring ownership recorded; experiment unchanged.')
