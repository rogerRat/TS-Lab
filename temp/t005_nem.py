import json
from pathlib import Path
run = Path("/root/autodl-tmp/TS3/outputs/main-shared-nine-v3-t005-20260917")
for s in ["chronos-base", "timesfm-2.5", "timemoe-200m"]:
    rows = []
    for c in sorted((run/"students"/s/"cells").glob("time_nem_demand_5t__*.json")):
        d = json.loads(c.read_text())
        rows.append((d.get("arm"), d.get("mase")))
    if rows:
        print(s, " | ".join(f"{a}={m:.4f}" if m else f"{a}=pending" for a, m in rows))
