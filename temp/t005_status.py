import sys, json
from pathlib import Path
run = Path("/root/autodl-tmp/TS3/outputs/main-shared-nine-v3-t005-20260917")
total = 0
for s in ["chronos-base", "timesfm-2.5", "timemoe-200m"]:
    cells = list((run/"students"/s/"cells").glob("*.json"))
    n = sum(1 for c in cells if json.loads(c.read_text()).get("experiment_version") == "v3")
    st = run/"students"/s/"state.json"
    d = json.loads(st.read_text()) if st.exists() else {}
    print(f"{s}={n} cells | {d.get('stage')}/{d.get('dataset')}/{d.get('arm','')}")
    total += n
print(f"total={total}/189")
