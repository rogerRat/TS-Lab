#!/root/miniconda3/envs/ts3/bin/python
from pathlib import Path
import re
import signal
import subprocess
import sys
import yaml

ROOT = Path("/root/TS3")
feature_config = ROOT / "ts3/features/config.py"
assets_config = ROOT / "configs/scripts/assets.yaml"
temp_config = ROOT / "temp/assets-anaseg-no-mechanism.yaml"
original = feature_config.read_text()
config = yaml.safe_load(assets_config.read_text())
config["mechanism_analysis"] = {"enabled": False}
temp_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="ascii")

def set_geometry(length, stride):
    text = feature_config.read_text()
    text, count_length = re.subn(r"ANALYSIS_SEGMENT_LENGTH: int = \d+", "ANALYSIS_SEGMENT_LENGTH: int = %d" % length, text)
    text, count_stride = re.subn(r"ANALYSIS_SEGMENT_STRIDE: int = \d+", "ANALYSIS_SEGMENT_STRIDE: int = %d" % stride, text)
    if count_length != 1 or count_stride != 1:
        raise RuntimeError("feature geometry constants were not found uniquely")
    feature_config.write_text(text)

try:
    for label, length, stride in (("l256s256", 256, 256), ("l512s512", 512, 512)):
        run_id = "profile-10b-seg%s" % label
        print("START ASSETS", label, run_id, flush=True)
        set_geometry(length, stride)
        result = subprocess.run(
            [sys.executable, "scripts/assets.py", "--config", str(temp_config), "--run-id", run_id],
            cwd=ROOT,
        )
        if result.returncode != 0:
            raise SystemExit(result.returncode)
        print("DONE ASSETS", label, run_id, flush=True)
finally:
    feature_config.write_text(original)
    print("RESTORED feature geometry", flush=True)
