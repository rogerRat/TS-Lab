#!/root/miniconda3/envs/ts3/bin/python
import re
import sys
from pathlib import Path

sys.path.insert(0, "/root/TS3")
from ts3.spec.catalogs.discovery import resolve_long_reference_catalog_dir
import scripts.exp.main as runner

runner.resolve_long_reference_catalog_dir = resolve_long_reference_catalog_dir
config_arg = None
for i, value in enumerate(sys.argv):
    if value == "--config" and i + 1 < len(sys.argv):
        config_arg = sys.argv[i + 1]
        break
match = re.search(r"anaseg-l(\d+)s(\d+)", str(config_arg or ""))
original = None
feature_config = Path("/root/TS3/ts3/features/config.py")
if match:
    length, stride = int(match.group(1)), int(match.group(2))
    if length != stride:
        raise SystemExit("AnaSeg task adapter requires lNsN config")
    original = feature_config.read_text()
    text, count_length = re.subn(
        r"ANALYSIS_SEGMENT_LENGTH: int = \d+",
        "ANALYSIS_SEGMENT_LENGTH: int = %d" % length,
        original,
    )
    text, count_stride = re.subn(
        r"ANALYSIS_SEGMENT_STRIDE: int = \d+",
        "ANALYSIS_SEGMENT_STRIDE: int = %d" % stride,
        text,
    )
    if count_length != 1 or count_stride != 1:
        raise SystemExit("AnaSeg geometry constants were not found uniquely")
    feature_config.write_text(text)
try:
    raise SystemExit(runner.main())
finally:
    if original is not None:
        feature_config.write_text(original)
