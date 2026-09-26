#!/root/miniconda3/envs/ts3/bin/python
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path("/root/TS3")
fc = ROOT / "ts3/features/config.py"
source_input = Path("/root/autodl-tmp/TS3/data/src/raw/profiles/profile-10b")
bundle = Path("/root/autodl-tmp/TS3/data/src/shard_dataset/profile-10b-segl512s512")
catalog = ROOT / "configs/catalog/src/profile-10b-segl512s512.yaml"
original = fc.read_text()

def set_geometry(text):
    text, a = re.subn(r"ANALYSIS_SEGMENT_LENGTH: int = \d+", "ANALYSIS_SEGMENT_LENGTH: int = 512", text)
    text, b = re.subn(r"ANALYSIS_SEGMENT_STRIDE: int = \d+", "ANALYSIS_SEGMENT_STRIDE: int = 512", text)
    if a != 1 or b != 1:
        raise RuntimeError("geometry constants not found uniquely")
    return text

def run(argv, label):
    print("START", label, flush=True)
    result = subprocess.run(argv, cwd=ROOT)
    if result.returncode:
        raise SystemExit(f"{label} failed with exit {result.returncode}")
    print("DONE", label, flush=True)

try:
    fc.write_text(set_geometry(original))
    if not bundle.exists():
        run([sys.executable, "-m", "assets.dataset_processing.load",
             "--dataset", "all", "--subset", "all",
             "--input-dir", str(source_input), "--bundle-dir", str(bundle)], "source load")
    else:
        print("SKIP source load existing", bundle, flush=True)
    if not (bundle / "dedup").exists():
        run([sys.executable, "-m", "assets.dataset_processing.dedup.src",
             "--bundle", str(bundle)], "source dedup")
    else:
        print("SKIP source dedup existing", flush=True)
    if not (bundle / "feature").exists():
        run([sys.executable, "-m", "assets.dataset_processing.features.src",
             "--bundle", str(bundle)], "source features")
    else:
        print("SKIP source features existing", flush=True)
    catalog.parent.mkdir(parents=True, exist_ok=True)
    if not catalog.exists():
        run([sys.executable, "-m", "assets.dataset_processing.catalog",
             "--bundle", str(bundle), "--type", "source",
             "--output", str(catalog)], "source catalog")
    else:
        print("SKIP source catalog existing", catalog, flush=True)
finally:
    fc.write_text(original)
    print("RESTORED feature geometry", flush=True)
