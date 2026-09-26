#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/TS3
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ts3
source /etc/network_turbo
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=/root/TS3
export HF_HOME=/root/autodl-tmp/TS3/data/bench/fev

echo "FEV_CACHE_MATERIALIZE_START $(date -Is)"
python - <<'PY'
from pathlib import Path
import yaml
import fev

profile_path = Path("configs/eval/fev/profile-10b-local.yaml")
default_path = Path("configs/eval/fev/default.yaml")
cache_root = Path("/root/TS3/data/bench/fev/datasets/autogluon___fev_datasets")
tasks = yaml.safe_load(default_path.read_text(encoding="utf-8"))["tasks"]
lines = profile_path.read_text(encoding="utf-8").splitlines()
path_lines = [i for i, line in enumerate(lines) if line.startswith("- dataset_path:")]
if len(tasks) != 20 or len(path_lines) != 20:
    raise RuntimeError("expected 20 FEV tasks and dataset_path lines")
for index, task_data in enumerate(tasks):
    config = task_data["dataset_config"]
    print("materializing FEV task", index, config, flush=True)
    task = fev.Task(**task_data)
    task.load_full_dataset()
    task.get_window(0, num_proc=1).get_input_data()
    candidates = sorted(cache_root.joinpath(config, "0.0.0").glob("*/fev_datasets-train.arrow"))
    if len(candidates) != 1:
        raise RuntimeError("expected one local FEV cache file for %s, got %s" % (config, candidates))
    lines[path_lines[index]] = "- dataset_path: %s" % candidates[0]
profile_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("FEV local declaration updated", profile_path, flush=True)
PY
echo "FEV_CACHE_MATERIALIZE_COMPLETE $(date -Is)"

python -m assets.dataset_processing.fetch.ref --spec configs/ref_spec/fev/spec.yaml --output-dir data/bench/fev
echo "REFERENCE_FETCH_COMPLETE $(date -Is)"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY HF_ENDPOINT
python scripts/assets.py --config configs/scripts/assets.yaml
echo "PIPELINE_COMPLETE $(date -Is)"
