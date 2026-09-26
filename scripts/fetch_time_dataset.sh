#!/usr/bin/env bash
# @input: Optional CONFIG_PATH selecting smoke or official_all task validation.
# @output: Pinned TIME dataset tree and data_manifest.json under data/bench/time.
# @scenario: Download only the declared Hugging Face revision for TIME evaluation.
# @logic: Refuses unmatched existing data, checks free space before downloading,
#          and validates every selected dataset directory with load_from_disk.

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PYTHON:-}" ]]; then
    if ! command -v conda >/dev/null && [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    fi
    command -v conda >/dev/null || { echo 'Run bash _install.sh --env time first, or set PYTHON.' >&2; exit 1; }
    PYTHON="$(conda run -n time python -c 'import sys; print(sys.executable)')"
fi
CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/eval/time/official_all.yaml}"
DATA_ROOT="$("${PYTHON}" - "${CONFIG_PATH}" "${ROOT_DIR}" <<'PY'
import sys
from pathlib import Path
import yaml
value = Path(yaml.safe_load(Path(sys.argv[1]).read_text())["bench_config"]["dataset"]["root"])
print((value if value.is_absolute() else Path(sys.argv[2]) / value).resolve())
PY
)"
MANIFEST_PATH="$(dirname "${DATA_ROOT}")/data_manifest.json"
LOG_DIR="${ROOT_DIR}/logs"
TIMESTAMP="$(TZ=Asia/Shanghai date +%Y%m%d%H%M%S)"
LOG_FILE="${LOG_DIR}/fetch_time_dataset_${TIMESTAMP}.log"
REPO_ID="Real-TSF/TIME"
REVISION="83e3d0b3be28d11c7182bffcc1892d19b36c4da1"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1
[[ -x "${PYTHON}" ]] || { echo '[Fetch] [ERROR] time environment Python is missing' >&2; exit 1; }
[[ -f "${CONFIG_PATH}" ]] || { echo '[Fetch] [ERROR] config is missing' >&2; exit 1; }
if [[ -f /etc/network_turbo ]]; then
    source /etc/network_turbo
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

disk_parent="${DATA_ROOT}"
while [[ ! -e "${disk_parent}" ]]; do disk_parent="$(dirname "${disk_parent}")"; done
free_kib="$(df -Pk "${disk_parent}" | awk 'NR == 2 {print $4}')"
echo "[Fetch] [INFO] target=${DATA_ROOT} free_kib=${free_kib} repo=${REPO_ID} revision=${REVISION}"
if [[ -e "${DATA_ROOT}" || -e "${MANIFEST_PATH}" ]]; then
    [[ -f "${MANIFEST_PATH}" ]] || { echo '[Fetch] [ERROR] Existing TIME data has no manifest' >&2; exit 1; }
    "${PYTHON}" - "${MANIFEST_PATH}" "${CONFIG_PATH}" "${ROOT_DIR}" "${REPO_ID}" "${REVISION}" <<'PY'
import json
import sys
from pathlib import Path
import yaml
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="ascii"))
config = yaml.safe_load(Path(sys.argv[2]).read_text(encoding="ascii"))
root = Path(sys.argv[3])
source = yaml.safe_load((root / config["bench_config"]["source"]["datasets_yaml"]).read_text(encoding="ascii"))["datasets"]
selection = config["bench_config"]["tasks"]
if selection["mode"] == "official_all":
    expected = [{"dataset": dataset, "term": term} for dataset, values in source.items() for term, item in values.items() if isinstance(item, dict) and "prediction_length" in item]
else:
    expected = [{"dataset": item["dataset"], "term": term} for item in selection["items"] for term in item["terms"]]
if manifest.get("repo_id") != sys.argv[4] or manifest.get("revision") != sys.argv[5] or manifest.get("license") != "CC BY-NC 4.0" or manifest.get("verified_tasks") != expected:
    raise SystemExit("Existing TIME manifest does not match pinned protocol")
PY
    echo "[Fetch] [SUCCESS] Reusing manifest-matched TIME data"
    exit 0
fi

"${PYTHON}" - "${ROOT_DIR}" "${CONFIG_PATH}" "${DATA_ROOT}" "${MANIFEST_PATH}" "${REPO_ID}" "${REVISION}" "${free_kib}" <<'PY'
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import datasets
import yaml
from huggingface_hub import HfApi, snapshot_download

project_root = Path(sys.argv[1])
config_path = Path(sys.argv[2])
data_root = Path(sys.argv[3])
manifest_path = Path(sys.argv[4])
repo_id = sys.argv[5]
revision = sys.argv[6]
free_kib = int(sys.argv[7])
config = yaml.safe_load(config_path.read_text(encoding="ascii"))
bench = config["bench_config"]
declared_root = Path(bench["dataset"]["root"])
declared_root = (declared_root if declared_root.is_absolute() else project_root / declared_root).resolve()
if bench["dataset"]["repo_id"] != repo_id or bench["dataset"]["revision"] != revision or declared_root != data_root:
    raise SystemExit("Config dataset block does not match the fixed TIME protocol")
source_yaml = project_root / bench["source"]["datasets_yaml"]
source = yaml.safe_load(source_yaml.read_text(encoding="ascii"))["datasets"]
tasks_value = bench["tasks"]
if tasks_value["mode"] == "official_all":
    if set(tasks_value) != {"mode"}:
        raise SystemExit("official_all must not declare items")
    tasks = [(dataset, term) for dataset, values in source.items() for term, item in values.items()
             if isinstance(item, dict) and "prediction_length" in item]
elif tasks_value["mode"] == "explicit":
    tasks = [(item["dataset"], term) for item in tasks_value["items"] for term in item["terms"]]
else:
    raise SystemExit("Unsupported TIME task mode")
for dataset, term in tasks:
    if dataset not in source or term not in source[dataset] or "prediction_length" not in source[dataset][term]:
        raise SystemExit("Config declares an unavailable TIME task: %s/%s" % (dataset, term))

api = HfApi()
info = api.repo_info(repo_id=repo_id, repo_type="dataset", revision=revision, files_metadata=True)
estimated_bytes = sum(int(item.size or 0) for item in info.siblings)
required_kib = int(estimated_bytes * 1.1 / 1024)
print("[Fetch] [INFO] remote_estimated_bytes=%d required_kib_with_margin=%d selected_tasks=%d" % (estimated_bytes, required_kib, len(tasks)))
if free_kib < required_kib:
    raise SystemExit("Insufficient disk space before TIME download")
data_root.parent.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision, local_dir=str(data_root))
verified = []
for dataset, term in tasks:
    dataset_path = data_root / dataset
    loaded = datasets.load_from_disk(str(dataset_path))
    if len(loaded) == 0:
        raise ValueError("TIME dataset is empty: %s" % dataset_path)
    verified.append({"dataset": dataset, "term": term})
manifest = {
    "repo_id": repo_id,
    "revision": revision,
    "license": "CC BY-NC 4.0",
    "downloaded_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    "tools": {"datasets": datasets.__version__},
    "verified_tasks": verified,
}
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii")
print("[Fetch] [SUCCESS] wrote manifest=%s verified_tasks=%d" % (manifest_path, len(verified)))
PY
