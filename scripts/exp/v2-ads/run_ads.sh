#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if ! command -v conda >/dev/null 2>&1; then
    source /root/miniconda3/etc/profile.d/conda.sh
fi
conda activate rage
cd "$ROOT"
exec python -u -m rage.experiments.run --config configs/rage/ads.yaml "$@"
