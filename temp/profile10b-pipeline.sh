#!/usr/bin/env bash
set -Eeuo pipefail

cd /root/TS3
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ts3

echo "PIPELINE_START $(date -Is)"
source /etc/network_turbo
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=/root/TS3

python -m assets.dataset_processing.fetch \
  --profile profile-10b \
  --output-dir data/src/raw/profiles/profile-10b \
  --num-workers 8

echo "PROFILE_FETCH_COMPLETE $(date -Is)"
python -m assets.dataset_processing.fetch.ref \
  --spec configs/ref_spec/fev/spec.yaml \
  --output-dir data/bench/fev

echo "REFERENCE_FETCH_COMPLETE $(date -Is)"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY HF_ENDPOINT
python scripts/assets.py --config configs/scripts/assets.yaml
echo "PIPELINE_COMPLETE $(date -Is)"
