#!/usr/bin/env bash
set -euo pipefail
cd /root/TS-Lab
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
/root/autodl-tmp/rage/envs/rage-timerxl/bin/pip install peft==0.10.0 datasets==4.8.5 gluonts==0.15.1 python-dotenv==1.0.0 pydantic==2.13.4 scipy==1.11.4 numpy==1.26.4
/root/autodl-tmp/rage/envs/rage-timerxl/bin/pip install --no-deps -e third_parties/fev
/root/autodl-tmp/rage/envs/rage-timesfm/bin/pip install peft==0.18.1 datasets==4.8.5 gluonts==0.15.1 python-dotenv==1.0.0 pydantic==2.13.4 scipy==1.11.4 numpy==1.26.4
/root/autodl-tmp/rage/envs/rage-timesfm/bin/pip install --no-deps -e third_parties/fev
source /etc/network_turbo
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_HUB_CACHE=/root/autodl-tmp/hf-cache/hub
/root/autodl-tmp/rage/envs/rage/bin/python -u scripts/temp/download_v1_trev_foundation_models.py
