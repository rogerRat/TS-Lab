#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/TS-Lab
log() { printf "%s-[queue_v1_rage_foundation_trev]-[queue]-[%s]\n" "$(date -Is)" "$*"; }
wait_for() {
  local marker="$1"
  while [[ ! -f "$marker" ]]; do
    log "waiting marker=$marker"
    sleep 60
  done
  log "dependency complete marker=$marker"
}
wait_for outputs/train/v1-rage-chronosT5-trev-base-rand/.complete
wait_for outputs/train/v1-rage-chronosT5-trev-base-full/.complete
log "starting TimerXL rand then full"
PYTHONPATH=src:third_parties/fev/src:third_parties/TIME/src:. PYTHONUNBUFFERED=1 /root/autodl-tmp/rage/envs/rage-timerxl/bin/python -u scripts/temp/v1_rage_foundation_trev.py --rand-config configs/rage/v1-rage-timerXL-trev-84m-rand.json --full-config configs/rage/v1-rage-timerXL-trev-84m-full.json
log "TimerXL batch pair complete; starting TimesFM rand then full"
PYTHONPATH=src:third_parties/fev/src:third_parties/TIME/src:. PYTHONUNBUFFERED=1 /root/autodl-tmp/rage/envs/rage-timesfm/bin/python -u scripts/temp/v1_rage_foundation_trev.py --rand-config configs/rage/v1-rage-timesFM-trev-2.5-rand.json --full-config configs/rage/v1-rage-timesFM-trev-2.5-full.json
log "all foundation Trev batches complete"
touch outputs/train/v1-rage-foundation-trev-queue.complete
