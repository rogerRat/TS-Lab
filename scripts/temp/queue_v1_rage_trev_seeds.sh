#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/TS-Lab
export PYTHONUNBUFFERED=1
export PYTHONPATH=src:third_parties/fev/src:third_parties/TIME/src:.
log() { printf "%s-[queue_v1_rage_trev_seeds]-[queue]-[%s]\n" "$(date -Is)" "$*"; }
chronos=v1-rage-chronosT5-trev-base
timer=v1-rage-timerXL-trev-84m
times=v1-rage-timesFM-trev-2.5
run_foundation() {
  local python="$1" prefix="$2" suffix="$3"
  "$python" -u scripts/temp/v1_rage_foundation_trev.py \
    --rand-config "configs/rage/${prefix}-rand${suffix}.json" \
    --full-config "configs/rage/${prefix}-full${suffix}.json"
}
log "resume seed 0 TimerXL and TimesFM"
run_foundation /root/autodl-tmp/rage/envs/rage-timerxl/bin/python "$timer" ""
run_foundation /root/autodl-tmp/rage/envs/rage-timesfm/bin/python "$times" ""
for seed in 1 2 3; do
  suffix="-seed${seed}"
  log "prepare and run Chronos seed=$seed"
  /root/autodl-tmp/rage/envs/rage/bin/python -u scripts/temp/v1_rage_chronost5_trev.py \
    --random-config "configs/rage/${chronos}-rand${suffix}.json" \
    --full-config "configs/rage/${chronos}-full${suffix}.json" --prepare-only
  tmux new-session -d -s "v1-trev-cleanup-seed${seed}" \
    "cd /root/TS-Lab && PYTHONPATH=src:. PYTHONUNBUFFERED=1 /root/autodl-tmp/rage/envs/rage/bin/python -u scripts/temp/v1_rage_trev_checkpoint_cleanup.py --seed $seed"
  /root/autodl-tmp/rage/envs/rage/bin/python -u scripts/temp/v1_rage_chronost5_trev.py \
    --random-config "configs/rage/${chronos}-rand${suffix}.json" \
    --full-config "configs/rage/${chronos}-full${suffix}.json"
  run_foundation /root/autodl-tmp/rage/envs/rage-timerxl/bin/python "$timer" "$suffix"
  run_foundation /root/autodl-tmp/rage/envs/rage-timesfm/bin/python "$times" "$suffix"
done
touch outputs/train/v1-rage-trev-seeds-0-3.complete
log "all seeds 0 through 3 complete"
