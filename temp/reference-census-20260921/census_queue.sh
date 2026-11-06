#!/bin/bash
# reference-census-20260921 continuation queue (handoff 2026-09-21).
# Launch conditions: Timer-XL formal run complete AND GPU fully released AND disk budget OK.
# Stages: A references (timerxl -> chronos+timesfm parallel), B alignment, C associations.
# Idempotent: reference_census compute resumes missing windows; stages skip on sentinels.
set -uo pipefail
cd /root/TS3

TIMER=/root/autodl-tmp/TS3/outputs/main-rank-volume-timerxl-20260920
MAIN=/root/autodl-tmp/TS3/outputs/main-rank-volume-20260918
UNI=/root/autodl-tmp/TS3/outputs/main-rank-volume-uni-20260920
A=/root/autodl-tmp/TS3/outputs/reference-census-20260921
LOG=$A/queue.log
mkdir -p "$A"

say(){ echo "[census-queue] $(date -Is) $*" | tee -a "$LOG"; }

# 1. Wait for Timer-XL suite completion (seed rounds write state.json status=complete).
say "waiting for timerxl suite completion"
while true; do
  s=$(/root/miniconda3/envs/ts3/bin/python -c "import json;print(json.load(open('$TIMER/state.json')).get('status',''))" 2>/dev/null || echo missing)
  [ "$s" = complete ] && break
  sleep 120
done
say "timerxl complete; waiting for GPU release"
while true; do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  [ "$apps" -eq 0 ] && break
  sleep 60
done
sleep 180

# 2. Disk budget: total BF16 payload 113.94GiB + index/alignment/logs; require 130GiB free.
free_gib=$(df -BG /root/autodl-tmp | awk 'NR==2{gsub("G","",$4); print $4}')
say "disk free ${free_gib}GiB"
if [ "$free_gib" -lt 130 ]; then
  say "DISK_BUDGET_INSUFFICIENT (${free_gib}GiB < 130GiB); not starting. Report to operator."
  exit 1
fi

PY_T=/root/autodl-tmp/ts3-integration/envs/ts3-timerxl/bin/python
PY_C=/root/miniconda3/envs/ts3/bin/python
PY_F=/root/autodl-tmp/conda/envs/ts3-timesfm/bin/python

# 3. Stage A: timerxl first (fastest; validates pipeline + real throughput), then
#    chronos and timesfm concurrently (distinct envs, distinct per-model caches).
if [ ! -f "$A/timerxl-84m/stage_a_complete.marker" ]; then
  say "stage A references timerxl-84m"
  "$PY_T" -m rage.experiments.reference_census compute --run "$TIMER" --student timerxl-84m \
    --out "$A" --stage references 2>&1 | tee -a "$A/timerxl-references.log" || { say "STAGE_A_TIMERXL_FAILED"; exit 1; }
  touch "$A/timerxl-84m/stage_a_complete.marker" 2>/dev/null || { mkdir -p "$A/timerxl-84m"; touch "$A/timerxl-84m/stage_a_complete.marker"; }
fi

if [ ! -f "$A/chronos-base/stage_a_complete.marker" ] || [ ! -f "$A/timesfm-2.5/stage_a_complete.marker" ]; then
  say "stage A references chronos-base + timesfm-2.5 (parallel)"
  ( "$PY_C" -m rage.experiments.reference_census compute --run "$MAIN" --student chronos-base \
      --out "$A" --stage references 2>&1 | tee -a "$A/chronos-references.log" \
      && touch "$A/chronos-base/stage_a_complete.marker" ) &
  P1=$!
  ( "$PY_F" -m rage.experiments.reference_census compute --run "$MAIN" --student timesfm-2.5 \
      --out "$A" --stage references 2>&1 | tee -a "$A/timesfm-references.log" \
      && touch "$A/timesfm-2.5/stage_a_complete.marker" ) &
  P2=$!
  rc=0; wait $P1 || rc=1; wait $P2 || rc=1
  [ $rc -ne 0 ] && { say "STAGE_A_PARALLEL_FAILED"; exit 1; }
fi
say "stage A complete"

# 4. Stage B: alignment. Same model serial across runs (single-writer cache);
#    different models parallel.
if [ ! -f "$A/.stage_b_complete" ]; then
  say "stage B alignment (chronos+timesfm serial MAIN->UNI, timerxl parallel)"
  ( "$PY_C" -m rage.experiments.reference_census compute --run "$MAIN" --student chronos-base --out "$A" --stage alignment 2>&1 | tee -a "$A/chronos-alignment-main.log" \
    && "$PY_C" -m rage.experiments.reference_census compute --run "$UNI" --student chronos-base --out "$A" --stage alignment 2>&1 | tee -a "$A/chronos-alignment-uni.log" ) &
  P1=$!
  ( "$PY_F" -m rage.experiments.reference_census compute --run "$MAIN" --student timesfm-2.5 --out "$A" --stage alignment 2>&1 | tee -a "$A/timesfm-alignment-main.log" \
    && "$PY_F" -m rage.experiments.reference_census compute --run "$UNI" --student timesfm-2.5 --out "$A" --stage alignment 2>&1 | tee -a "$A/timesfm-alignment-uni.log" ) &
  P2=$!
  ( "$PY_T" -m rage.experiments.reference_census compute --run "$TIMER" --student timerxl-84m --out "$A" --stage alignment 2>&1 | tee -a "$A/timerxl-alignment.log" ) &
  P3=$!
  rc=0; wait $P1 || rc=1; wait $P2 || rc=1; wait $P3 || rc=1
  [ $rc -ne 0 ] && { say "STAGE_B_FAILED"; exit 1; }
  touch "$A/.stage_b_complete"
fi
say "stage B complete"

# 5. Stage C: CPU associations + report.
if [ ! -f "$A/report/REPORT.md" ]; then
  say "stage C associations"
  OPENBLAS_NUM_THREADS=4 "$PY_C" -m rage.experiments.reference_associations \
    --runs "$MAIN" "$UNI" "$TIMER" --census "$A" --out "$A/report" 2>&1 | tee -a "$A/associations.log" || {
    say "STAGE_C_FAILED"; exit 1; }
fi
say "CENSUS_ALL_COMPLETE"
