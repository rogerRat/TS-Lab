#!/bin/bash
# Timer-XL continuation queue (PLAN.md): starts only after the v5-uni suite is complete
# AND the GPU is fully released. Stages: smoke -> calibration stage 1 (devhistory, seed 0)
# -> stage-2 config from pre-registered top-2 rule -> stage 2 (earlierhistory, seed 1)
# -> frozen-recipe proposal. Each stage is resumable: full_pool_exploration skips
# completed trials, and this script skips stages whose completion marker exists.
set -uo pipefail
cd /root/TS3  # run_exploration records provenance via git rev-parse HEAD

export HF_HUB_CACHE=/root/autodl-tmp/hf-integration/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PY=/root/autodl-tmp/ts3-integration/envs/ts3-timerxl/bin/python
SUITE=/root/autodl-tmp/TS3/outputs/main-rank-volume-uni-20260920
WORK=/root/autodl-tmp/TS3/outputs/timerxl-calibration-20260920
PLAN=/root/TS3/temp/timerxl-plan-20260920
mkdir -p "$WORK"

echo "[queue] waiting for v5-uni suite completion"
while true; do
  status=$("$PY" -c "import json;print(json.load(open('$SUITE/state.json')).get('status',''))" 2>/dev/null || echo missing)
  if [ "$status" = complete ]; then break; fi
  sleep 120
done
echo "[queue] suite complete; waiting for GPU release"
while true; do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  if [ "$apps" -eq 0 ]; then break; fi
  sleep 60
done
sleep 180  # settle: workers flush and exit after the barrier
echo "[queue] GPU free at $(date -Is)"

if [ ! -f "$WORK/smoke/smoke.json" ]; then
  echo "[queue] stage SMOKE"
  "$PY" "$PLAN/timerxl_smoke.py" --out "$WORK/smoke" 2>&1 | tee "$WORK/smoke.log" || {
    echo "[queue] SMOKE_FAILED"; exit 1; }
fi

if [ ! -f "$WORK/stage1-devhistory/results.json" ]; then
  echo "[queue] stage CALIBRATION-1 devhistory seed0"
  "$PY" -m rags.experiments.full_pool_exploration \
    --config /root/TS3/configs/rags/development_timerxl_history_dev_0920.json \
    --out "$WORK/stage1-devhistory" 2>&1 | tee "$WORK/stage1.log" || {
    echo "[queue] STAGE1_FAILED"; exit 1; }
fi

if [ ! -f "$WORK/stage1-devhistory/stage2_selection.json" ]; then
  "$PY" "$PLAN/make_stage2_config.py" --stage1 "$WORK/stage1-devhistory" || {
    echo "[queue] STAGE2_CONFIG_FAILED"; exit 1; }
fi

if [ ! -f "$WORK/stage2-earlierhistory/results.json" ]; then
  echo "[queue] stage CALIBRATION-2 earlierhistory seed1"
  "$PY" -m rags.experiments.full_pool_exploration \
    --config /root/TS3/configs/rags/development_timerxl_history_early_0920.json \
    --out "$WORK/stage2-earlierhistory" 2>&1 | tee "$WORK/stage2.log" || {
    echo "[queue] STAGE2_FAILED"; exit 1; }
fi

"$PY" "$PLAN/propose_recipe.py" --stage1 "$WORK/stage1-devhistory" \
  --stage2 "$WORK/stage2-earlierhistory" --out "$WORK/recipe_proposal.json" || {
  echo "[queue] RECIPE_PROPOSAL_FAILED"; exit 1; }

echo "[queue] TIMERXL_CALIBRATION_COMPLETE $(date -Is)"
