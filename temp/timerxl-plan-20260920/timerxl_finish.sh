#!/bin/bash
# One-shot: diagnose timerxl seed2 worker tail; if all 387 cells verified, restart supervisor to finish the suite.
R=/root/autodl-tmp/TS3/outputs/main-rank-volume-timerxl-20260920
echo "=== worker log tail ==="
tail -n 15 $R/rounds/seed2/timerxl-84m.log
echo "=== supervisor log tail ==="
tail -n 5 $R/supervisor.log
echo "=== cell count ==="
ls $R/rounds/seed0/students/timerxl-84m/cells/*.json $R/rounds/seed1/students/timerxl-84m/cells/*.json $R/rounds/seed2/students/timerxl-84m/cells/*.json 2>/dev/null | wc -l
echo "=== restart supervisor (idempotent: round_completion validates existing cells) ==="
tmux kill-session -t timerxl-run 2>/dev/null
tmux new-session -d -s timerxl-run "export PATH=/root/miniconda3/bin:\$PATH && cd /root/TS3 && /root/miniconda3/envs/ts3/bin/python temp/timerxl-plan-20260920/supervisor.py 2>&1 | tee -a $R/supervisor.log"
sleep 90
echo "=== after restart ==="
cat $R/state.json
cat $R/supervisor.json
tmux ls | grep timerxl-run || echo "timerxl-run session exited (check state)"
