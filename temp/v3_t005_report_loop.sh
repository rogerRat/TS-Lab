#!/bin/bash
cd /root/TS3
while true; do
PYTHONPATH=/root/TS3/src /root/miniconda3/envs/ts3/bin/python -m rage.experiments.v3_report --run /root/autodl-tmp/TS3/outputs/main-shared-nine-v3-t005-20260917
sleep 1800
done
