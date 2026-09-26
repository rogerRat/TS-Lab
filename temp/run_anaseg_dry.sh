#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/TS3
for c in configs/scripts/run.chronos46m.anaseg-l256s256.yaml configs/scripts/run.chronos46m.anaseg-l512s512.yaml configs/scripts/run.chronos46m.anaseg-l1024s1024.yaml; do
  n=$(basename "$c" .yaml)
  echo "START $n"
  OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 /root/miniconda3/envs/ts3/bin/python temp/run_anaseg_main.py --config "$c" --seed 0 --dry-run > "temp/dry-$n.log" 2>&1
  tail -n 5 "temp/dry-$n.log"
done
echo ALL_DRYRUN_OK
