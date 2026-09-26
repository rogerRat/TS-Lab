#!/bin/bash
end=$(($(date +%s)+7200))
while [ $(date +%s) -lt $end ]; do
  echo "$(date '+%H:%M:%S') $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr ',' ' ') | $(ps -eo rss,args | grep full_pool_worker | grep -v grep | awk '{printf "%s %.1fGB ", $NF, $1/1048576}')" >> /root/TS3/temp/proc_watch.log
  sleep 30
done
