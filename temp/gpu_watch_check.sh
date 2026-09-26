#!/bin/bash
# Kimi GPU watch: lightweight 60s check. Prints ALERT lines + details, or OK.
R=/root/autodl-tmp/TS3/outputs/main-v7-20260923
STATE=/root/TS3/temp/gpu_watch_state
NOW=$(date +%s)
UTILS=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits)
U0=$(echo "$UTILS" | sed -n 1p | awk -F', ' '{print $2}'); M0=$(echo "$UTILS" | sed -n 1p | awk -F', ' '{print $3}'); P0=$(echo "$UTILS" | sed -n 1p | awk -F', ' '{print $4}')
U1=$(echo "$UTILS" | sed -n 2p | awk -F', ' '{print $2}'); M1=$(echo "$UTILS" | sed -n 2p | awk -F', ' '{print $3}'); P1=$(echo "$UTILS" | sed -n 2p | awk -F', ' '{print $4}')
# real workers (exclude tmux wrappers)
NW=$(ps -eo args | grep -E "run_v7|extension_queue|cell_queue|v7_queue" | grep -v grep | grep -cv tmux)
# queue state
QPOS=""; QSTATE=""
[ -f $R/extension_queue_status.json ] && QPOS=$(grep -o '"position": [0-9]*' $R/extension_queue_status.json | awk '{print $2}') && QSTATE=$(grep -o '"state": "[a-z]*"' $R/extension_queue_status.json | cut -d'"' -f4)
# cells progress (extensions + main seeds)
CEXT=$(ls $R/extensions/*/seeds/seed*/*/cells/*.json 2>/dev/null | wc -l)
CMAIN=$(find $R/seeds -path "*cells*.json" 2>/dev/null | wc -l)
# worker phase detail (up to 4)
PHASES=$(ps -eo pid,etimes,args | grep -E "run_v7|extension_queue|cell_queue|v7_queue" | grep -v grep | grep -v tmux | head -5 | awk '{printf "%s(%ss):", $1,$2; for(i=4;i<=NF;i++) if($i=="--phase"||$i=="--student"||$i=="--seed"||$i=="--train-task"||$i=="--extra-chronos-gpu1") printf " %s %s",$i,$(i+1); printf ";"}')
# last progress timestamp
LASTLOG=$(find $R/logs $R/extensions $R/shape_backfill_20260925/logs $R/extensions/*/shape_backfill_20260925/logs -name "*.log" -newermt "-10 minutes" 2>/dev/null | wc -l)
ALERT=0; MSG=""
# Check1: queue unfinished but no workers
if [ "$NW" -eq 0 ]; then
  if [ "$QSTATE" = "running" ] || [ "$QSTATE" = "pending" ] || [ "$CEXT" -lt 360 ]; then
    ALERT=1; MSG="QUEUE_ALIVE_BUT_NO_WORKER queue_state=$QSTATE pos=$QPOS/3 ext_cells=$CEXT/360"
  fi
fi
# Check2: low util twice (persist counters)
echo "$U0 $U1 $NOW" >> $STATE.lowutil
tail -2 $STATE.lowutil > $STATE.lowutil2; mv $STATE.lowutil2 $STATE.lowutil
L=$(cat $STATE.lowutil)
LU0a=$(sed -n 1p $STATE.lowutil | awk '{print $1}'); LU1a=$(sed -n 1p $STATE.lowutil | awk '{print $2}')
LU0b=$(sed -n 2p $STATE.lowutil | awk '{print $1}'); LU1b=$(sed -n 2p $STATE.lowutil | awk '{print $2}')
# hysteresis: alert <40 twice, clear >=55; 40-55 keeps previous class (no flapping)
PREV=$(cat $STATE.lowclass 2>/dev/null || echo OK)
CUR=$PREV
if [ -n "$LU0a" ] && [ -n "$LU0b" ]; then
  if [ "$LU0a" -lt 40 ] && [ "$LU0b" -lt 40 ] && [ "$U0" -lt 40 ]; then CUR=LOW0; fi
  if [ "$LU1a" -lt 40 ] && [ "$LU1b" -lt 40 ] && [ "$U1" -lt 40 ]; then CUR=LOW1; fi
fi
if [ "$CUR" = "LOW0" ] || [ "$CUR" = "LOW1" ]; then
  if [ "$CUR" = "LOW0" ] && [ "$U0" -ge 55 ]; then CUR=OK; fi
  if [ "$CUR" = "LOW1" ] && [ "$U1" -ge 55 ]; then CUR=OK; fi
else
  CUR=OK
fi
echo "$CUR" > $STATE.lowclass
if [ "$CUR" = "LOW0" ]; then ALERT=1; MSG="$MSG LOW_UTIL_GPU0_TWICE=${LU0a},${LU0b}%"; fi
if [ "$CUR" = "LOW1" ]; then ALERT=1; MSG="$MSG LOW_UTIL_GPU1_TWICE=${LU1a},${LU1b}%"; fi
# double zero single shot with mem also zero = no process
if [ "$U0" -eq 0 ] && [ "$U1" -eq 0 ] && [ "$M0" -lt 1000 ] && [ "$M1" -lt 1000 ]; then
  ALERT=1; MSG="$MSG BOTH_GPU_IDLE_EMPTY mem0=${M0}MiB mem1=${M1}MiB"
fi
# log silence > 15min while workers exist
if [ "$NW" -gt 0 ] && [ "$LASTLOG" -eq 0 ]; then
  ALERT=1; MSG="$MSG LOG_SILENCE_15MIN workers=$NW"
fi
echo "SNAPSHOT util0=${U0}%/${M0}MiB/${P0}W util1=${U1}%/${M1}MiB/${P1}W workers=$NW queue=$QSTATE:$QPOS/3 ext_cells=$CEXT main_cells=$CMAIN phases=$PHASES"
if [ "$ALERT" -eq 1 ]; then echo "ALERT $MSG"; else echo OK; fi
