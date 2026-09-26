#!/usr/bin/env bash
# Container watchdog with per-session countdowns.
#
# Each monitored tmux session owns an independent --timeout countdown. New
# pane output in a session resets only that session's countdown. The container
# is shut down only after EVERY monitored session's countdown has expired
# (sessions that never appeared do not count; sessions that disappeared after
# being seen count as expired).
#
# A monitored session must therefore keep emitting output (heartbeat) while
# its work is alive; otherwise long silent computations would trigger a false
# shutdown.
#
# usage: scripts/dock_shut.sh [--mode term|log] [--timeout SECONDS] [--poll SECONDS] --sessions SESSION [SESSION ...]

set -u

usage() {
    echo "usage: scripts/dock_shut.sh [--mode term|log] [--timeout SECONDS] [--poll SECONDS] --sessions SESSION [SESSION ...]" >&2
    exit 2
}

mode=log
timeout=300
poll=10
sessions=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --mode)
            [ "$#" -ge 2 ] || usage
            mode=$2
            shift 2
            ;;
        --timeout)
            [ "$#" -ge 2 ] || usage
            case "$2" in
                ''|*[!0-9]*) usage ;;
            esac
            timeout=$2
            shift 2
            ;;
        --poll)
            [ "$#" -ge 2 ] || usage
            case "$2" in
                ''|*[!0-9]*) usage ;;
            esac
            poll=$2
            shift 2
            ;;
        --sessions)
            shift
            [ "$#" -gt 0 ] || usage
            while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do
                sessions+=("$1")
                shift
            done
            ;;
        *)
            usage
            ;;
    esac
done

[ "$mode" = term ] || [ "$mode" = log ] || usage
[ "${#sessions[@]}" -gt 0 ] || usage

declare -A seen=()
declare -A last_hash=()
declare -A deadline=()

now() { date +%s; }

session_alive() {
    tmux has-session -t "$1" 2>/dev/null
}

pane_hash() {
    tmux capture-pane -p -t "$1" -S -200 2>/dev/null | cksum
}

echo "$(date '+%F %T') watchdog started: mode=$mode timeout=${timeout}s poll=${poll}s sessions=${sessions[*]}"

while true; do
    current=$(now)
    all_seen=1
    all_dead=1
    all_expired=1

    for session in "${sessions[@]}"; do
        if session_alive "$session"; then
            if [ "${seen[$session]:-0}" -eq 0 ]; then
                seen[$session]=1
                last_hash[$session]=$(pane_hash "$session")
                deadline[$session]=$((current + timeout))
                echo "$(date '+%F %T') session=$session first seen; countdown set to ${timeout}s"
            else
                hash=$(pane_hash "$session")
                if [ "$hash" != "${last_hash[$session]}" ]; then
                    last_hash[$session]=$hash
                    deadline[$session]=$((current + timeout))
                elif [ "$current" -ge "${deadline[$session]}" ]; then
                    echo "$(date '+%F %T') session=$session silent >= ${timeout}s (countdown expired)"
                fi
            fi
            all_dead=0
        else
            if [ "${seen[$session]:-0}" -eq 1 ]; then
                echo "$(date '+%F %T') session=$session gone (countdown expired)"
            else
                all_seen=0
                all_expired=0
                continue
            fi
        fi
        [ "${seen[$session]:-0}" -eq 1 ] || all_seen=0
        if [ "${seen[$session]:-0}" -eq 1 ] && session_alive "$session" && [ "$current" -lt "${deadline[$session]}" ]; then
            all_expired=0
        fi
    done

    if [ "$mode" = term ]; then
        if [ "$all_seen" -eq 1 ] && [ "$all_dead" -eq 1 ]; then
            echo "$(date '+%F %T') all monitored sessions finished; shutting down"
            /usr/bin/shutdown
            exit 0
        fi
    else
        if [ "$all_seen" -eq 1 ] && [ "$all_expired" -eq 1 ]; then
            echo "$(date '+%F %T') every monitored session silent >= ${timeout}s; shutting down"
            /usr/bin/shutdown
            exit 0
        fi
    fi
    sleep "$poll"
done
