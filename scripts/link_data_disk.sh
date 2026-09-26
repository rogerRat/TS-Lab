#!/usr/bin/env bash
# Move storage-heavy project directories to the AutoDL data disk and replace
# them with absolute symlinks.  The operation is deliberately non-destructive:
# existing non-empty destinations and symlinks to another location are refused.
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
DATA_ROOT="${AUTODL_DATA_ROOT:-/root/autodl-tmp/TS3}"
DRY_RUN=0

usage() {
    printf 'Usage: %s [--data-root PATH] [--dry-run]\n' "$0"
    printf '\nMoves and links: data outputs logs cache archive\n'
}

while (($#)); do
    case "$1" in
        --data-root)
            (($# >= 2)) || { echo "--data-root requires a path" >&2; exit 2; }
            DATA_ROOT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

DATA_ROOT="$(mkdir -p "$DATA_ROOT" && cd "$DATA_ROOT" && pwd -P)"
case "$DATA_ROOT" in
    "$PROJECT_ROOT"|"$PROJECT_ROOT"/*)
        echo "refusing data root inside project root: $DATA_ROOT" >&2
        exit 1
        ;;
esac

project_device="$(df -P "$PROJECT_ROOT" | awk 'NR == 2 {print $1}')"
data_device="$(df -P "$DATA_ROOT" | awk 'NR == 2 {print $1}')"
[[ -n "$data_device" ]] || { echo "cannot determine filesystem for $DATA_ROOT" >&2; exit 1; }
if [[ "$project_device" == "$data_device" ]]; then
    echo "refusing data root on the same filesystem as project: $DATA_ROOT ($data_device)" >&2
    exit 1
fi

if ((DRY_RUN)); then
    echo "dry-run: project=$PROJECT_ROOT data=$DATA_ROOT filesystem=$data_device"
else
    echo "linking project storage to $DATA_ROOT (filesystem $data_device)"
fi

for relative in data outputs logs cache archive; do
    source="$PROJECT_ROOT/$relative"
    target="$DATA_ROOT/$relative"

    if [[ -L "$source" ]]; then
        resolved="$(readlink -f "$source")"
        [[ "$resolved" == "$target" ]] || {
            echo "refusing symlink to unexpected target: $source -> $resolved" >&2
            exit 1
        }
        echo "ok: $source -> $target"
        continue
    fi

    if [[ -e "$source" ]]; then
        [[ -d "$source" ]] || { echo "refusing non-directory path: $source" >&2; exit 1; }
        [[ ! -e "$target" ]] || {
            if [[ -d "$target" ]] && [[ -z "$(find "$target" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
                echo "refusing ambiguous existing destination (empty): $target" >&2
            else
                echo "refusing existing destination: $target" >&2
            fi
            exit 1
        }
        if ((DRY_RUN)); then
            echo "would move $source -> $target and create symlink"
            continue
        fi
        mkdir -p "$(dirname "$target")"
        mv "$source" "$target"
    else
        if ((DRY_RUN)); then
            echo "would create $target and symlink $source"
            continue
        fi
        mkdir -p "$target"
    fi

    if ((DRY_RUN == 0)); then
        temporary="$source.link-tmp.$$"
        ln -s "$target" "$temporary"
        mv "$temporary" "$source"
        echo "linked: $source -> $target"
    fi
done

if ((DRY_RUN == 0)); then
    printf 'project=%s\ndata_root=%s\nfilesystem=%s\n' "$PROJECT_ROOT" "$DATA_ROOT" "$data_device" > "$DATA_ROOT/.ts3-data-links"
    echo "completed; manifest=$DATA_ROOT/.ts3-data-links"
fi
