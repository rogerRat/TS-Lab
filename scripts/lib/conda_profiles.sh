#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    echo 'Usage: bash _install.sh [--env main|timesfm|timemoe|sundial|timerxl|time|all] [--verify-only]'
    echo 'Default: all six environments. Requires conda, git and a data disk.'
}
run_python() { conda run --no-capture-output -n "$ENV_NAME" python "$@"; }
run_pip() { run_python -m pip "$@"; }

network_git() (
    # AutoDL's GitHub proxy must not slow down domestic pip/conda mirrors.
    if [[ -f /etc/network_turbo ]]; then
        set +u
        source /etc/network_turbo >/dev/null
    fi
    git "$@"
)

checkout_source() {
    local name="$1" url="$2" revision="$3" target stored
    target="$PROJECT_ROOT/third_parties/$name"
    stored="$RAGE_INSTALL_ROOT/third_parties/$name"
    if [[ ! -e "$target" && ! -L "$target" ]]; then
        [[ -e "$stored" ]] || network_git clone "$url" "$stored"
        [[ -d "$stored/.git" ]] || { echo "Not a checkout: $stored" >&2; return 1; }
        ln -s "$stored" "$target"
    fi
    [[ -d "$target/.git" ]] || { echo "Not a checkout: $target" >&2; return 1; }
    [[ -z "$(git -C "$target" status --porcelain --untracked-files=all)" ]] || {
        echo "Refusing modified checkout: $target" >&2; return 1;
    }
    if [[ "$(git -C "$target" rev-parse HEAD)" != "$revision" ]]; then
        network_git -C "$target" fetch origin "$revision"
        git -C "$target" checkout --detach "$revision"
    fi
    [[ "$(git -C "$target" rev-parse HEAD)" == "$revision" ]]
}

install_sources() {
    local lock="$1" name url revision extra
    while read -r name url revision extra; do
        [[ -z "$name" || "$name" == \#* ]] && continue
        [[ -n "$url" && -n "$revision" && -z "$extra" ]] || return 1
        checkout_source "$name" "$url" "$revision"
        run_pip install --no-deps --no-build-isolation "$PROJECT_ROOT/third_parties/$name"
    done < "$PROJECT_ROOT/requ/$lock"
}

install_profile() {
    if ! conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$ENV_NAME"; then
        conda create --yes --name "$ENV_NAME" python=3.10 'setuptools>=68' pip
    fi
    run_python -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'
    # Resolve CUDA libraries too; --no-deps breaks a fresh environment.
    run_pip install --extra-index-url "$PYTORCH_INDEX_URL" 'torch==2.7.1+cu128'
    run_pip install 'setuptools>=68' wheel -r "$PROJECT_ROOT/requ/$REQUIREMENTS"
    run_pip install --no-deps --no-build-isolation -e "$PROJECT_ROOT"
    case "$PROFILE" in
        main) install_sources third_parties.lock ;;
        time) install_sources time.third_parties.lock ;;
        timemoe)
            local name url revision
            read -r name url revision < <(awk '$1 == "timemoe-source"' "$PROJECT_ROOT/requ/model_integrations.lock")
            checkout_source Time-MoE-pinned "$url" "$revision"
            ;;
    esac
}

verify_profile() {
    run_pip check
    run_python - "$PROFILE" <<'PY'
import importlib
import sys
import torch
profile = sys.argv[1]
modules = {
    "main": ["chronos", "fev", "numpy", "pandas", "pyarrow", "scipy", "faiss",
             "ot", "threadpoolctl", "umap", "tensorboard", "matplotlib", "rage", "tsl",
             "rage.static_selection", "scripts.select_static"],
    "timesfm": ["timesfm3", "faiss", "scipy", "yaml", "plotly", "rage"],
    "timemoe": ["faiss", "scipy", "yaml", "plotly", "rage"],
    "sundial": ["faiss", "scipy", "yaml", "plotly", "rage"],
    "timerxl": ["faiss", "scipy", "yaml", "plotly", "rage"],
    "time": ["timebench", "timebench.evaluation.data", "chronos"],
}
for module in modules[profile]:
    importlib.import_module(module)
if profile == "timemoe":
    from adapters.foundation.timemoe import official_classes
    official_classes()
if profile == "timesfm":
    from transformers import TimesFm2_5ModelForPrediction
if profile in ("timemoe", "sundial", "timerxl"):
    import transformers
    assert transformers.__version__ == "4.40.1"
assert torch.__version__ == "2.7.1+cu128", torch.__version__
if torch.cuda.is_available():
    x = torch.randn(32, 32, device="cuda")
    (x @ x).sum().item()
    torch.cuda.synchronize()
    print("CUDA smoke passed:", torch.cuda.get_device_name())
else:
    print("CUDA unavailable: imports checked; GPU smoke skipped.")
print(profile, "verified", sys.executable)
PY
    if [[ "$PROFILE" == time && -n "${CHECKPOINT:-}" ]]; then
        run_python - "$CHECKPOINT" <<'PY'
import sys
from chronos import BaseChronosPipeline
BaseChronosPipeline.from_pretrained(sys.argv[1], device_map="cpu", torch_dtype="float32")
PY
    fi
}

main() {
    local selection=all verify_only=false
    while (( $# )); do
        case "$1" in
            --env) [[ $# -ge 2 ]] || { usage; return 2; }; selection="$2"; shift 2 ;;
            --verify-only) verify_only=true; shift ;;
            -h|--help) usage; return 0 ;;
            *) usage >&2; return 2 ;;
        esac
    done
    case "$selection" in all|main|timesfm|timemoe|sundial|timerxl|time) ;; *) usage >&2; return 2 ;; esac
    if ! command -v conda >/dev/null && [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    fi
    command -v conda >/dev/null || { echo 'conda is required' >&2; return 1; }
    command -v git >/dev/null || { echo 'git is required' >&2; return 1; }
    export RAGE_INSTALL_ROOT="${RAGE_INSTALL_ROOT:-/root/autodl-tmp/rage}"
    local existing_env_dirs
    existing_env_dirs="$(conda info --json | "$(conda info --base)/bin/python" -c 'import json,sys; print(":".join(json.load(sys.stdin)["envs_dirs"]))')"
    export CONDA_ENVS_PATH="$RAGE_INSTALL_ROOT/envs:$existing_env_dirs"
    export PYTHONNOUSERSITE=1
    unset PYTHONPATH
    export PIP_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
    export PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
    export PIP_NO_INPUT=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_USER=false
    export PIP_CACHE_DIR="$RAGE_INSTALL_ROOT/cache/pip"
    export CONDA_PKGS_DIRS="$RAGE_INSTALL_ROOT/cache/conda"
    export TMPDIR="$RAGE_INSTALL_ROOT/cache/tmp"
    if [[ "$verify_only" == false ]]; then
        [[ -d "$(dirname "$RAGE_INSTALL_ROOT")" ]] || {
            echo 'Set RAGE_INSTALL_ROOT to a directory on a writable data disk.' >&2; return 1;
        }
        df -h "$(dirname "$RAGE_INSTALL_ROOT")"
        local available
        available="$(df -Pk "$(dirname "$RAGE_INSTALL_ROOT")" | awk 'END {print $4}')"
        (( available >= 20 * 1024 * 1024 )) || {
            echo 'Need 20 GiB free: expand the data disk or set RAGE_INSTALL_ROOT.' >&2; return 1;
        }
        mkdir -p "$RAGE_INSTALL_ROOT"/{envs,third_parties,logs/install} \
            "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$TMPDIR" "$PROJECT_ROOT/third_parties"
        # Keep new environments addressable by name in future interactive shells.
        case ":$existing_env_dirs:" in
            *":$RAGE_INSTALL_ROOT/envs:"*) ;;
            *) conda config --append envs_dirs "$RAGE_INSTALL_ROOT/envs" ;;
        esac
        local logfile="$RAGE_INSTALL_ROOT/logs/install/$(date +%Y%m%d-%H%M%S)-$$.log"
        exec > >(tee -a "$logfile") 2>&1
        echo "Installation log: $logfile"
    fi
    cd "$PROJECT_ROOT"
    local profiles=(main timesfm timemoe sundial timerxl time)
    [[ "$selection" == all ]] || profiles=("$selection")
    local PROFILE ENV_NAME REQUIREMENTS
    for PROFILE in "${profiles[@]}"; do
        case "$PROFILE" in
            main) ENV_NAME=rage; REQUIREMENTS=requirements.txt ;;
            time) ENV_NAME=time; REQUIREMENTS=time-requirements.txt ;;
            *) ENV_NAME="rage-$PROFILE"; REQUIREMENTS="$PROFILE-requirements.txt" ;;
        esac
        echo "=== $PROFILE ($ENV_NAME) ==="
        [[ "$verify_only" == true ]] || install_profile
        verify_profile
        printf "verified\n" > "$RAGE_INSTALL_ROOT/envs/$ENV_NAME/.complete"
        if [[ "$verify_only" == false ]]; then
            run_pip freeze > "$RAGE_INSTALL_ROOT/logs/install/$ENV_NAME-$(date +%Y%m%d-%H%M%S).txt"
        fi
    done
    echo "Completed: ${profiles[*]}"
}
