#!/usr/bin/env bash
set -Eeuo pipefail

plotly_chrome_url() {
    run_python -c 'import json, pathlib, choreographer; p=pathlib.Path(choreographer.__file__).parent / "resources" / "last_known_good_chrome.json"; data=json.loads(p.read_text()); print(next(x["url"] for x in data["downloads"]["chrome"] if x["platform"] == "linux64"))'
}

select_plotly_chrome_route() {
    local mode="${PLOTLY_CHROME_ROUTE:-auto}" url proxy_speed direct_speed
    case "$mode" in
        direct)
            unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
            echo direct
            return
            ;;
        proxy|auto) ;;
        *) echo "PLOTLY_CHROME_ROUTE must be auto, proxy, or direct." >&2; return 2 ;;
    esac
    if [[ -f /etc/network_turbo ]]; then
        set +u
        source /etc/network_turbo >/dev/null 2>&1
        set -u
    fi
    if [[ "$mode" == proxy ]]; then
        [[ -n "${https_proxy:-}${HTTPS_PROXY:-}" ]] || {
            echo "Proxy route requested but no HTTPS proxy is configured." >&2; return 1;
        }
        echo proxy
        return
    fi
    command -v curl >/dev/null || { echo proxy; return; }
    url="$(plotly_chrome_url)"
    proxy_speed=0
    if [[ -n "${https_proxy:-}${HTTPS_PROXY:-}" ]]; then
        proxy_speed="$(curl -sS --max-time 15 -r 0-262143 -o /dev/null -w "%{speed_download}" "$url" 2>/dev/null || true)"
    fi
    direct_speed="$(curl --noproxy "*" -sS --max-time 15 -r 0-262143 -o /dev/null -w "%{speed_download}" "$url" 2>/dev/null || true)"
    [[ "$proxy_speed" =~ ^[0-9]+([.][0-9]+)?$ ]] || proxy_speed=0
    [[ "$direct_speed" =~ ^[0-9]+([.][0-9]+)?$ ]] || direct_speed=0
    if awk -v direct="$direct_speed" -v proxy="$proxy_speed" 'BEGIN {exit !(direct > proxy)}'; then
        unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
        echo "Chrome route: direct is faster (proxy ${proxy_speed} B/s, direct ${direct_speed} B/s)." >&2
        echo direct
    else
        echo "Chrome route: proxy selected (proxy ${proxy_speed} B/s, direct ${direct_speed} B/s)." >&2
        echo proxy
    fi
}

chrome_system_packages() {
    case "$1" in
        libgbm.so.1) echo libgbm1 ;;
        libasound.so.2) echo libasound2 ;;
        libnss3.so) echo libnss3 ;;
        libnspr4.so) echo libnspr4 ;;
        libatk-1.0.so.0) echo libatk1.0-0 ;;
        libatk-bridge-2.0.so.0) echo libatk-bridge2.0-0 ;;
        libatspi.so.0) echo libatspi2.0-0 ;;
        libX11.so.6) echo libx11-6 ;;
        libXcomposite.so.1) echo libxcomposite1 ;;
        libXdamage.so.1) echo libxdamage1 ;;
        libXext.so.6) echo libxext6 ;;
        libXfixes.so.3) echo libxfixes3 ;;
        libXrandr.so.2) echo libxrandr2 ;;
        libxcb.so.1) echo libxcb1 ;;
        libxkbcommon.so.0) echo libxkbcommon0 ;;
        libdrm.so.2) echo libdrm2 ;;
        libcups.so.2) echo libcups2 ;;
        libpango-1.0.so.0) echo libpango-1.0-0 ;;
        libcairo.so.2) echo libcairo2 ;;
        libgtk-3.so.0) echo libgtk-3-0 ;;
        libXss.so.1) echo libxss1 ;;
        libXrender.so.1) echo libxrender1 ;;
        libXi.so.6) echo libxi6 ;;
        libXtst.so.6) echo libxtst6 ;;
        libGLESv2.so.2) echo libgles2 ;;
        libEGL.so.1) echo libegl1 ;;
        libwayland-client.so.0) echo libwayland-client0 ;;
        libwayland-server.so.0) echo libwayland-server0 ;;
        libXkbcommon-x11.so.0) echo libxkbcommon-x11-0 ;;
        *) return 1 ;;
    esac
}

check_plotly_chrome_libraries() {
    local browser="$1" install_missing="$2" missing library package
    local -a packages=() unresolved=()
    missing="$(ldd "$browser" | awk '/not found/ {print $1}')"
    [[ -z "$missing" ]] && return 0
    while read -r library; do
        [[ -z "$library" ]] && continue
        if package="$(chrome_system_packages "$library")"; then
            packages+=("$package")
        else
            unresolved+=("$library")
        fi
    done <<< "$missing"
    if (( ${#unresolved[@]} )); then
        printf 'Unmapped Chrome shared libraries: %s\n' "${unresolved[*]}" >&2
        return 1
    fi
    if [[ "$install_missing" != true ]]; then
        printf 'Chrome is missing system libraries: %s\n' "$missing" >&2
        return 1
    fi
    command -v apt-get >/dev/null || {
        printf 'Install Chrome libraries manually: %s\n' "${packages[*]}" >&2; return 1;
    }
    if (( EUID != 0 )) && ! command -v sudo >/dev/null; then
        printf 'Root or sudo is required to install Chrome libraries: %s\n' "${packages[*]}" >&2
        return 1
    fi
    echo "Installing Chrome shared libraries: ${packages[*]}"
    if (( EUID == 0 )); then
        env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY apt-get update -qq
        env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY apt-get install -y --no-install-recommends "${packages[@]}"
    else
        sudo env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY apt-get update -qq
        sudo env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY apt-get install -y --no-install-recommends "${packages[@]}"
    fi
    missing="$(ldd "$browser" | awk '/not found/ {print $1}')"
    [[ -z "$missing" ]] || {
        printf 'Chrome still has unresolved shared libraries: %s\n' "$missing" >&2; return 1;
    }
}

ensure_plotly_font() {
    local family
    command -v fc-match >/dev/null || { echo "fontconfig fc-match is required for Plotly font validation." >&2; return 1; }
    family="$(fc-match "Times New Roman" -f "%{family}" | head -1)"
    if [[ "$family" == "Times New Roman" ]]; then
        echo "Times New Roman font is available."
        return 0
    fi
    if [[ "$family" != "Liberation Serif" ]]; then
        command -v apt-get >/dev/null || { echo "Install Times New Roman or Liberation Serif fonts manually." >&2; return 1; }
        echo "Times New Roman is unavailable; installing the open-source Liberation Serif fallback."
        if (( EUID == 0 )); then
            env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY apt-get update -qq
            env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY apt-get install -y --no-install-recommends fonts-liberation2
        else
            sudo env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY apt-get update -qq
            sudo env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY apt-get install -y --no-install-recommends fonts-liberation2
        fi
        family="$(fc-match "Times New Roman" -f "%{family}" | head -1)"
    fi
    [[ "$family" == "Liberation Serif" || "$family" == "Times New Roman" ]] || {
        echo "Times New Roman-compatible serif font is unavailable: $family" >&2; return 1;
    }
    echo "Times New Roman-compatible font resolved as $family."
}

check_plotly_font() {
    local family
    command -v fc-match >/dev/null || { echo "fontconfig fc-match is required for Plotly font validation." >&2; return 1; }
    family="$(fc-match "Times New Roman" -f "%{family}" | head -1)"
    [[ "$family" == "Times New Roman" || "$family" == "Liberation Serif" ]] || {
        echo "Times New Roman-compatible serif font is unavailable: $family" >&2; return 1;
    }
    echo "Plotly serif font resolved as $family."
}

install_plotly_backend() {
    local cache_dir browser route env_prefix
    cache_dir="$RAGE_INSTALL_ROOT/cache/chrome"
    mkdir -p "$cache_dir"
    browser="$cache_dir/chrome-linux64/chrome"
    if [[ ! -x "$browser" ]]; then
        route="$(select_plotly_chrome_route)"
        echo "Downloading Kaleido Chrome backend into $cache_dir via $route."
        if [[ "$route" == direct ]]; then
            env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
                conda run --no-capture-output -n "$ENV_NAME" python -c 'import sys; from pathlib import Path; import kaleido; print(kaleido.get_chrome_sync(path=Path(sys.argv[1])))' "$cache_dir"
        else
            run_python -c 'import sys; from pathlib import Path; import kaleido; print(kaleido.get_chrome_sync(path=Path(sys.argv[1])))' "$cache_dir"
        fi
    fi
    [[ -x "$browser" ]] || { echo "Kaleido Chrome binary not found: $browser" >&2; return 1; }
    check_plotly_chrome_libraries "$browser" true
    ensure_plotly_font
    env_prefix="$(run_python -c 'import sys; print(sys.prefix)')"
    mkdir -p "$env_prefix/etc/conda/activate.d" "$env_prefix/etc/conda/deactivate.d"
    printf 'export BROWSER_PATH=%q\n' "$browser" > "$env_prefix/etc/conda/activate.d/plotly_browser.sh"
    printf 'unset BROWSER_PATH\n' > "$env_prefix/etc/conda/deactivate.d/plotly_browser.sh"
    BROWSER_PATH="$browser" run_python -c 'import plotly.graph_objects as go; import plotly.io as pio; image=pio.to_image(go.Figure(go.Scatter(x=[0,1], y=[0,1])), format="png"); assert image.startswith(bytes.fromhex("89504e470d0a1a0a")); print("Plotly static PNG smoke passed:", len(image), "bytes")'
}

verify_plotly_backend() {
    local browser cache_dir
    cache_dir="$RAGE_INSTALL_ROOT/cache/chrome"
    browser="${BROWSER_PATH:-$cache_dir/chrome-linux64/chrome}"
    [[ -x "$browser" ]] || { echo "Kaleido Chrome binary missing: $browser" >&2; return 1; }
    check_plotly_chrome_libraries "$browser" false
    check_plotly_font
    BROWSER_PATH="$browser" run_python -c 'import kaleido, plotly; import plotly.graph_objects as go; import plotly.io as pio; image=pio.to_image(go.Figure(go.Scatter(x=[0,1], y=[0,1])), format="png"); assert image.startswith(bytes.fromhex("89504e470d0a1a0a")); print("Plotly static PNG verified:", plotly.__version__, len(image), "bytes")'
}
