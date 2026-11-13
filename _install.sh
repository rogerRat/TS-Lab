#!/usr/bin/env bash
# Single entry point: install and verify every environment by default.
set -Eeuo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${PROJECT_ROOT}/scripts/env/conda_profiles.sh"
main "$@"
