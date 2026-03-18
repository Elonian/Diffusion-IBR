#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec env FREEFIX_BACKEND=flux \
  bash "${SCRIPT_DIR}/run_ours_freefix_dl3dv_scene.sh" "$@"
