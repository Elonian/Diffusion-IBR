#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_ours_freefix_dl3dv_scene.sh" \
  06da796666297fe4c683c231edf56ec00148a6a52ab5bb159fe1be31f53a58df "$@"
