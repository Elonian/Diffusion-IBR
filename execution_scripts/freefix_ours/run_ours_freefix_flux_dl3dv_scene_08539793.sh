#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_ours_freefix_flux_dl3dv_scene.sh" \
  0853979305f7ecb80bd8fc2c8df916410d471ef04ed5f1a64e9651baa41d7695 "$@"
