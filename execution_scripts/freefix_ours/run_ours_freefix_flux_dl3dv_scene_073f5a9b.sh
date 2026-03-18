#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_ours_freefix_flux_dl3dv_scene.sh" \
  073f5a9b983ced6fb28b23051260558b165f328a16b2d33fe20585b7ee4ad561 "$@"
