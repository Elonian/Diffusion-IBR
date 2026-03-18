#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_ours_freefix_dl3dv_scene.sh" \
  0569e83fdc248a51fc0ab082ce5e2baff15755c53c207f545e6d02d91f01d166 "$@"
