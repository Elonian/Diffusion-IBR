#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_ID="06da796666297fe4c683c231edf56ec00148a6a52ab5bb159fe1be31f53a58df"

export INSTALL_DEPS="${INSTALL_DEPS:-1}"
export INSTALL_BUILD_DEPS="${INSTALL_BUILD_DEPS:-1}"

exec bash "${SCRIPT_DIR}/run_ours_difix3dplus_dl3dv_scene.sh" "${SCENE_ID}"
