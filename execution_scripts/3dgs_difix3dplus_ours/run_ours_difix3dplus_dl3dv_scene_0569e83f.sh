#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_ID="0569e83fdc248a51fc0ab082ce5e2baff15755c53c207f545e6d02d91f01d166"

export INSTALL_DEPS="${INSTALL_DEPS:-1}"
export INSTALL_BUILD_DEPS="${INSTALL_BUILD_DEPS:-1}"

exec bash "${SCRIPT_DIR}/run_ours_difix3dplus_dl3dv_scene.sh" "${SCENE_ID}"
