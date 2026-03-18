#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_ID="073f5a9b983ced6fb28b23051260558b165f328a16b2d33fe20585b7ee4ad561"

export INSTALL_DEPS="${INSTALL_DEPS:-1}"
export INSTALL_BUILD_DEPS="${INSTALL_BUILD_DEPS:-1}"

exec bash "${SCRIPT_DIR}/run_ours_difix3dplus_dl3dv_scene.sh" "${SCENE_ID}"
