#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_ID="032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7"

export INSTALL_DEPS="${INSTALL_DEPS:-1}"
export INSTALL_BUILD_DEPS="${INSTALL_BUILD_DEPS:-1}"

exec bash "${SCRIPT_DIR}/run_ours_difix3dplus_dl3dv_scene.sh" "${SCENE_ID}"
