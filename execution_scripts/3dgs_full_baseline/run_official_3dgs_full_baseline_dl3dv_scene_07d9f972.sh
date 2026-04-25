#!/usr/bin/env bash
set -euo pipefail

# Uses the existing persistent CUDA toolkit and cached gsplat build by default.
# Set INSTALL_DEPS=1 and/or INSTALL_BUILD_DEPS=1 to run setup through this wrapper.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_ID="07d9f9724ca854fae07cb4c57d7ea22bf667d5decd4058f547728922f909956b"

export INSTALL_DEPS="${INSTALL_DEPS:-0}"
export INSTALL_BUILD_DEPS="${INSTALL_BUILD_DEPS:-0}"

exec bash "${SCRIPT_DIR}/run_official_3dgs_full_baseline_dl3dv_scene.sh" "${SCENE_ID}"
