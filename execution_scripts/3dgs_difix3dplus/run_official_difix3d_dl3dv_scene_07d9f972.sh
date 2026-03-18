#!/usr/bin/env bash
set -euo pipefail

# Uses the existing persistent CUDA toolkit and cached gsplat build on disk.
# This wrapper intentionally skips dependency/build setup and goes straight to training.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_ID="07d9f9724ca854fae07cb4c57d7ea22bf667d5decd4058f547728922f909956b"

export INSTALL_DEPS=0
export INSTALL_BUILD_DEPS=0

exec bash "${SCRIPT_DIR}/run_official_difix3d_dl3dv_scene.sh" "${SCENE_ID}"
