#!/usr/bin/env bash
set -euo pipefail

# Uses the existing persistent CUDA toolkit and cached gsplat build on disk.
# This wrapper intentionally skips dependency/build setup and goes straight to training.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_ID="0853979305f7ecb80bd8fc2c8df916410d471ef04ed5f1a64e9651baa41d7695"

export INSTALL_DEPS=0
export INSTALL_BUILD_DEPS=0

exec bash "${SCRIPT_DIR}/run_official_difix3d_dl3dv_scene.sh" "${SCENE_ID}"
