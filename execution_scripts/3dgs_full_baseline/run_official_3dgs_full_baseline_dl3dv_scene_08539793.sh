#!/usr/bin/env bash
set -euo pipefail

# Uses the existing persistent CUDA toolkit and cached gsplat build by default.
# Set INSTALL_DEPS=1 and/or INSTALL_BUILD_DEPS=1 to run setup through this wrapper.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_ID="0853979305f7ecb80bd8fc2c8df916410d471ef04ed5f1a64e9651baa41d7695"

export INSTALL_DEPS="${INSTALL_DEPS:-0}"
export INSTALL_BUILD_DEPS="${INSTALL_BUILD_DEPS:-0}"

exec bash "${SCRIPT_DIR}/run_official_3dgs_full_baseline_dl3dv_scene.sh" "${SCENE_ID}"
