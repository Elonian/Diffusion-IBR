#!/usr/bin/env bash
set -euo pipefail

# Uses the existing persistent CUDA toolkit and cached gsplat build on disk.
# This wrapper intentionally skips dependency/build setup and goes straight to training.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_ID="06da796666297fe4c683c231edf56ec00148a6a52ab5bb159fe1be31f53a58df"

export INSTALL_DEPS=0
export INSTALL_BUILD_DEPS=0

exec bash "${SCRIPT_DIR}/run_official_difix3d_dl3dv_scene.sh" "${SCENE_ID}"
