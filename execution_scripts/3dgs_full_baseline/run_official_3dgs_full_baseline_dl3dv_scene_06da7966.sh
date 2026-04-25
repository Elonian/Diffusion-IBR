#!/usr/bin/env bash
set -euo pipefail

# Uses the existing persistent CUDA toolkit and cached gsplat build by default.
# Set INSTALL_DEPS=1 and/or INSTALL_BUILD_DEPS=1 to run setup through this wrapper.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_ID="06da796666297fe4c683c231edf56ec00148a6a52ab5bb159fe1be31f53a58df"

export INSTALL_DEPS="${INSTALL_DEPS:-0}"
export INSTALL_BUILD_DEPS="${INSTALL_BUILD_DEPS:-0}"

exec bash "${SCRIPT_DIR}/run_official_3dgs_full_baseline_dl3dv_scene.sh" "${SCENE_ID}"
