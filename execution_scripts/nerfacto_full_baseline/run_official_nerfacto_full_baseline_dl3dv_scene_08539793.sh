#!/usr/bin/env bash
set -euo pipefail

# Uses the official nerfacto baseline flow with persistent caches on disk.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_ID="0853979305f7ecb80bd8fc2c8df916410d471ef04ed5f1a64e9651baa41d7695"

export INSTALL_NERF_ENV="${INSTALL_NERF_ENV:-1}"

exec bash "${SCRIPT_DIR}/run_official_nerfacto_full_baseline_dl3dv_scene.sh" "${SCENE_ID}"
