#!/usr/bin/env bash
set -euo pipefail

# Uses the official nerfacto baseline flow with persistent caches on disk.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_ID="06da796666297fe4c683c231edf56ec00148a6a52ab5bb159fe1be31f53a58df"

export INSTALL_NERF_ENV="${INSTALL_NERF_ENV:-1}"

exec bash "${SCRIPT_DIR}/run_official_nerfacto_full_baseline_dl3dv_scene.sh" "${SCENE_ID}"
