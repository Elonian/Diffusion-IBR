#!/usr/bin/env bash
set -euo pipefail

# Uses the official nerfacto baseline flow with persistent caches on disk.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_ID="073f5a9b983ced6fb28b23051260558b165f328a16b2d33fe20585b7ee4ad561"

export INSTALL_NERF_ENV="${INSTALL_NERF_ENV:-1}"

exec bash "${SCRIPT_DIR}/run_official_nerfacto_full_baseline_dl3dv_scene.sh" "${SCENE_ID}"
