#!/usr/bin/env bash
set -euo pipefail

# Uses the official nerfacto baseline flow with persistent caches on disk.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_ID="0569e83fdc248a51fc0ab082ce5e2baff15755c53c207f545e6d02d91f01d166"

export INSTALL_NERF_ENV="${INSTALL_NERF_ENV:-1}"

exec bash "${SCRIPT_DIR}/run_official_nerfacto_full_baseline_dl3dv_scene.sh" "${SCENE_ID}"
