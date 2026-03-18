#!/usr/bin/env bash
set -euo pipefail

# Uses the official nerfacto baseline flow with persistent caches on disk.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_ID="07d9f9724ca854fae07cb4c57d7ea22bf667d5decd4058f547728922f909956b"

export INSTALL_NERF_ENV="${INSTALL_NERF_ENV:-1}"

exec bash "${SCRIPT_DIR}/run_official_nerfacto_full_baseline_dl3dv_scene.sh" "${SCENE_ID}"
