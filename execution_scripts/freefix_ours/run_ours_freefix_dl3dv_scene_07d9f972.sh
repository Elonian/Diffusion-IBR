#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_ours_freefix_dl3dv_scene.sh" \
  07d9f9724ca854fae07cb4c57d7ea22bf667d5decd4058f547728922f909956b "$@"
