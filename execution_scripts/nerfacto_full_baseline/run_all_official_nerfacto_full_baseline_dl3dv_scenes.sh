#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/mntdatalora/src/Diffusion-IBR}"
MAX_JOBS="${MAX_JOBS:-2}"
SHARED_SETUP_LOCK_FILE="${SETUP_LOCK_FILE:-${SETUP_LOCK_DIR:-${REPO_ROOT}/cache_weights/locks}/nerfacto_full_baseline_setup.lock}"

SCRIPTS=(
  "${SCRIPT_DIR}/run_official_nerfacto_full_baseline_dl3dv_scene.sh"
  "${SCRIPT_DIR}/run_official_nerfacto_full_baseline_dl3dv_scene_0569e83f.sh"
  "${SCRIPT_DIR}/run_official_nerfacto_full_baseline_dl3dv_scene_06da7966.sh"
  "${SCRIPT_DIR}/run_official_nerfacto_full_baseline_dl3dv_scene_073f5a9b.sh"
  "${SCRIPT_DIR}/run_official_nerfacto_full_baseline_dl3dv_scene_07d9f972.sh"
  "${SCRIPT_DIR}/run_official_nerfacto_full_baseline_dl3dv_scene_08539793.sh"
)

echo "[nerfacto-rerun] UTC start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[nerfacto-rerun] max_jobs=${MAX_JOBS}"

run_one() {
  local script="$1"
  local name
  local rc

  name="$(basename "${script}")"
  echo "[nerfacto-rerun] START ${name} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  SETUP_LOCK_FILE="${SHARED_SETUP_LOCK_FILE}" \
    bash "${script}"
  rc=$?

  if [[ "${rc}" -eq 0 ]]; then
    echo "[nerfacto-rerun] DONE ${name} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  else
    echo "[nerfacto-rerun] FAILED ${name} rc=${rc} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  fi

  return "${rc}"
}

fail=0
for script in "${SCRIPTS[@]}"; do
  while [[ "$(jobs -rp | wc -l)" -ge "${MAX_JOBS}" ]]; do
    wait -n || fail=1
  done
  run_one "${script}" &
done

while [[ "$(jobs -rp | wc -l)" -gt 0 ]]; do
  wait -n || fail=1
done

echo "[nerfacto-rerun] UTC done $(date -u +%Y-%m-%dT%H:%M:%SZ) fail=${fail}"
exit "${fail}"
