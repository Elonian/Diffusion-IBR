#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/mntdatalora/src/Diffusion-IBR}"
MAX_JOBS="${MAX_JOBS:-3}"

export VENV_DIR="${VENV_DIR:-${REPO_ROOT}/cache_weights/persistent_python_nerfacto_vanilla_py312_20260423a}"
export INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-1}"
export FORCE_REINSTALL_REQUIREMENTS="${FORCE_REINSTALL_REQUIREMENTS:-0}"
export INSTALL_TCNN="${INSTALL_TCNN:-0}"
export AUTO_FALLBACK_TORCH="${AUTO_FALLBACK_TORCH:-1}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/official_nerfacto_full_baseline}"
export LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/execution}"
export CUDA_DEVICE="${CUDA_DEVICE:-0}"
export DEVICE_TYPE="${DEVICE_TYPE:-cuda}"
export NUM_DEVICES="${NUM_DEVICES:-1}"
export BASELINE_TOTAL_STEPS="${BASELINE_TOTAL_STEPS:-60000}"
export BASELINE_STEPS_PER_SAVE="${BASELINE_STEPS_PER_SAVE:-10000}"
export BASELINE_STEPS_PER_EVAL_BATCH="${BASELINE_STEPS_PER_EVAL_BATCH:-10000}"
export NS_VIS="${NS_VIS:-tensorboard}"
export SAVE_ALL_CHECKPOINTS="${SAVE_ALL_CHECKPOINTS:-1}"
export SETUP_LOCK_FILE="${SETUP_LOCK_FILE:-${REPO_ROOT}/cache_weights/locks/nerfacto_full_baseline_setup.lock}"
export SETUP_LOCK_WAIT_SECONDS="${SETUP_LOCK_WAIT_SECONDS:-1800}"
export DRY_RUN="${DRY_RUN:-0}"

SCRIPTS=(
  "${SCRIPT_DIR}/run_official_nerfacto_full_baseline_dl3dv_scene_0569e83f.sh"
  "${SCRIPT_DIR}/run_official_nerfacto_full_baseline_dl3dv_scene_06da7966.sh"
  "${SCRIPT_DIR}/run_official_nerfacto_full_baseline_dl3dv_scene_073f5a9b.sh"
  "${SCRIPT_DIR}/run_official_nerfacto_full_baseline_dl3dv_scene_07d9f972.sh"
  "${SCRIPT_DIR}/run_official_nerfacto_full_baseline_dl3dv_scene_08539793.sh"
)

echo "[nerfacto-pending] UTC start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[nerfacto-pending] max_jobs=${MAX_JOBS}"
echo "[nerfacto-pending] venv=${VENV_DIR}"
echo "[nerfacto-pending] output_root=${OUTPUT_ROOT}"

run_one() {
  local script="$1"
  local name
  local rc

  name="$(basename "${script}")"
  echo "[nerfacto-pending] START ${name} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  bash "${script}"
  rc=$?

  if [[ "${rc}" -eq 0 ]]; then
    echo "[nerfacto-pending] DONE ${name} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  else
    echo "[nerfacto-pending] FAILED ${name} rc=${rc} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
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

echo "[nerfacto-pending] UTC done $(date -u +%Y-%m-%dT%H:%M:%SZ) fail=${fail}"
exit "${fail}"
