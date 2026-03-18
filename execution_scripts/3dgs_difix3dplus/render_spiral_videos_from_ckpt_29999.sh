#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mntdatalora/src/Diffusion-IBR}"
ACTIVATE_SCRIPT="${REPO_ROOT}/execution_scripts/3dgs_difix3dplus/activate_persistent_difix3d_env.sh"
DATA_ROOT="${REPO_ROOT}/data/DL3DV-10K-Benchmark"
VANILLA_ROOT="${REPO_ROOT}/outputs/official_difix3d/vanilla_gs"

if [[ $# -gt 0 ]]; then
  SCENES=("$@")
else
  SCENES=(
    "032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7"
    "0569e83fdc248a51fc0ab082ce5e2baff15755c53c207f545e6d02d91f01d166"
    "073f5a9b983ced6fb28b23051260558b165f328a16b2d33fe20585b7ee4ad561"
    "07d9f9724ca854fae07cb4c57d7ea22bf667d5decd4058f547728922f909956b"
    "0853979305f7ecb80bd8fc2c8df916410d471ef04ed5f1a64e9651baa41d7695"
    "06da796666297fe4c683c231edf56ec00148a6a52ab5bb159fe1be31f53a58df"
  )
fi

CUDA_DEVICE="${CUDA_DEVICE:-0}"
DATA_FACTOR="${DATA_FACTOR:-4}"
TRJ_LABELS="${TRJ_LABELS:-1}"
TRJ_OUTPUT="${TRJ_OUTPUT:-separate}"

test -f "${ACTIVATE_SCRIPT}" && source "${ACTIVATE_SCRIPT}"

for SCENE_ID in "${SCENES[@]}"; do
  BASE_DIR="${VANILLA_ROOT}/${SCENE_ID}"
  CKPT="${BASE_DIR}/ckpts/ckpt_29999_rank0.pt"
  DATA_DIR="${DATA_ROOT}/${SCENE_ID}/gaussian_splat"
  if [[ "${TRJ_OUTPUT}" == "inplace" ]]; then
    RUN_DIR="${BASE_DIR}"
  else
    RUN_DIR="${BASE_DIR}/traj_reruns/spiral_$(date +%Y%m%d_%H%M%S)"
  fi

  if [[ ! -f "${CKPT}" ]]; then
    echo "[skip] missing checkpoint: ${CKPT}"
    continue
  fi
  if [[ ! -d "${DATA_DIR}" ]]; then
    echo "[skip] missing data dir: ${DATA_DIR}"
    continue
  fi

  mkdir -p "${RUN_DIR}"
  echo "[scene] ${SCENE_ID}"
  echo "[ckpt] ${CKPT}"
  echo "[out ] ${RUN_DIR}"
  echo "[labels] ${TRJ_LABELS}"

  LABEL_ARGS=()
  if [[ "${TRJ_LABELS}" == "1" || "${TRJ_LABELS}" == "true" || "${TRJ_LABELS}" == "True" ]]; then
    LABEL_ARGS=(--traj-labels)
  fi

  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python "${REPO_ROOT}/works/Difix3D/examples/gsplat/simple_trainer_vanilla.py" default \
    --data_dir "${DATA_DIR}" \
    --data_factor "${DATA_FACTOR}" \
    --result_dir "${RUN_DIR}" \
    --ckpt "${CKPT}" \
    --no-normalize-world-space \
    --render-traj-path spiral \
    "${LABEL_ARGS[@]}"

done
