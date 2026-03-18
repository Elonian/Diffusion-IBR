#!/usr/bin/env bash
set -euo pipefail

# Our FreeFix flow on one DL3DV scene:
# - uses scripts/trainers/freefix_runner.py via execution_scripts/freefix/run_freefix_self_dl3dv_scene.sh
# - does not depend on works/FreeFix runtime imports
# - writes to outputs/freefix_self by default (override with OUTPUT_ROOT)

REPO_ROOT="/mntdatalora/src/Diffusion-IBR"
DL3DV_ROOT="${DL3DV_ROOT:-${REPO_ROOT}/data/DL3DV-10K-Benchmark}"
LOG_ROOT="${REPO_ROOT}/logs/execution"
BASE_SCRIPT="${REPO_ROOT}/execution_scripts/freefix/run_freefix_self_dl3dv_scene.sh"
PERSISTENT_SETUP_SCRIPT="${PERSISTENT_SETUP_SCRIPT:-${REPO_ROOT}/execution_scripts/freefix/setup_persistent_official_freefix_env.sh}"
PERSISTENT_ACTIVATE_SCRIPT="${PERSISTENT_ACTIVATE_SCRIPT:-${REPO_ROOT}/execution_scripts/freefix/activate_persistent_official_freefix_env.sh}"

SCENE_ID="${1:-032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/opt/conda/bin/python" ]]; then
    PYTHON_BIN="/opt/conda/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

FREEFIX_BACKEND="${FREEFIX_BACKEND:-sdxl}"   # sdxl | flux
FREEFIX_STAGE="${FREEFIX_STAGE:-full}"       # recon | refine | eval | full
# Strict comparison default: match Difix split convention (278 fixer targets on this scene).
FREEFIX_SPLIT_MODE="${FREEFIX_SPLIT_MODE:-difix3d}"  # difix3d | freefix
AUTO_RESUME_STAGE="${AUTO_RESUME_STAGE:-1}"

DATA_FACTOR="${DATA_FACTOR:-4}"
TEST_EVERY="${TEST_EVERY:-8}"
NUM_WORKERS="2"
RECON_STEPS="${RECON_STEPS:-30000}"
REFINE_CYCLES="${REFINE_CYCLES:-1}"
REFINE_STEPS_PER_CYCLE="${REFINE_STEPS_PER_CYCLE:-400}"
REFINE_NUM_VIEWS="${REFINE_NUM_VIEWS:-0}"

FREEFIX_NUM_INFERENCE_STEPS="${FREEFIX_NUM_INFERENCE_STEPS:-50}"
FREEFIX_STRENGTH="${FREEFIX_STRENGTH:-0.6}"
FREEFIX_GUIDANCE_SCALE="${FREEFIX_GUIDANCE_SCALE:-3.5}"
FREEFIX_MASK_SCHEDULER="${FREEFIX_MASK_SCHEDULER:-0.3,0.9,1.0}"
FREEFIX_GUIDE_RATIO="${FREEFIX_GUIDE_RATIO:-1.0}"
FREEFIX_WARP_RATIO="${FREEFIX_WARP_RATIO:-0.5}"
FREEFIX_CERTAINTY_SCALES="${FREEFIX_CERTAINTY_SCALES:-0.001,0.01,0.1}"
FREEFIX_HESSIAN_ATTRS="${FREEFIX_HESSIAN_ATTRS:-means,quats,scales}"
GEN_PROB="${GEN_PROB:-0.1}"
GEN_LOSS_WEIGHT="${GEN_LOSS_WEIGHT:-0.2}"

PROMPT="${PROMPT:-A photorealistic real-world scene with consistent geometry, detailed textures, and natural lighting.}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, low quality, foggy, overall gray, subtitles, incomplete, ghost image, too close to camera}"

INSTALL_DEPS="${INSTALL_DEPS:-0}"
AUTO_INSTALL_MISSING_DEPS="${AUTO_INSTALL_MISSING_DEPS:-0}"
DRY_RUN="${DRY_RUN:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/freefix_self}"
PERSISTENT_PYTHON_ROOT="${PERSISTENT_PYTHON_ROOT:-${REPO_ROOT}/cache_weights/persistent_python_freefix}"

setup_logging() {
  local timestamp host_name script_name
  timestamp="$(date -u +%Y%m%d-%H%M%S)"
  host_name="${HOSTNAME:-unknownhost}"
  script_name="$(basename "$0" .sh)"
  LOG_DIR="${LOG_ROOT}/freefix_ours/${SCENE_ID}"
  LOG_FILE="${LOG_DIR}/${timestamp}_${host_name}_${script_name}.log"

  mkdir -p "${LOG_DIR}"
  ln -sfn "$(basename "${LOG_FILE}")" "${LOG_DIR}/latest.log" 2>/dev/null || true

  if [[ -z "${__DIFFUSION_IBR_SCRIPT_LOGGING:-}" ]]; then
    __DIFFUSION_IBR_SCRIPT_LOGGING=1
    exec > >(tee -a "${LOG_FILE}") 2>&1
  fi

  echo "[log] Writing stdout/stderr to ${LOG_FILE}"
}

setup_logging

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "Missing base script: ${BASE_SCRIPT}" >&2
  exit 1
fi

if [[ ! -d "${DL3DV_ROOT}/${SCENE_ID}/gaussian_splat" ]]; then
  echo "Missing scene data: ${DL3DV_ROOT}/${SCENE_ID}/gaussian_splat" >&2
  exit 1
fi

echo "Scene: ${SCENE_ID}"
echo "Backend: ${FREEFIX_BACKEND}"
echo "Stage: ${FREEFIX_STAGE}"
echo "Split mode: ${FREEFIX_SPLIT_MODE}"
echo "Data root: ${DL3DV_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Persistent setup script: ${PERSISTENT_SETUP_SCRIPT}"

REPO_ROOT="${REPO_ROOT}" \
DL3DV_ROOT="${DL3DV_ROOT}" \
FREEFIX_BACKEND="${FREEFIX_BACKEND}" \
FREEFIX_STAGE="${FREEFIX_STAGE}" \
FREEFIX_SPLIT_MODE="${FREEFIX_SPLIT_MODE}" \
AUTO_RESUME_STAGE="${AUTO_RESUME_STAGE}" \
DATA_FACTOR="${DATA_FACTOR}" \
TEST_EVERY="${TEST_EVERY}" \
NUM_WORKERS="${NUM_WORKERS}" \
RECON_STEPS="${RECON_STEPS}" \
REFINE_CYCLES="${REFINE_CYCLES}" \
REFINE_STEPS_PER_CYCLE="${REFINE_STEPS_PER_CYCLE}" \
REFINE_NUM_VIEWS="${REFINE_NUM_VIEWS}" \
FREEFIX_NUM_INFERENCE_STEPS="${FREEFIX_NUM_INFERENCE_STEPS}" \
FREEFIX_STRENGTH="${FREEFIX_STRENGTH}" \
FREEFIX_GUIDANCE_SCALE="${FREEFIX_GUIDANCE_SCALE}" \
FREEFIX_MASK_SCHEDULER="${FREEFIX_MASK_SCHEDULER}" \
FREEFIX_GUIDE_RATIO="${FREEFIX_GUIDE_RATIO}" \
FREEFIX_WARP_RATIO="${FREEFIX_WARP_RATIO}" \
FREEFIX_CERTAINTY_SCALES="${FREEFIX_CERTAINTY_SCALES}" \
FREEFIX_HESSIAN_ATTRS="${FREEFIX_HESSIAN_ATTRS}" \
GEN_PROB="${GEN_PROB}" \
GEN_LOSS_WEIGHT="${GEN_LOSS_WEIGHT}" \
PROMPT="${PROMPT}" \
NEGATIVE_PROMPT="${NEGATIVE_PROMPT}" \
INSTALL_DEPS="${INSTALL_DEPS}" \
AUTO_INSTALL_MISSING_DEPS="${AUTO_INSTALL_MISSING_DEPS}" \
DRY_RUN="${DRY_RUN}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
ACTIVATE_SCRIPT="${PERSISTENT_ACTIVATE_SCRIPT}" \
PERSISTENT_SETUP_SCRIPT="${PERSISTENT_SETUP_SCRIPT}" \
PERSISTENT_PYTHON_ROOT="${PERSISTENT_PYTHON_ROOT}" \
PYTHON_BIN="${PYTHON_BIN}" \
bash "${BASE_SCRIPT}" "${SCENE_ID}"

echo "[done] ours freefix run finished."
