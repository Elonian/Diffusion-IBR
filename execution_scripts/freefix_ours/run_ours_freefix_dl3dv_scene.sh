#!/usr/bin/env bash
set -euo pipefail

# Our FreeFix flow on one DL3DV scene:
# - uses scripts/trainers/freefix_runner.py directly
# - does not depend on external FreeFix runtime imports
# - writes to outputs/freefix_self by default (override with OUTPUT_ROOT)

REPO_ROOT="/mntdatalora/src/Diffusion-IBR"
DL3DV_ROOT="${DL3DV_ROOT:-${REPO_ROOT}/data/DL3DV-10K-Benchmark}"
RUNNER_PY="${REPO_ROOT}/scripts/trainers/freefix_runner.py"
LOG_ROOT="${REPO_ROOT}/logs/execution"

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

DATA_FACTOR="${DATA_FACTOR:-4}"
TEST_EVERY="${TEST_EVERY:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
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
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/cache_weights}"

case "${FREEFIX_BACKEND}" in
  sdxl)
    CONFIG_JSON="${CONFIG_JSON:-${REPO_ROOT}/configs/freefix_self_sdxl.json}"
    ;;
  flux)
    CONFIG_JSON="${CONFIG_JSON:-${REPO_ROOT}/configs/freefix_self_flux.json}"
    ;;
  *)
    echo "FREEFIX_BACKEND must be 'sdxl' or 'flux' (got '${FREEFIX_BACKEND}')." >&2
    exit 1
    ;;
esac

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

if [[ ! -f "${RUNNER_PY}" ]]; then
  echo "Missing FreeFix runner: ${RUNNER_PY}" >&2
  exit 1
fi

if [[ ! -f "${CONFIG_JSON}" ]]; then
  echo "Missing FreeFix config: ${CONFIG_JSON}" >&2
  exit 1
fi

if [[ ! -d "${DL3DV_ROOT}/${SCENE_ID}/gaussian_splat" ]]; then
  echo "Missing scene data: ${DL3DV_ROOT}/${SCENE_ID}/gaussian_splat" >&2
  exit 1
fi

echo "Scene: ${SCENE_ID}"
echo "Backend: ${FREEFIX_BACKEND}"
echo "Stage: ${FREEFIX_STAGE}"
echo "Data root: ${DL3DV_ROOT}"
echo "Config: ${CONFIG_JSON}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Cache root: ${CACHE_ROOT}"
echo "Python: ${PYTHON_BIN}"

if [[ "${INSTALL_DEPS}" == "1" || "${AUTO_INSTALL_MISSING_DEPS}" == "1" ]]; then
  echo "[warn] Dependency auto-install is no longer handled by this scene wrapper."
  echo "[warn] Prepare the Python environment first, then rerun this script."
fi

CMD=(
  "${PYTHON_BIN}" "${RUNNER_PY}"
  --config "${CONFIG_JSON}"
  --stage "${FREEFIX_STAGE}"
  --scene_id "${SCENE_ID}"
  --backend "${FREEFIX_BACKEND}"
  --repo_root "${REPO_ROOT}"
  --dl3dv_root "${DL3DV_ROOT}"
  --output_root "${OUTPUT_ROOT}"
  --cache_root "${CACHE_ROOT}"
  --data_factor "${DATA_FACTOR}"
  --test_every "${TEST_EVERY}"
  --num_workers "${NUM_WORKERS}"
  --recon_steps "${RECON_STEPS}"
  --refine_cycles "${REFINE_CYCLES}"
  --refine_steps_per_cycle "${REFINE_STEPS_PER_CYCLE}"
  --refine_num_views "${REFINE_NUM_VIEWS}"
  --freefix_num_inference_steps "${FREEFIX_NUM_INFERENCE_STEPS}"
  --freefix_strength "${FREEFIX_STRENGTH}"
  --freefix_guidance_scale "${FREEFIX_GUIDANCE_SCALE}"
  --freefix_mask_scheduler "${FREEFIX_MASK_SCHEDULER}"
  --freefix_guide_ratio "${FREEFIX_GUIDE_RATIO}"
  --freefix_warp_ratio "${FREEFIX_WARP_RATIO}"
  --freefix_certainty_scales "${FREEFIX_CERTAINTY_SCALES}"
  --freefix_hessian_attrs "${FREEFIX_HESSIAN_ATTRS}"
  --gen_prob "${GEN_PROB}"
  --gen_loss_weight "${GEN_LOSS_WEIGHT}"
  --prompt "${PROMPT}"
  --negative_prompt "${NEGATIVE_PROMPT}"
  --python_bin "${PYTHON_BIN}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  CMD+=(--dry_run)
fi

"${CMD[@]}"

echo "[done] ours freefix run finished."
