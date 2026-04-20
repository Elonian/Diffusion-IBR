#!/usr/bin/env bash
set -euo pipefail

# Our Difix3D+ 3DGS flow on one DL3DV scene:
# - uses scripts/trainers/trainer.py (training_recipe=difix3d)
# - prefers an existing checkpoint from the local output root
# - can start from scratch when no local checkpoint exists
# - writes to a separate output root

REPO_ROOT="/mntdatalora/src/Diffusion-IBR"
DL3DV_ROOT="${REPO_ROOT}/data/DL3DV-10K-Benchmark"
TRAINER_PY="${REPO_ROOT}/scripts/trainers/trainer.py"
CONFIG_JSON="${CONFIG_JSON:-${REPO_ROOT}/configs/difix3d_plus_train.json}"
LOG_ROOT="${REPO_ROOT}/logs/execution"
ACTIVATE_SCRIPT="${ACTIVATE_SCRIPT:-}"

SCENE_ID="${1:-032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/opt/conda/bin/python" ]]; then
    PYTHON_BIN="/opt/conda/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi
DATA_FACTOR="${DATA_FACTOR:-4}"
TEST_EVERY="${TEST_EVERY:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
INSTALL_BUILD_DEPS="${INSTALL_BUILD_DEPS:-1}"
DRY_RUN="${DRY_RUN:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/ours_difix3dplus_gs}"
MAX_STEPS="${MAX_STEPS:-60000}"
EVAL_EVERY="${EVAL_EVERY:-10000}"
SAVE_EVERY="${SAVE_EVERY:-10000}"

STRICT_OFFICIAL_DIFIX="${STRICT_OFFICIAL_DIFIX:-1}"
ALLOW_RUNTIME_DIFIX_OVERRIDES="${ALLOW_RUNTIME_DIFIX_OVERRIDES:-0}"
ALLOW_REFERENCE_CKPT_FALLBACK="${ALLOW_REFERENCE_CKPT_FALLBACK:-0}"
ALLOW_LEGACY_VANILLA_FALLBACK="${ALLOW_LEGACY_VANILLA_FALLBACK:-0}"
START_FROM_SCRATCH="${START_FROM_SCRATCH:-1}"

OFFICIAL_FIX_STEPS="${OFFICIAL_FIX_STEPS:-3000,6000,8000,10000,12000,14000,16000,18000,20000,22000,24000,26000,28000,30000,32000,34000,36000,38000,40000,42000,44000,46000,48000,50000,52000,54000,56000,58000,60000}"
OFFICIAL_EVAL_STEPS="${OFFICIAL_EVAL_STEPS:-10000,20000,30000,35000,40000,45000,50000,55000,60000}"
OFFICIAL_SAVE_STEPS="${OFFICIAL_SAVE_STEPS:-10000,20000,30000,40000,45000,50000,55000,60000}"

FORCE_FROM_30000="${FORCE_FROM_30000:-0}"
START_CKPT="${START_CKPT:-}"

SCENE_DIR="${DL3DV_ROOT}/${SCENE_ID}"
GS_DATA_DIR="${SCENE_DIR}/gaussian_splat"
OURS_OUT="${OUTPUT_ROOT}/${SCENE_ID}"
OURS_CKPT_DIR="${OURS_OUT}/ckpts"

REFERENCE_VANILLA_CKPT="${REPO_ROOT}/outputs/official_difix3d/vanilla_gs/${SCENE_ID}/ckpts/ckpt_29999_rank0.pt"
REFERENCE_LEGACY_VANILLA_CKPT="${REPO_ROOT}/outputs/official_3dgs_full_baseline/${SCENE_ID}/ckpts/ckpt_29999_rank0.pt"
REFERENCE_DIFIX_CKPT_DIR="${REPO_ROOT}/outputs/official_difix3d/difix3d_gs/${SCENE_ID}/ckpts"

find_latest_ckpt() {
  local ckpt_dir="$1"
  local latest_path=""
  local latest_step=-1
  local path base step

  shopt -s nullglob
  for path in "${ckpt_dir}"/ckpt_*.pt; do
    base="$(basename "${path}")"
    step="${base#ckpt_}"
    step="${step%%_rank*}"
    step="${step%%.*}"
    if [[ "${step}" =~ ^[0-9]+$ ]] && (( step > latest_step )); then
      latest_step="${step}"
      latest_path="${path}"
    fi
  done
  shopt -u nullglob

  printf '%s' "${latest_path}"
}

setup_logging() {
  local timestamp host_name script_name
  timestamp="$(date -u +%Y%m%d-%H%M%S)"
  host_name="${HOSTNAME:-unknownhost}"
  script_name="$(basename "$0" .sh)"
  LOG_DIR="${LOG_ROOT}/3dgs_difix3dplus_ours/${SCENE_ID}"
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

if [[ ! -d "${GS_DATA_DIR}" ]]; then
  echo "Missing scene data: ${GS_DATA_DIR}" >&2
  exit 1
fi

if [[ ! -f "${TRAINER_PY}" ]]; then
  echo "Trainer not found: ${TRAINER_PY}" >&2
  exit 1
fi

if [[ ! -f "${CONFIG_JSON}" ]]; then
  echo "Config JSON not found: ${CONFIG_JSON}" >&2
  exit 1
fi

if [[ -f "${ACTIVATE_SCRIPT}" ]]; then
  # shellcheck disable=SC1090
  source "${ACTIVATE_SCRIPT}"
fi

install_runtime_deps() {
  echo "[setup] Checking Python runtime dependencies for ${PYTHON_BIN}"
  "${PYTHON_BIN}" - <<'PY'
import importlib.util
import subprocess
import sys

checks = [
    ("imageio", "imageio[ffmpeg]"),
    ("tyro", "tyro"),
    ("cv2", "opencv-python-headless"),
    ("pycolmap", "pycolmap"),
    ("torchmetrics", "torchmetrics"),
    ("matplotlib", "matplotlib"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("diffusers", "diffusers==0.25.1"),
    ("transformers", "transformers==4.38.0"),
    ("huggingface_hub", "huggingface-hub==0.25.1"),
    ("accelerate", "accelerate"),
    ("safetensors", "safetensors"),
    ("einops", "einops"),
    ("lpips", "lpips"),
    ("peft", "peft==0.9.0"),
    ("viser", "viser"),
    ("nerfview", "nerfview"),
    ("splines", "splines"),
    ("tensorly", "tensorly"),
    ("gsplat", "gsplat>=1.4.0"),
]

missing = [pkg for module, pkg in checks if importlib.util.find_spec(module) is None]
if not missing:
    print("[setup] All required packages already installed.")
    raise SystemExit(0)

print("[setup] Installing missing packages:")
for pkg in missing:
    print("  -", pkg)

cmd = [sys.executable, "-m", "pip", "install"] + missing
subprocess.check_call(cmd)
PY
}

install_build_deps() {
  echo "[setup] Checking system build toolchain"
  if command -v gcc >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1 && command -v ninja >/dev/null 2>&1 && command -v cmake >/dev/null 2>&1; then
    echo "[setup] gcc/g++/ninja/cmake already available."
    return
  fi

  if [[ "$(id -u)" != "0" ]]; then
    echo "[setup] Missing build tools and current user is not root." >&2
    echo "[setup] Run: apt-get update && apt-get install -y gcc g++ build-essential ninja-build cmake" >&2
    exit 1
  fi

  echo "[setup] Installing gcc/g++/ninja/cmake via apt-get"
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y gcc g++ build-essential ninja-build cmake
}

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[setup] Dry run: skipping dependency installation."
elif [[ "${INSTALL_DEPS}" == "1" ]]; then
  install_runtime_deps
else
  echo "[setup] Skipping dependency installation (INSTALL_DEPS=${INSTALL_DEPS})."
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[setup] Dry run: skipping build toolchain setup."
elif [[ "${INSTALL_BUILD_DEPS}" == "1" ]]; then
  install_build_deps
else
  echo "[setup] Skipping build toolchain setup (INSTALL_BUILD_DEPS=${INSTALL_BUILD_DEPS})."
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  if ! command -v gcc >/dev/null 2>&1 || ! command -v g++ >/dev/null 2>&1 || ! command -v ninja >/dev/null 2>&1 || ! command -v cmake >/dev/null 2>&1; then
    echo "[setup] Missing required build tools (gcc/g++/ninja/cmake)." >&2
    echo "[setup] Rerun with INSTALL_BUILD_DEPS=1 or prepare the persistent environment first." >&2
    exit 1
  fi
fi

RESUME_CKPT=""
if [[ -n "${START_CKPT}" ]]; then
  RESUME_CKPT="${START_CKPT}"
elif [[ "${FORCE_FROM_30000}" == "1" ]]; then
  if [[ "${ALLOW_REFERENCE_CKPT_FALLBACK}" == "1" && -f "${REFERENCE_VANILLA_CKPT}" ]]; then
    RESUME_CKPT="${REFERENCE_VANILLA_CKPT}"
  elif [[ "${ALLOW_REFERENCE_CKPT_FALLBACK}" == "1" && "${ALLOW_LEGACY_VANILLA_FALLBACK}" == "1" && -f "${REFERENCE_LEGACY_VANILLA_CKPT}" ]]; then
    RESUME_CKPT="${REFERENCE_LEGACY_VANILLA_CKPT}"
    echo "[warn] Reference vanilla ckpt not found; falling back to legacy 3dgs_full_baseline ckpt at 30k."
  else
    echo "Missing 30k vanilla checkpoint. Set START_CKPT, or set ALLOW_REFERENCE_CKPT_FALLBACK=1 to use stored comparison checkpoints." >&2
    echo "Checked:" >&2
    echo "  - ${REFERENCE_VANILLA_CKPT}" >&2
    echo "  - ${REFERENCE_LEGACY_VANILLA_CKPT}" >&2
    exit 1
  fi
else
  LATEST_OURS="$(find_latest_ckpt "${OURS_CKPT_DIR}")"
  if [[ -n "${LATEST_OURS}" ]]; then
    RESUME_CKPT="${LATEST_OURS}"
  elif [[ "${ALLOW_REFERENCE_CKPT_FALLBACK}" == "1" && -f "${REFERENCE_VANILLA_CKPT}" ]]; then
    RESUME_CKPT="${REFERENCE_VANILLA_CKPT}"
  elif [[ "${ALLOW_REFERENCE_CKPT_FALLBACK}" == "1" && "${ALLOW_LEGACY_VANILLA_FALLBACK}" == "1" && -f "${REFERENCE_LEGACY_VANILLA_CKPT}" ]]; then
    RESUME_CKPT="${REFERENCE_LEGACY_VANILLA_CKPT}"
    echo "[warn] Reference vanilla ckpt not found; falling back to legacy 3dgs_full_baseline ckpt at 30k."
  else
    LATEST_REFERENCE_DIFIX=""
    if [[ "${ALLOW_REFERENCE_CKPT_FALLBACK}" == "1" ]]; then
      LATEST_REFERENCE_DIFIX="$(find_latest_ckpt "${REFERENCE_DIFIX_CKPT_DIR}")"
    fi
    if [[ -n "${LATEST_REFERENCE_DIFIX}" ]]; then
      RESUME_CKPT="${LATEST_REFERENCE_DIFIX}"
    elif [[ "${START_FROM_SCRATCH}" == "1" ]]; then
      echo "[info] No local checkpoint found; starting from scratch."
    else
      echo "No checkpoint found to resume from." >&2
      echo "Set START_CKPT, START_FROM_SCRATCH=1, or ALLOW_REFERENCE_CKPT_FALLBACK=1." >&2
      exit 1
    fi
  fi
fi

if [[ -n "${RESUME_CKPT}" && ! -f "${RESUME_CKPT}" ]]; then
  echo "Resume checkpoint not found: ${RESUME_CKPT}" >&2
  exit 1
fi

if [[ -n "${RESUME_CKPT}" ]]; then
  RESUME_STEP="$("${PYTHON_BIN}" - "${RESUME_CKPT}" <<'PY'
import sys
import torch
ckpt = torch.load(sys.argv[1], map_location="cpu")
print(int(ckpt.get("step", -1)))
PY
)"
else
  RESUME_STEP="-1"
fi

if [[ "${MAX_STEPS}" -le "${RESUME_STEP}" ]]; then
  echo "[warn] MAX_STEPS=${MAX_STEPS} <= resume step ${RESUME_STEP}. Bumping MAX_STEPS to $((RESUME_STEP + 1))."
  MAX_STEPS="$((RESUME_STEP + 1))"
fi

mkdir -p "${OURS_OUT}"
export DIFFUSION_IBR_CACHE_DIR="${REPO_ROOT}/cache_weights"

echo "Scene: ${SCENE_ID}"
echo "Data: ${GS_DATA_DIR}"
echo "Config: ${CONFIG_JSON}"
echo "Resume ckpt: ${RESUME_CKPT:-<scratch>}"
echo "Resume step: ${RESUME_STEP}"
echo "Output (ours): ${OURS_OUT}"
echo "Max steps: ${MAX_STEPS}"
echo "Strict reference Difix schedule: ${STRICT_OFFICIAL_DIFIX}"
echo

EXTRA_ARGS=()

if [[ "${STRICT_OFFICIAL_DIFIX}" == "1" ]]; then
  # Force reference Difix3D schedule/behavior to avoid hidden env/config drift.
  EXTRA_ARGS+=(
    --training_recipe difix3d
    --use_difix
    --no-normalize_world
    --no-lazy_fixer_init
    --difix_start_step 3000
    --difix_fix_every 3000
    --fix_steps "${OFFICIAL_FIX_STEPS}"
    --eval_steps "${OFFICIAL_EVAL_STEPS}"
    --save_steps "${OFFICIAL_SAVE_STEPS}"
    --difix_num_views 0
    --difix_num_inference_steps 1
    --difix_timestep 199
    --difix_guidance_scale 0.0
    --difix_novel_prob 0.3
    --difix_novel_lambda 0.3
    --difix_progressive_updates
    --difix_progressive_pose_step 0.5
    --no-difix_post_render
    --difix_prompt "remove degradation"
  )
fi

if [[ "${ALLOW_RUNTIME_DIFIX_OVERRIDES}" == "1" ]]; then
  if [[ -n "${DIFIX_START_STEP:-}" ]]; then
    EXTRA_ARGS+=(--difix_start_step "${DIFIX_START_STEP}")
  fi
  if [[ -n "${DIFIX_FIX_EVERY:-}" ]]; then
    EXTRA_ARGS+=(--difix_fix_every "${DIFIX_FIX_EVERY}")
  fi
  if [[ -n "${FIX_STEPS:-}" ]]; then
    EXTRA_ARGS+=(--fix_steps "${FIX_STEPS}")
  fi
else
  if [[ -n "${DIFIX_START_STEP:-}" || -n "${DIFIX_FIX_EVERY:-}" || -n "${FIX_STEPS:-}" ]]; then
    echo "[warn] Ignoring DIFIX_* schedule env overrides because ALLOW_RUNTIME_DIFIX_OVERRIDES=0."
  fi
fi

if [[ "${STRICT_OFFICIAL_DIFIX}" == "1" ]]; then
  if [[ -n "${LAZY_FIXER_INIT:-}" ]]; then
    echo "[warn] Ignoring LAZY_FIXER_INIT because STRICT_OFFICIAL_DIFIX=1."
  fi
elif [[ "${LAZY_FIXER_INIT:-}" == "0" ]]; then
  EXTRA_ARGS+=(--no-lazy_fixer_init)
elif [[ "${LAZY_FIXER_INIT:-}" == "1" ]]; then
  EXTRA_ARGS+=(--lazy_fixer_init)
fi

CKPT_ARGS=()
if [[ -n "${RESUME_CKPT}" ]]; then
  CKPT_ARGS+=(--ckpt "${RESUME_CKPT}")
fi

CMD=(
  "${PYTHON_BIN}" "${TRAINER_PY}"
  "${CKPT_ARGS[@]}"
  --config "${CONFIG_JSON}"
  --data_dir "${GS_DATA_DIR}"
  --result_dir "${OURS_OUT}"
  --data_factor "${DATA_FACTOR}"
  --test_every "${TEST_EVERY}"
  --num_workers "${NUM_WORKERS}"
  --mode train
  --device cuda
  --max_steps "${MAX_STEPS}"
  --eval_every "${EVAL_EVERY}"
  --save_every "${SAVE_EVERY}"
  "${EXTRA_ARGS[@]}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[dry-run]'
  printf ' %q' CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${CMD[@]}"
  printf '\n'
  exit 0
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${CMD[@]}"
