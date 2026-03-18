#!/usr/bin/env bash
set -euo pipefail

# Official Difix3D gsplat flow on one DL3DV scene:
# 1. Train vanilla gsplat for 30k steps and save ckpt_29999_rank0.pt
# 2. Resume official Difix3D from the latest Difix checkpoint if present,
#    otherwise start from the vanilla checkpoint.

REPO_ROOT="/mntdatalora/src/Diffusion-IBR"
OFFICIAL_DIFIX3D_ROOT="${REPO_ROOT}/works/Difix3D"
DL3DV_ROOT="${REPO_ROOT}/data/DL3DV-10K-Benchmark"
CACHE_ROOT="${REPO_ROOT}/cache_weights"
ACTIVATE_SCRIPT="${REPO_ROOT}/execution_scripts/3dgs_difix3dplus/activate_persistent_difix3d_env.sh"
LOG_ROOT="${REPO_ROOT}/logs/execution"

SCENE_ID="${1:-032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7}"
DATA_FACTOR="${DATA_FACTOR:-4}"
TEST_EVERY="${TEST_EVERY:-8}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
INSTALL_BUILD_DEPS="${INSTALL_BUILD_DEPS:-1}"
INSTALL_XFORMERS="${INSTALL_XFORMERS:-0}"
SETUP_ONLY="${SETUP_ONLY:-0}"
SKIP_CUDA_PREFLIGHT="${SKIP_CUDA_PREFLIGHT:-0}"
FORCE_VANILLA="${FORCE_VANILLA:-0}"
FORCE_DIFIX_FROM_VANILLA="${FORCE_DIFIX_FROM_VANILLA:-0}"

SCENE_DIR="${DL3DV_ROOT}/${SCENE_ID}"
GS_DATA_DIR="${SCENE_DIR}/gaussian_splat"

VANILLA_OUT="${REPO_ROOT}/outputs/official_difix3d/vanilla_gs/${SCENE_ID}"
DIFIX3D_OUT="${REPO_ROOT}/outputs/official_difix3d/difix3d_gs/${SCENE_ID}"
VANILLA_CKPT="${VANILLA_OUT}/ckpts/ckpt_29999_rank0.pt"

find_latest_difix_ckpt() {
  local ckpt_dir="$1"
  local latest_path=""
  local latest_step=-1
  local path base step

  shopt -s nullglob
  for path in "${ckpt_dir}"/ckpt_*_rank0.pt; do
    base="$(basename "${path}")"
    step="${base#ckpt_}"
    step="${step%%_rank*}"
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
  LOG_DIR="${LOG_ROOT}/3dgs_difix3dplus/${SCENE_ID}"
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

if [[ -f "${ACTIVATE_SCRIPT}" ]]; then
  # shellcheck disable=SC1090
  source "${ACTIVATE_SCRIPT}"
fi

export PYTHONPATH="${OFFICIAL_DIFIX3D_ROOT}:${PYTHONPATH:-}"

install_runtime_deps() {
  echo "[setup] Checking Python runtime dependencies"
  "${PYTHON_BIN}" - "${INSTALL_XFORMERS}" <<'PY'
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

if sys.argv[1] == "1":
    checks.append(("xformers", "xformers"))

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

ensure_build_toolchain() {
  if command -v gcc >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1 && command -v ninja >/dev/null 2>&1 && command -v cmake >/dev/null 2>&1; then
    return
  fi

  echo "[setup] Missing build tools detected in this pod. Repairing automatically."
  install_build_deps
}

check_cuda_runtime() {
  echo "[setup] Checking CUDA / gsplat runtime"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" - <<'PY'
import sys

import torch
from gsplat.cuda import _backend

print(f"[setup] torch={torch.__version__} torch_cuda={torch.version.cuda}")
print(f"[setup] cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
print(f"[setup] gsplat_cuda_backend={'ok' if _backend._C is not None else 'missing'}")

errors = []
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    errors.append(
        "PyTorch cannot see a CUDA GPU in this container. "
        "Official gsplat Difix3D requires a visible NVIDIA GPU."
    )
if _backend._C is None:
    errors.append(
        "gsplat CUDA extension is unavailable. This usually means the CUDA toolkit/runtime "
        "is not fully available inside the environment."
    )

if errors:
    print("[setup] Cannot start official Difix3D:")
    for err in errors:
        print(f"  - {err}")
    print("[setup] Fix this first, then rerun the script.")
    sys.exit(2)
PY
}

cd "${OFFICIAL_DIFIX3D_ROOT}"

echo "Scene: ${SCENE_ID}"
echo "Data: ${GS_DATA_DIR}"
echo "Vanilla output: ${VANILLA_OUT}"
echo "Difix3D output: ${DIFIX3D_OUT}"
echo

install_runtime_deps
echo

if [[ "${INSTALL_BUILD_DEPS}" == "1" ]]; then
  install_build_deps
  echo
fi

ensure_build_toolchain
if [[ -f "${ACTIVATE_SCRIPT}" ]]; then
  # shellcheck disable=SC1090
  source "${ACTIVATE_SCRIPT}"
fi
echo

if [[ "${SKIP_CUDA_PREFLIGHT}" != "1" ]]; then
  echo "[setup] Using CUDA_DEVICE=${CUDA_DEVICE} for preflight and training"
  check_cuda_runtime
  echo
fi

if [[ "${SETUP_ONLY}" == "1" ]]; then
  echo "[setup] Dependency installation completed. Exiting because SETUP_ONLY=1."
  exit 0
fi

if [[ "${FORCE_VANILLA}" == "1" || ! -f "${VANILLA_CKPT}" ]]; then
  echo "[1/2] Train vanilla gsplat to the official 30k-step checkpoint"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" examples/gsplat/simple_trainer_vanilla.py default \
    --data_dir "${GS_DATA_DIR}" \
    --data_factor "${DATA_FACTOR}" \
    --result_dir "${VANILLA_OUT}" \
    --no-normalize-world-space \
    --test_every "${TEST_EVERY}" \
    --max_steps 30000
else
  echo "[1/2] Skipping vanilla stage because checkpoint already exists:"
  echo "      ${VANILLA_CKPT}"
fi

if [[ ! -f "${VANILLA_CKPT}" ]]; then
  echo "Expected checkpoint not found: ${VANILLA_CKPT}" >&2
  echo "Check ${VANILLA_OUT}/ckpts for the actual saved file." >&2
  exit 1
fi

RESUME_CKPT="${VANILLA_CKPT}"
LATEST_DIFIX_CKPT=""
if [[ "${FORCE_DIFIX_FROM_VANILLA}" != "1" ]]; then
  LATEST_DIFIX_CKPT="$(find_latest_difix_ckpt "${DIFIX3D_OUT}/ckpts")"
fi

echo
if [[ -n "${LATEST_DIFIX_CKPT}" ]]; then
  RESUME_CKPT="${LATEST_DIFIX_CKPT}"
  echo "[2/2] Resume official Difix3D training from latest Difix checkpoint:"
  echo "      ${RESUME_CKPT}"
else
  echo "[2/2] Start official Difix3D training from vanilla checkpoint:"
  echo "      ${RESUME_CKPT}"
fi

TRAIN_CMD=(
  "${PYTHON_BIN}"
  "examples/gsplat/simple_trainer_difix3d.py"
  "default"
  --data_dir "${GS_DATA_DIR}"
  --data_factor "${DATA_FACTOR}"
  --result_dir "${DIFIX3D_OUT}"
  --no-normalize-world-space
  --test_every "${TEST_EVERY}"
  --ckpt "${RESUME_CKPT}"
)
echo "[2/2] Launch command: ${TRAIN_CMD[*]}"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${TRAIN_CMD[@]}"
