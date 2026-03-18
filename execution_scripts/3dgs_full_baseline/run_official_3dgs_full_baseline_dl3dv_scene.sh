#!/usr/bin/env bash
set -euo pipefail

# Pure 3DGS baseline flow on one DL3DV scene:
# - Train official vanilla gsplat to a configurable total step budget (default: 60k).
# - Store outputs separately from Difix runs for direct comparison.

REPO_ROOT="${REPO_ROOT:-/mntdatalora/src/Diffusion-IBR}"
OFFICIAL_DIFIX3D_ROOT="${REPO_ROOT}/works/Difix3D"
DL3DV_ROOT="${REPO_ROOT}/data/DL3DV-10K-Benchmark"
ACTIVATE_SCRIPT="${REPO_ROOT}/execution_scripts/3dgs_difix3dplus/activate_persistent_difix3d_env.sh"
LOG_ROOT="${REPO_ROOT}/logs/execution"
PERSISTENT_PYTHON_SITE_PACKAGES="${PERSISTENT_PYTHON_SITE_PACKAGES:-${REPO_ROOT}/cache_weights/persistent_python_freefix}"

SCENE_ID="${1:-032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7}"
DATA_FACTOR="${DATA_FACTOR:-4}"
TEST_EVERY="${TEST_EVERY:-8}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
INSTALL_BUILD_DEPS="${INSTALL_BUILD_DEPS:-1}"
SETUP_ONLY="${SETUP_ONLY:-0}"
SKIP_CUDA_PREFLIGHT="${SKIP_CUDA_PREFLIGHT:-0}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
BASELINE_MAX_STEPS="${BASELINE_MAX_STEPS:-60000}"
SKIP_ACTIVATE_SCRIPT="${SKIP_ACTIVATE_SCRIPT:-0}"

SCENE_DIR="${DL3DV_ROOT}/${SCENE_ID}"
GS_DATA_DIR="${SCENE_DIR}/gaussian_splat"
BASELINE_OUT="${REPO_ROOT}/outputs/official_3dgs_full_baseline/${SCENE_ID}"
FINAL_CKPT="${BASELINE_OUT}/ckpts/ckpt_$((BASELINE_MAX_STEPS - 1))_rank0.pt"

setup_logging() {
  local timestamp host_name script_name

  timestamp="$(date -u +%Y%m%d-%H%M%S)"
  host_name="${HOSTNAME:-unknownhost}"
  script_name="$(basename "$0" .sh)"
  LOG_DIR="${LOG_ROOT}/3dgs_full_baseline/${SCENE_ID}"
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

if [[ "${SKIP_ACTIVATE_SCRIPT}" != "1" && -f "${ACTIVATE_SCRIPT}" ]]; then
  # shellcheck disable=SC1090
  source "${ACTIVATE_SCRIPT}"
fi

if [[ -d "${PERSISTENT_PYTHON_SITE_PACKAGES}" ]]; then
  export PYTHONPATH="${PERSISTENT_PYTHON_SITE_PACKAGES}:${OFFICIAL_DIFIX3D_ROOT}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${OFFICIAL_DIFIX3D_ROOT}:${PYTHONPATH:-}"
fi

install_runtime_deps() {
  echo "[setup] Checking Python runtime dependencies"
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
    ("yaml", "pyyaml"),
    ("tensorboard", "tensorboard"),
    ("viser", "viser"),
    ("nerfview", "nerfview"),
    ("splines", "splines"),
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

ensure_build_toolchain() {
  if command -v gcc >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1 && command -v ninja >/dev/null 2>&1 && command -v cmake >/dev/null 2>&1; then
    return
  fi

  echo "[setup] Missing build tools detected in this pod. Repairing automatically."
  install_build_deps
}

check_cuda_runtime() {
  echo "[setup] Checking CUDA / gsplat runtime"
  "${PYTHON_BIN}" - <<'PY'
import sys

import torch
from gsplat.cuda import _backend

print(f"[setup] torch={torch.__version__} torch_cuda={torch.version.cuda}")
print(f"[setup] cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
print(f"[setup] gsplat_cuda_backend={'ok' if _backend._C is not None else 'missing'}")

errors = []
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    errors.append(
        "PyTorch cannot see a CUDA GPU in this container. Official gsplat training requires a visible NVIDIA GPU."
    )
if _backend._C is None:
    errors.append(
        "gsplat CUDA extension is unavailable. This usually means the CUDA toolkit/runtime is not fully available."
    )

if errors:
    print("[setup] Cannot start official 3DGS baseline:")
    for err in errors:
        print(f"  - {err}")
    print("[setup] Fix this first, then rerun the script.")
    sys.exit(2)
PY
}

cd "${OFFICIAL_DIFIX3D_ROOT}"

echo "Scene: ${SCENE_ID}"
echo "Data: ${GS_DATA_DIR}"
echo "Baseline output: ${BASELINE_OUT}"
echo "Target steps: ${BASELINE_MAX_STEPS}"
echo

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  install_runtime_deps
  echo
else
  echo "[setup] Skipping Python runtime dependency installation (INSTALL_DEPS=${INSTALL_DEPS})."
  echo
fi

if [[ "${INSTALL_BUILD_DEPS}" == "1" ]]; then
  install_build_deps
  ensure_build_toolchain
  echo
else
  echo "[setup] Skipping build toolchain setup (INSTALL_BUILD_DEPS=${INSTALL_BUILD_DEPS})."
  echo
fi

if [[ "${SKIP_ACTIVATE_SCRIPT}" != "1" && -f "${ACTIVATE_SCRIPT}" ]]; then
  # shellcheck disable=SC1090
  source "${ACTIVATE_SCRIPT}"
fi
echo

if [[ "${SKIP_CUDA_PREFLIGHT}" != "1" ]]; then
  check_cuda_runtime
  echo
fi

if [[ "${SETUP_ONLY}" == "1" ]]; then
  echo "[setup] Dependency installation completed. Exiting because SETUP_ONLY=1."
  exit 0
fi

if [[ "${FORCE_RETRAIN}" == "1" || ! -f "${FINAL_CKPT}" ]]; then
  echo "[1/1] Train official vanilla gsplat baseline"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" examples/gsplat/simple_trainer_vanilla.py default \
    --data_dir "${GS_DATA_DIR}" \
    --data_factor "${DATA_FACTOR}" \
    --result_dir "${BASELINE_OUT}" \
    --no-normalize-world-space \
    --test_every "${TEST_EVERY}" \
    --max_steps "${BASELINE_MAX_STEPS}" \
    --eval_steps 7000 30000 "${BASELINE_MAX_STEPS}" \
    --save_steps 7000 30000 "${BASELINE_MAX_STEPS}"
else
  echo "[1/1] Skipping training because final baseline checkpoint already exists:"
  echo "      ${FINAL_CKPT}"
fi

if [[ ! -f "${FINAL_CKPT}" ]]; then
  echo "Expected final baseline checkpoint not found: ${FINAL_CKPT}" >&2
  echo "Note: vanilla trainer does not support in-script resume from partial checkpoints." >&2
  exit 1
fi

echo
echo "[done] Baseline checkpoint ready:"
echo "      ${FINAL_CKPT}"
