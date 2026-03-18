#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mntdatalora/src/Diffusion-IBR}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/cache_weights}"
PERSISTENT_CUDA_ROOT="${PERSISTENT_CUDA_ROOT:-${CACHE_ROOT}/persistent_cuda_clean}"
PERSISTENT_CUDA_HOME="${PERSISTENT_CUDA_HOME:-${PERSISTENT_CUDA_ROOT}/nvidia/cu13}"
ARCH_FILE="${ARCH_FILE:-${PERSISTENT_CUDA_ROOT}/torch_cuda_arch_list.txt}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INSTALL_RUNTIME_DEPS="${INSTALL_RUNTIME_DEPS:-1}"
INSTALL_BUILD_DEPS="${INSTALL_BUILD_DEPS:-1}"
FORCE_REINSTALL_CUDA="${FORCE_REINSTALL_CUDA:-0}"
MAX_JOBS="${MAX_JOBS:-8}"
LOG_ROOT="${REPO_ROOT}/logs/execution"

setup_logging() {
  local timestamp host_name script_name

  timestamp="$(date -u +%Y%m%d-%H%M%S)"
  host_name="${HOSTNAME:-unknownhost}"
  script_name="$(basename "$0" .sh)"
  LOG_DIR="${LOG_ROOT}/3dgs_difix3dplus/setup"
  LOG_FILE="${LOG_DIR}/${timestamp}_${host_name}_${script_name}.log"

  mkdir -p "${LOG_DIR}"
  ln -sfn "$(basename "${LOG_FILE}")" "${LOG_DIR}/latest.log" 2>/dev/null || true

  if [[ -z "${__DIFFUSION_IBR_SCRIPT_LOGGING:-}" ]]; then
    __DIFFUSION_IBR_SCRIPT_LOGGING=1
    exec > >(tee -a "${LOG_FILE}") 2>&1
  fi

  echo "[log] Writing stdout/stderr to ${LOG_FILE}"
}

mkdir -p "${CACHE_ROOT}"

setup_logging

if [[ "${FORCE_REINSTALL_CUDA}" == "1" ]]; then
  rm -rf "${PERSISTENT_CUDA_ROOT}"
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

subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
PY
}

install_build_deps() {
  echo "[setup] Checking system build toolchain"
  if command -v gcc >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1 && command -v ninja >/dev/null 2>&1; then
    echo "[setup] gcc/g++/ninja already available."
    return
  fi

  if [[ "$(id -u)" != "0" ]]; then
    echo "[setup] Missing gcc/g++/ninja and current user is not root." >&2
    echo "[setup] Run: apt-get update && apt-get install -y gcc g++ build-essential ninja-build" >&2
    exit 1
  fi

  echo "[setup] Installing gcc/g++/ninja via apt-get"
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y gcc g++ build-essential ninja-build
}

install_persistent_cuda_toolkit() {
  if [[ -x "${PERSISTENT_CUDA_HOME}/bin/nvcc" ]]; then
    echo "[setup] Persistent CUDA toolkit already present at ${PERSISTENT_CUDA_HOME}"
    return
  fi

  echo "[setup] Installing persistent CUDA toolkit to ${PERSISTENT_CUDA_ROOT}"
  mkdir -p "${PERSISTENT_CUDA_ROOT}"
  "${PYTHON_BIN}" -m pip install --upgrade --target "${PERSISTENT_CUDA_ROOT}" \
    nvidia-cuda-nvcc==13.1.80 \
    nvidia-cuda-runtime==13.1.80 \
    nvidia-cuda-crt==13.1.80 \
    nvidia-nvvm==13.1.80 \
    nvidia-cuda-cccl==13.1.78
}

activate_persistent_env() {
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/execution_scripts/3dgs_difix3dplus/activate_persistent_difix3d_env.sh"
}

detect_arch() {
  if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    printf '%s\n' "${TORCH_CUDA_ARCH_LIST}" > "${ARCH_FILE}"
    return
  fi

  TORCH_CUDA_ARCH_LIST="$("${PYTHON_BIN}" - <<'PY'
import torch
if not torch.cuda.is_available():
    print("8.6")
else:
    major, minor = torch.cuda.get_device_capability(0)
    print(f"{major}.{minor}")
PY
)"
  export TORCH_CUDA_ARCH_LIST
  printf '%s\n' "${TORCH_CUDA_ARCH_LIST}" > "${ARCH_FILE}"
}

warmup_gsplat() {
  echo "[setup] Warming up persistent gsplat CUDA extension"
  detect_arch
  export MAX_JOBS
  "${PYTHON_BIN}" - <<'PY'
import os
import torch

print("[setup] torch", torch.__version__, "torch_cuda", torch.version.cuda)
print("[setup] cuda_available", torch.cuda.is_available(), "device_count", torch.cuda.device_count())
print("[setup] CUDA_HOME", os.environ.get("CUDA_HOME"))
print("[setup] TORCH_CUDA_ARCH_LIST", os.environ.get("TORCH_CUDA_ARCH_LIST"))
print("[setup] TORCH_EXTENSIONS_DIR", os.environ.get("TORCH_EXTENSIONS_DIR"))

from gsplat.cuda import _backend

print("[setup] gsplat backend", _backend._C)
if _backend._C is None:
    raise SystemExit("gsplat CUDA backend is still missing after setup")
PY
}

if [[ "${INSTALL_RUNTIME_DEPS}" == "1" ]]; then
  install_runtime_deps
fi

install_persistent_cuda_toolkit

if [[ "${INSTALL_BUILD_DEPS}" == "1" ]]; then
  install_build_deps
fi

activate_persistent_env
warmup_gsplat

echo
echo "[setup] Persistent Difix3D CUDA environment is ready."
echo "[setup] Future pods only need:"
echo "source ${REPO_ROOT}/execution_scripts/3dgs_difix3dplus/activate_persistent_difix3d_env.sh"
