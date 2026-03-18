#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mntdatalora/src/Diffusion-IBR}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/cache_weights}"
HF_CACHE_ROOT="${CACHE_ROOT}/huggingface"
MPL_CACHE_ROOT="${CACHE_ROOT}/matplotlib"
XDG_CACHE_ROOT="${CACHE_ROOT}/xdg"
TORCH_EXT_ROOT="${CACHE_ROOT}/torch_extensions"
PERSISTENT_CUDA_ROOT="${PERSISTENT_CUDA_ROOT:-${CACHE_ROOT}/persistent_cuda_clean}"
PERSISTENT_CUDA_HOME="${PERSISTENT_CUDA_HOME:-${PERSISTENT_CUDA_ROOT}/nvidia/cu13}"
ARCH_FILE="${ARCH_FILE:-${PERSISTENT_CUDA_ROOT}/torch_cuda_arch_list.txt}"

mkdir -p "${HF_CACHE_ROOT}/hub" "${MPL_CACHE_ROOT}" "${XDG_CACHE_ROOT}" "${TORCH_EXT_ROOT}"

export DIFFUSION_IBR_CACHE_DIR="${CACHE_ROOT}"
export HF_HOME="${HF_CACHE_ROOT}"
export HF_HUB_CACHE="${HF_CACHE_ROOT}/hub"
export TRANSFORMERS_CACHE="${HF_CACHE_ROOT}/hub"
export MPLCONFIGDIR="${MPL_CACHE_ROOT}"
export XDG_CACHE_HOME="${XDG_CACHE_ROOT}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXT_ROOT}"
export MAX_JOBS="${MAX_JOBS:-8}"

if command -v python >/dev/null 2>&1; then
  python "${REPO_ROOT}/execution_scripts/3dgs_difix3dplus/patch_gsplat_persistent_import.py" >/dev/null 2>&1 || true
fi

if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  if [[ -f "${ARCH_FILE}" ]]; then
    export TORCH_CUDA_ARCH_LIST
    TORCH_CUDA_ARCH_LIST="$(<"${ARCH_FILE}")"
  else
    export TORCH_CUDA_ARCH_LIST="8.6"
  fi
fi

if [[ -x "${PERSISTENT_CUDA_HOME}/bin/nvcc" ]]; then
  export CUDA_HOME="${PERSISTENT_CUDA_HOME}"
  if [[ -f "${CUDA_HOME}/lib/libcudart.so.13" && ! -e "${CUDA_HOME}/lib/libcudart.so" ]]; then
    ln -sf libcudart.so.13 "${CUDA_HOME}/lib/libcudart.so"
  fi
  if [[ -f "${CUDA_HOME}/lib/libnvvm.so.4" && ! -e "${CUDA_HOME}/lib/libnvvm.so" ]]; then
    ln -sf libnvvm.so.4 "${CUDA_HOME}/lib/libnvvm.so"
  fi
  export PATH="${CUDA_HOME}/bin:${CUDA_HOME}/nvvm/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib:/opt/conda/lib/python3.11/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
fi

if command -v gcc >/dev/null 2>&1; then
  export CC="${CC:-$(command -v gcc)}"
fi
if command -v g++ >/dev/null 2>&1; then
  export CXX="${CXX:-$(command -v g++)}"
  export CUDAHOSTCXX="${CUDAHOSTCXX:-${CXX}}"
fi

echo "[env] REPO_ROOT=${REPO_ROOT}"
echo "[env] TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}"
echo "[env] CUDA_HOME=${CUDA_HOME:-}"
echo "[env] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-}"
echo "[env] CC=${CC:-}"
echo "[env] CXX=${CXX:-}"
