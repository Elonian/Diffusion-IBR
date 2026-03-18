#!/usr/bin/env bash
set -euo pipefail

# Vanilla nerfacto baseline launcher (DL3DV, single scene).
# Uses scripts/trainers/nerfacto_vanilla_trainer.py and can bootstrap
# a dedicated persistent venv with nerfstudio installed.

REPO_ROOT="${REPO_ROOT:-/mntdatalora/src/Diffusion-IBR}"
TRAINER_SCRIPT="${REPO_ROOT}/scripts/trainers/nerfacto_vanilla_trainer.py"

SCENE_ID="${1:-032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7}"
DL3DV_ROOT="${DL3DV_ROOT:-${REPO_ROOT}/data/DL3DV-10K-Benchmark}"
SCENE_SOURCE="${SCENE_SOURCE:-auto}" # auto|nerfstudio|gaussian_splat

PYTHON_BIN="${PYTHON_BIN:-python}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/cache_weights/persistent_python_nerfacto_vanilla_v2}"

# Backward compatibility with old wrapper variables.
if [[ -n "${INSTALL_NERF_ENV:-}" ]]; then
  INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-${INSTALL_NERF_ENV}}"
else
  INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-1}"
fi

FORCE_REINSTALL_REQUIREMENTS="${FORCE_REINSTALL_REQUIREMENTS:-0}"
INSTALL_TCNN="${INSTALL_TCNN:-0}"
AUTO_FALLBACK_TORCH="${AUTO_FALLBACK_TORCH:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/official_nerfacto_full_baseline}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${SCENE_ID}}"
METHOD_NAME="${METHOD_NAME:-nerfacto}"
TIMESTAMP="${TIMESTAMP:-}" # empty => deterministic output folder (no timestamp component)
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/execution}"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
DEVICE_TYPE="${DEVICE_TYPE:-cuda}" # cpu|cuda|mps
NUM_DEVICES="${NUM_DEVICES:-1}"

BASELINE_TOTAL_STEPS="${BASELINE_TOTAL_STEPS:-60000}"
BASELINE_STEPS_PER_SAVE="${BASELINE_STEPS_PER_SAVE:-2000}"
BASELINE_STEPS_PER_EVAL_BATCH="${BASELINE_STEPS_PER_EVAL_BATCH:-500}"
NS_VIS="${NS_VIS:-viewer}"

DATA_FACTOR="${DATA_FACTOR:-4}"
EVAL_MODE="${EVAL_MODE:-interval}" # fraction|filename|interval|all
EVAL_INTERVAL="${EVAL_INTERVAL:-8}"
TRAIN_SPLIT_FRACTION="${TRAIN_SPLIT_FRACTION:-0.9}"
TRAIN_NUM_RAYS_PER_BATCH="${TRAIN_NUM_RAYS_PER_BATCH:-4096}"
EVAL_NUM_RAYS_PER_BATCH="${EVAL_NUM_RAYS_PER_BATCH:-4096}"
EVAL_NUM_RAYS_PER_CHUNK="${EVAL_NUM_RAYS_PER_CHUNK:-32768}"
AVERAGE_INIT_DENSITY="${AVERAGE_INIT_DENSITY:-0.01}"
CAMERA_OPTIMIZER_MODE="${CAMERA_OPTIMIZER_MODE:-SO3xR3}" # off|SO3xR3|SE3
NERFACTO_IMPLEMENTATION="${NERFACTO_IMPLEMENTATION:-torch}" # torch|tcnn

PREFER_PARALLEL_DATAMANAGER="${PREFER_PARALLEL_DATAMANAGER:-0}"
DISABLE_MIXED_PRECISION="${DISABLE_MIXED_PRECISION:-0}"
SAVE_ALL_CHECKPOINTS="${SAVE_ALL_CHECKPOINTS:-0}"
DRY_RUN="${DRY_RUN:-0}"
SETUP_LOCK_FILE="${SETUP_LOCK_FILE:-${REPO_ROOT}/cache_weights/locks/nerfacto_full_baseline_setup.lock}"
SETUP_LOCK_WAIT_SECONDS="${SETUP_LOCK_WAIT_SECONDS:-1800}"

setup_logging() {
  local timestamp host_name script_name

  timestamp="$(date -u +%Y%m%d-%H%M%S)"
  host_name="${HOSTNAME:-unknownhost}"
  script_name="$(basename "$0" .sh)"
  LOG_DIR="${LOG_ROOT}/nerfacto_full_baseline/${SCENE_ID}"
  LOG_FILE="${LOG_DIR}/${timestamp}_${host_name}_${script_name}.log"

  mkdir -p "${LOG_DIR}"
  ln -sfn "$(basename "${LOG_FILE}")" "${LOG_DIR}/latest.log" 2>/dev/null || true

  if [[ -z "${__DIFFUSION_IBR_SCRIPT_LOGGING:-}" ]]; then
    __DIFFUSION_IBR_SCRIPT_LOGGING=1
    exec > >(tee -a "${LOG_FILE}") 2>&1
  fi

  echo "[log] Writing stdout/stderr to ${LOG_FILE}"
}

module_importable() {
  local python_bin="$1"
  local module_name="$2"
  "${python_bin}" - "$module_name" <<'PY'
import importlib
import sys

name = sys.argv[1]
try:
    importlib.import_module(name)
except Exception:
    raise SystemExit(1)
PY
}

create_or_reuse_venv() {
  local recreate=0

  if [[ -x "${VENV_DIR}/bin/python" && -f "${VENV_DIR}/pyvenv.cfg" ]]; then
    if grep -q '^include-system-site-packages = false' "${VENV_DIR}/pyvenv.cfg"; then
      recreate=1
    fi
  fi

  if [[ ! -x "${VENV_DIR}/bin/python" || "${recreate}" == "1" ]]; then
    if [[ "${recreate}" == "1" ]]; then
      echo "[setup] Recreating venv with --system-site-packages: ${VENV_DIR}"
      rm -rf "${VENV_DIR}"
    else
      echo "[setup] Creating venv: ${VENV_DIR}"
    fi
    "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
  else
    echo "[setup] Reusing venv: ${VENV_DIR}"
  fi
}

ensure_working_pip() {
  local vpy="$1"

  if "${vpy}" -m pip --version >/dev/null 2>&1; then
    return
  fi

  echo "[setup] Detected broken pip in ${VENV_DIR}. Trying ensurepip repair."
  "${vpy}" -m ensurepip --upgrade || true
  if "${vpy}" -m pip --version >/dev/null 2>&1; then
    return
  fi

  echo "[setup] ensurepip repair failed; recreating venv ${VENV_DIR}."
  rm -rf "${VENV_DIR}"
  create_or_reuse_venv

  if ! "${VENV_DIR}/bin/python" -m pip --version >/dev/null 2>&1; then
    echo "[setup] pip is still unavailable after venv recreation." >&2
    exit 2
  fi
}

install_requirements_if_needed() {
  local vpy="$1"

  if [[ "${INSTALL_REQUIREMENTS}" != "1" ]]; then
    if module_importable "${vpy}" "nerfstudio"; then
      echo "[setup] INSTALL_REQUIREMENTS=${INSTALL_REQUIREMENTS}; nerfstudio already present, skipping pip install."
      return
    fi
    echo "[setup] INSTALL_REQUIREMENTS=${INSTALL_REQUIREMENTS}, but nerfstudio is missing."
    echo "[setup] Proceeding with one-time install to repair runtime."
  fi

  if [[ "${FORCE_REINSTALL_REQUIREMENTS}" == "1" ]] || ! module_importable "${vpy}" "nerfstudio"; then
    echo "[setup] Installing/refreshing nerfstudio in venv."
    # Keep installation lightweight and avoid replacing the existing torch stack.
    "${vpy}" -m pip install --disable-pip-version-check --upgrade --no-deps "nerfstudio>=1.1.0,<1.2.0"
  else
    echo "[setup] nerfstudio already importable in venv."
  fi

  echo "[setup] Verifying Nerfstudio runtime dependency imports."
  ensure_nerfstudio_runtime_deps "${vpy}"
}

probe_nerfstudio_runtime_imports() {
  local vpy="$1"
  "${vpy}" - <<'PY'
import importlib
import sys

modules = [
    "nerfstudio.engine.trainer",
    "nerfstudio.configs.base_config",
    "nerfstudio.cameras.camera_optimizers",
    "nerfstudio.data.dataparsers.nerfstudio_dataparser",
    "nerfstudio.pipelines.base_pipeline",
    "nerfstudio.models.nerfacto",
    "nerfstudio.engine.optimizers",
    "nerfstudio.engine.schedulers",
]

for module_name in modules:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        print(exc.name)
        raise SystemExit(2)
    except Exception as exc:
        print(f"EXCEPTION:{type(exc).__name__}:{exc}")
        raise SystemExit(3)

print("OK")
PY
}

missing_module_to_package() {
  local module_name="$1"
  case "${module_name}" in
    appdirs) echo "appdirs" ;;
    av) echo "av>=9.2.0" ;;
    comet_ml) echo "comet-ml>=3.33.8" ;;
    cryptography) echo "cryptography>=38" ;;
    cv2) echo "opencv-python-headless==4.10.0.84" ;;
    fpsample) echo "fpsample" ;;
    gdown) echo "gdown>=4.6.0" ;;
    gsplat) echo "gsplat==1.4.0" ;;
    h5py) echo "h5py>=2.9.0" ;;
    imageio) echo "imageio>=2.21.1" ;;
    jaxtyping) echo "jaxtyping>=0.2.15" ;;
    mediapy) echo "mediapy>=1.1.0" ;;
    msgpack_numpy) echo "msgpack-numpy>=0.4.8" ;;
    nerfacc) echo "nerfacc==0.5.2" ;;
    nerfacc_cuda) echo "nerfacc==0.5.2" ;;
    newrawpy) echo "newrawpy>=1.0.0b0" ;;
    open3d) echo "open3d>=0.16.0" ;;
    packaging) echo "packaging" ;;
    pathos) echo "pathos" ;;
    PIL) echo "Pillow>=10.3.0" ;;
    plotly) echo "plotly>=5.7.0" ;;
    protobuf) echo "protobuf!=3.20.0,<=3.20.3" ;;
    pyngrok) echo "pyngrok>=5.1.0" ;;
    pyquaternion) echo "pyquaternion>=0.9.9" ;;
    pymeshlab) echo "pymeshlab>=2022.2.post2" ;;
    pytorch_msssim) echo "pytorch-msssim" ;;
    rawpy) echo "rawpy>=0.18.1" ;;
    rich) echo "rich>=12.5.1" ;;
    skimage) echo "scikit-image>=0.19.3" ;;
    socketio) echo "python-socketio>=5.7.1" ;;
    splines) echo "splines==0.3.0" ;;
    tensorboard) echo "tensorboard>=2.13.0" ;;
    tensorly) echo "tensorly" ;;
    timm) echo "timm==0.6.7" ;;
    torchmetrics) echo "torchmetrics[image]>=1.0.1" ;;
    trimesh) echo "trimesh>=3.20.2" ;;
    tyro) echo "tyro>=0.6.6" ;;
    viser) echo "viser==0.2.7" ;;
    wandb) echo "wandb>=0.13.3" ;;
    xatlas) echo "xatlas" ;;
    yaml) echo "PyYAML" ;;
    *) return 1 ;;
  esac
}

ensure_nerfstudio_runtime_deps() {
  local vpy="$1"
  local attempt result missing package_spec

  for attempt in $(seq 1 40); do
    result="$(probe_nerfstudio_runtime_imports "${vpy}" || true)"
    result="${result##*$'\n'}"

    if [[ "${result}" == "OK" ]]; then
      echo "[setup] Nerfstudio runtime imports are ready."
      return 0
    fi

    if [[ "${result}" == EXCEPTION:* ]]; then
      echo "[setup] Nerfstudio runtime probe failed: ${result}" >&2
      return 1
    fi

    missing="${result}"
    if ! package_spec="$(missing_module_to_package "${missing}")"; then
      echo "[setup] Missing module '${missing}' has no install mapping." >&2
      echo "[setup] Please install it in ${VENV_DIR} manually and retry." >&2
      return 1
    fi

    echo "[setup] Installing missing runtime dependency: ${package_spec} (for module ${missing})"
    "${vpy}" -m pip install --disable-pip-version-check --upgrade "${package_spec}"
  done

  echo "[setup] Runtime dependency probe exceeded max attempts." >&2
  return 1
}

maybe_enable_tcnn() {
  local vpy="$1"

  if [[ "${NERFACTO_IMPLEMENTATION}" != "tcnn" ]]; then
    return
  fi

  if module_importable "${vpy}" "tinycudann"; then
    echo "[setup] tinycudann is available."
    return
  fi

  if [[ "${INSTALL_TCNN}" == "1" ]]; then
    echo "[setup] tinycudann missing; attempting install."
    set +e
    "${vpy}" -m pip install --upgrade tinycudann
    rc=$?
    set -e
    if [[ "${rc}" -ne 0 ]]; then
      echo "[setup] tinycudann install failed (exit ${rc})."
    fi
  fi

  if module_importable "${vpy}" "tinycudann"; then
    echo "[setup] tinycudann is available after install."
    return
  fi

  if [[ "${AUTO_FALLBACK_TORCH}" == "1" ]]; then
    echo "[setup] tinycudann unavailable; falling back to torch implementation."
    NERFACTO_IMPLEMENTATION="torch"
  else
    echo "[setup] tinycudann is required for tcnn implementation but unavailable." >&2
    exit 3
  fi
}

setup_logging

if [[ ! -f "${TRAINER_SCRIPT}" ]]; then
  echo "Missing trainer script: ${TRAINER_SCRIPT}" >&2
  exit 1
fi

SCENE_ROOT="${DL3DV_ROOT}/${SCENE_ID}"
if [[ ! -d "${SCENE_ROOT}" ]]; then
  echo "Missing DL3DV scene folder: ${SCENE_ROOT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/${EXPERIMENT_NAME}"

if [[ "${DRY_RUN}" == "1" && ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "[setup] DRY_RUN=1 and venv is missing; skipping venv creation."
  VENV_PYTHON="${PYTHON_BIN}"
else
  if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "$(dirname "${SETUP_LOCK_FILE}")"
    exec 200>"${SETUP_LOCK_FILE}"
    echo "[setup] Waiting for setup lock: ${SETUP_LOCK_FILE} (timeout=${SETUP_LOCK_WAIT_SECONDS}s)"
    if ! flock -x -w "${SETUP_LOCK_WAIT_SECONDS}" 200; then
      echo "[setup] Timed out waiting for setup lock after ${SETUP_LOCK_WAIT_SECONDS}s: ${SETUP_LOCK_FILE}" >&2
      exit 1
    fi
    echo "[setup] Acquired setup lock: ${SETUP_LOCK_FILE}"
  fi
  create_or_reuse_venv
  VENV_PYTHON="${VENV_DIR}/bin/python"
  if [[ "${DRY_RUN}" != "1" ]]; then
    ensure_working_pip "${VENV_PYTHON}"
    VENV_PYTHON="${VENV_DIR}/bin/python"
  fi
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  install_requirements_if_needed "${VENV_PYTHON}"

  if ! module_importable "${VENV_PYTHON}" "nerfstudio"; then
    echo "[setup] nerfstudio still not importable in ${VENV_DIR}." >&2
    echo "[setup] Set INSTALL_REQUIREMENTS=1 (default) or inspect pip logs above." >&2
    exit 2
  fi

  maybe_enable_tcnn "${VENV_PYTHON}"
fi

CMD=(
  "${VENV_PYTHON}" "${TRAINER_SCRIPT}"
  --scene_id "${SCENE_ID}"
  --dl3dv_root "${DL3DV_ROOT}"
  --scene_source "${SCENE_SOURCE}"
  --output_dir "${OUTPUT_ROOT}"
  --experiment_name "${EXPERIMENT_NAME}"
  --method_name "${METHOD_NAME}"
  --timestamp "${TIMESTAMP}"
  --device_type "${DEVICE_TYPE}"
  --num_devices "${NUM_DEVICES}"
  --max_num_iterations "${BASELINE_TOTAL_STEPS}"
  --steps_per_save "${BASELINE_STEPS_PER_SAVE}"
  --steps_per_eval_batch "${BASELINE_STEPS_PER_EVAL_BATCH}"
  --vis "${NS_VIS}"
  --downscale_factor "${DATA_FACTOR}"
  --eval_mode "${EVAL_MODE}"
  --eval_interval "${EVAL_INTERVAL}"
  --train_split_fraction "${TRAIN_SPLIT_FRACTION}"
  --train_num_rays_per_batch "${TRAIN_NUM_RAYS_PER_BATCH}"
  --eval_num_rays_per_batch "${EVAL_NUM_RAYS_PER_BATCH}"
  --eval_num_rays_per_chunk "${EVAL_NUM_RAYS_PER_CHUNK}"
  --average_init_density "${AVERAGE_INIT_DENSITY}"
  --camera_optimizer_mode "${CAMERA_OPTIMIZER_MODE}"
  --implementation "${NERFACTO_IMPLEMENTATION}"
)

if [[ "${PREFER_PARALLEL_DATAMANAGER}" == "1" ]]; then
  CMD+=(--prefer_parallel_datamanager)
fi
if [[ "${DISABLE_MIXED_PRECISION}" == "1" ]]; then
  CMD+=(--disable_mixed_precision)
fi
if [[ "${SAVE_ALL_CHECKPOINTS}" == "1" ]]; then
  CMD+=(--save_all_checkpoints)
fi

echo "Scene: ${SCENE_ID}"
echo "Scene root: ${SCENE_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Implementation: ${NERFACTO_IMPLEMENTATION}"
echo "Python: ${VENV_PYTHON}"
echo
echo "[run] ${CMD[*]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[run] DRY_RUN=1, skipping execution."
  exit 0
fi

if [[ "${DEVICE_TYPE}" == "cuda" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${CMD[@]}"
else
  "${CMD[@]}"
fi
