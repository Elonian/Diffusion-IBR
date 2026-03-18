#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mntdatalora/src/Diffusion-IBR}"
DIFIX_ROOT="${REPO_ROOT}/outputs/official_difix3d/difix3d_gs"
FPS="${FPS:-30}"
OVERWRITE="${OVERWRITE:-0}"

if [[ $# -gt 0 ]]; then
  SCENES=("$@")
else
  SCENES=(
    "032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7"
    "0569e83fdc248a51fc0ab082ce5e2baff15755c53c207f545e6d02d91f01d166"
    "06da796666297fe4c683c231edf56ec00148a6a52ab5bb159fe1be31f53a58df"
    "073f5a9b983ced6fb28b23051260558b165f328a16b2d33fe20585b7ee4ad561"
    "07d9f9724ca854fae07cb4c57d7ea22bf667d5decd4058f547728922f909956b"
    "0853979305f7ecb80bd8fc2c8df916410d471ef04ed5f1a64e9651baa41d7695"
  )
fi

for SCENE_ID in "${SCENES[@]}"; do
  SCENE_DIR="${DIFIX_ROOT}/${SCENE_ID}"
  NOVEL_DIR="${SCENE_DIR}/renders/novel"

  if [[ ! -d "${NOVEL_DIR}" ]]; then
    echo "[skip] ${SCENE_ID}: missing ${NOVEL_DIR}"
    continue
  fi

  LATEST_STEP="$(python - "${NOVEL_DIR}" <<'PY'
import glob
import os
import sys

novel_dir = sys.argv[1]
steps = sorted([int(d) for d in os.listdir(novel_dir) if d.isdigit()])
best = ""
for step in steps:
    pred_dir = os.path.join(novel_dir, str(step), "Pred")
    fixed_dir = os.path.join(novel_dir, str(step), "Fixed")
    pred_count = len(glob.glob(os.path.join(pred_dir, "*.png"))) if os.path.isdir(pred_dir) else 0
    fixed_count = len(glob.glob(os.path.join(fixed_dir, "*.png"))) if os.path.isdir(fixed_dir) else 0
    if pred_count > 0 and pred_count == fixed_count:
        best = str(step)
print(best)
PY
)"
  if [[ -z "${LATEST_STEP}" ]]; then
    echo "[skip] ${SCENE_ID}: no numeric novel step dirs"
    continue
  fi

  PRED_DIR="${NOVEL_DIR}/${LATEST_STEP}/Pred"
  FIXED_DIR="${NOVEL_DIR}/${LATEST_STEP}/Fixed"
  if [[ ! -d "${PRED_DIR}" || ! -d "${FIXED_DIR}" ]]; then
    echo "[skip] ${SCENE_ID}: missing Pred/Fixed in chosen step ${LATEST_STEP}"
    continue
  fi

  VIDEO_DIR="${SCENE_DIR}/videos"
  mkdir -p "${VIDEO_DIR}"

  DIFIX_VIDEO="${VIDEO_DIR}/difix3d_step${LATEST_STEP}.mp4"
  DIFIX_PLUS_VIDEO="${VIDEO_DIR}/difix3dplus_step${LATEST_STEP}.mp4"
  COMP_VIDEO="${VIDEO_DIR}/difix3d_vs_difix3dplus_step${LATEST_STEP}.mp4"

  if [[ "${OVERWRITE}" != "1" && -f "${DIFIX_VIDEO}" && -f "${DIFIX_PLUS_VIDEO}" && -f "${COMP_VIDEO}" ]]; then
    echo "[skip] ${SCENE_ID}: all latest-step videos already exist (step ${LATEST_STEP})"
    continue
  fi

  echo "[scene] ${SCENE_ID}"
  echo "[step ] ${LATEST_STEP}"
  echo "[pred ] ${PRED_DIR}"
  echo "[fixed] ${FIXED_DIR}"
  echo "[out  ] ${VIDEO_DIR}"

  python - "${PRED_DIR}" "${FIXED_DIR}" "${DIFIX_VIDEO}" "${DIFIX_PLUS_VIDEO}" "${COMP_VIDEO}" "${FPS}" "${OVERWRITE}" <<'PY'
import glob
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

pred_dir = Path(sys.argv[1])
fixed_dir = Path(sys.argv[2])
difix_video = Path(sys.argv[3])
difix_plus_video = Path(sys.argv[4])
comp_video = Path(sys.argv[5])
fps = int(sys.argv[6])
overwrite = sys.argv[7] == "1"

pred_paths = sorted(glob.glob(str(pred_dir / "*.png")))
fixed_paths = sorted(glob.glob(str(fixed_dir / "*.png")))
frame_count = min(len(pred_paths), len(fixed_paths))

if frame_count == 0:
    raise SystemExit("no frames found in Pred/Fixed")

pred_paths = pred_paths[:frame_count]
fixed_paths = fixed_paths[:frame_count]

if not overwrite:
    missing_outputs = [p for p in (difix_video, difix_plus_video, comp_video) if not p.exists()]
    if not missing_outputs:
        print("[python] outputs already exist; skip")
        raise SystemExit(0)

def align_to_min(a: np.ndarray, b: np.ndarray):
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a = a[:h, :w, :3]
    b = b[:h, :w, :3]
    return a, b

writer_pred = imageio.get_writer(difix_video.as_posix(), fps=fps)
writer_fixed = imageio.get_writer(difix_plus_video.as_posix(), fps=fps)
writer_comp = imageio.get_writer(comp_video.as_posix(), fps=fps)

try:
    for p_pred, p_fixed in zip(pred_paths, fixed_paths):
        pred = imageio.imread(p_pred)
        fixed = imageio.imread(p_fixed)
        if pred.ndim == 2:
            pred = np.repeat(pred[..., None], 3, axis=2)
        if fixed.ndim == 2:
            fixed = np.repeat(fixed[..., None], 3, axis=2)

        pred, fixed = align_to_min(pred, fixed)
        sep = np.full((pred.shape[0], 16, 3), 255, dtype=np.uint8)

        writer_pred.append_data(pred.astype(np.uint8))
        writer_fixed.append_data(fixed.astype(np.uint8))
        writer_comp.append_data(np.concatenate([pred, sep, fixed], axis=1))
finally:
    writer_pred.close()
    writer_fixed.close()
    writer_comp.close()

print(f"[python] wrote {frame_count} frames")
print(f"[python] {difix_video}")
print(f"[python] {difix_plus_video}")
print(f"[python] {comp_video}")
PY
done
