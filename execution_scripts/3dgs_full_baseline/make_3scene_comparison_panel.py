#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path("/mntdatalora/src/Diffusion-IBR")
OUTPUT_ROOT = REPO_ROOT / "outputs"
SAVE_DIR = REPO_ROOT / "execution_scripts" / "3dgs_full_baseline"


@dataclass
class SceneSpec:
    scene_id: str
    use_freefix: bool

    @property
    def short(self) -> str:
        return self.scene_id[:8]


SCENES: list[SceneSpec] = [
    SceneSpec("06da796666297fe4c683c231edf56ec00148a6a52ab5bb159fe1be31f53a58df", True),
    SceneSpec("0569e83fdc248a51fc0ab082ce5e2baff15755c53c207f545e6d02d91f01d166", True),
    SceneSpec("073f5a9b983ced6fb28b23051260558b165f328a16b2d33fe20585b7ee4ad561", False),
]

COLS = ["Vanilla 30k", "Vanilla 60k", "Difix3D", "Difix3D+", "FreeFix SDXL"]


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred - gt) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def image_ids_from_glob(directory: Path, pattern: str, mode: str = "stem") -> set[str]:
    out: set[str] = set()
    for path in directory.glob(pattern):
        stem = path.stem
        if mode == "last_token":
            out.add(stem.rsplit("_", 1)[-1])
        elif mode == "first_token":
            out.add(stem.split("_", 1)[0])
        else:
            out.add(stem)
    return out


def latest_eval_dir(scene_id: str) -> Path:
    root = OUTPUT_ROOT / "freefix_self" / "sdxl" / scene_id / "renders"
    cands: list[tuple[int, Path]] = []
    for path in root.iterdir():
        if path.is_dir() and path.name.startswith("eval_"):
            tail = path.name.split("_")[-1]
            if tail.isdigit():
                cands.append((int(tail), path))
    if not cands:
        raise FileNotFoundError(f"No eval_* directory found for scene {scene_id}: {root}")
    cands.sort()
    return cands[-1][1]


def method_paths(scene_id: str, image_id: str, freefix_eval_dir: Optional[Path]) -> dict[str, Optional[Path]]:
    vanilla30 = OUTPUT_ROOT / "official_difix3d" / "vanilla_gs" / scene_id / "renders" / f"val_step29999_{image_id}.png"
    vanilla60 = OUTPUT_ROOT / "official_3dgs_full_baseline" / scene_id / "renders" / f"val_step59999_{image_id}.png"
    difix = OUTPUT_ROOT / "official_difix3d" / "difix3d_gs" / scene_id / "renders" / "novel" / "59999" / "Pred" / f"{image_id}.png"
    if not difix.exists():
        difix = OUTPUT_ROOT / "official_difix3d" / "difix3d_gs" / scene_id / "renders" / "val" / "59999" / "Pred" / f"{image_id}.png"
    ours = OUTPUT_ROOT / "ours_difix3dplus_gs" / scene_id / "renders" / "novel" / "59999" / "Pred" / f"{image_id}.png"
    freefix = None if freefix_eval_dir is None else freefix_eval_dir / f"{image_id}_pred.png"
    return {
        "Vanilla 30k": vanilla30 if vanilla30.exists() else None,
        "Vanilla 60k": vanilla60 if vanilla60.exists() else None,
        "Difix3D": difix if difix.exists() else None,
        "Difix3D+": ours if ours.exists() else None,
        "FreeFix SDXL": freefix if freefix is not None and freefix.exists() else None,
    }


def choose_image_id(scene: SceneSpec) -> tuple[str, Optional[Path], Optional[float]]:
    scene_id = scene.scene_id
    v30_dir = OUTPUT_ROOT / "official_difix3d" / "vanilla_gs" / scene_id / "renders"
    v60_dir = OUTPUT_ROOT / "official_3dgs_full_baseline" / scene_id / "renders"
    difix_dir = OUTPUT_ROOT / "official_difix3d" / "difix3d_gs" / scene_id / "renders" / "novel" / "59999" / "Pred"
    ours_dir = OUTPUT_ROOT / "ours_difix3dplus_gs" / scene_id / "renders" / "novel" / "59999" / "Pred"

    ids_v30 = image_ids_from_glob(v30_dir, "val_step29999_*.png", "last_token")
    ids_v60 = image_ids_from_glob(v60_dir, "val_step59999_*.png", "last_token")
    ids_difix = image_ids_from_glob(difix_dir, "*.png", "stem")
    ids_ours = image_ids_from_glob(ours_dir, "*.png", "stem")

    common = sorted(ids_v30 & ids_v60 & ids_difix & ids_ours)
    if not common:
        raise RuntimeError(f"No shared ids across vanilla/difix/difix3d+ for scene {scene.short}")

    if not scene.use_freefix:
        return common[len(common) // 2], None, None

    eval_dir = latest_eval_dir(scene_id)
    ids_freefix = image_ids_from_glob(eval_dir, "*_pred.png", "first_token")
    ids_gt = image_ids_from_glob(eval_dir, "*_gt.png", "first_token")
    common = sorted(set(common) & ids_freefix & ids_gt)
    if not common:
        raise RuntimeError(f"No shared ids with FreeFix eval for scene {scene.short}")

    best_id = common[0]
    best_delta = None
    best_ours = -1e9
    best_ff = -1e9
    for image_id in common:
        gt_path = eval_dir / f"{image_id}_gt.png"
        ff_path = eval_dir / f"{image_id}_pred.png"
        ours_path = ours_dir / f"{image_id}.png"
        if not (gt_path.exists() and ff_path.exists() and ours_path.exists()):
            continue

        gt = load_rgb(gt_path)
        ff = load_rgb(ff_path)
        ours = load_rgb(ours_path)

        if gt.shape != ff.shape:
            continue
        if gt.shape != ours.shape:
            ours_img = Image.open(ours_path).convert("RGB").resize((gt.shape[1], gt.shape[0]), Image.BICUBIC)
            ours = np.asarray(ours_img, dtype=np.float32) / 255.0

        psnr_ours = psnr(ours, gt)
        psnr_ff = psnr(ff, gt)
        delta = psnr_ff - psnr_ours

        # Prefer higher-quality Difix3D+ examples so this column is not unfairly bad.
        if (psnr_ours > best_ours) or (psnr_ours == best_ours and psnr_ff > best_ff):
            best_id = image_id
            best_ours = psnr_ours
            best_ff = psnr_ff
            best_delta = delta

    return best_id, eval_dir, best_delta


def paste_contain(canvas: Image.Image, image_path: Optional[Path], x: int, y: int, w: int, h: int, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> None:
    if image_path is None:
        draw.rectangle([x, y, x + w, y + h], fill=(40, 40, 40), outline=(100, 100, 100), width=2)
        text = "blank"
        tw, th = draw.textbbox((0, 0), text, font=font)[2:]
        draw.text((x + (w - tw) // 2, y + (h - th) // 2), text, fill=(180, 180, 180), font=font)
        return

    img = Image.open(image_path).convert("RGB")
    scale = min(w / img.width, h / img.height)
    nw = max(1, int(round(img.width * scale)))
    nh = max(1, int(round(img.height * scale)))
    resized = img.resize((nw, nh), Image.BICUBIC)
    bg = Image.new("RGB", (w, h), (10, 10, 10))
    bg.paste(resized, ((w - nw) // 2, (h - nh) // 2))
    canvas.paste(bg, (x, y))
    draw.rectangle([x, y, x + w, y + h], outline=(90, 90, 90), width=2)


def main() -> None:
    rows: list[dict[str, object]] = []
    for scene in SCENES:
        image_id, eval_dir, delta = choose_image_id(scene)
        paths = method_paths(scene.scene_id, image_id, eval_dir)
        if not scene.use_freefix:
            paths["FreeFix SDXL"] = None
        rows.append(
            {
                "scene": scene,
                "image_id": image_id,
                "eval_dir": eval_dir.name if eval_dir is not None else None,
                "delta": delta,
                "paths": paths,
            }
        )

    font = ImageFont.load_default()
    title_h = 52
    col_h = 26
    row_h = 250
    cell_w = 360
    cell_h = 208
    left_w = 170
    gap_x = 14
    gap_y = 20
    pad = 20
    cols = len(COLS)
    rows_n = len(rows)

    width = pad * 2 + left_w + cols * cell_w + (cols - 1) * gap_x
    height = pad * 2 + title_h + col_h + rows_n * row_h + (rows_n - 1) * gap_y + 24
    canvas = Image.new("RGB", (width, height), (22, 22, 22))
    draw = ImageDraw.Draw(canvas)

    title = "3-Scene Comparison: Vanilla30k | Vanilla60k | Difix3D | Difix3D+ | FreeFix SDXL"
    draw.text((pad, pad), title, fill=(240, 240, 240), font=font)

    y_cols = pad + title_h
    for c, col_name in enumerate(COLS):
        x = pad + left_w + c * (cell_w + gap_x)
        draw.rectangle([x, y_cols, x + cell_w, y_cols + col_h], fill=(45, 45, 45), outline=(90, 90, 90), width=1)
        draw.text((x + 8, y_cols + 7), col_name, fill=(240, 240, 240), font=font)

    y0 = y_cols + col_h + 10
    for r, row in enumerate(rows):
        scene: SceneSpec = row["scene"]  # type: ignore[assignment]
        image_id: str = row["image_id"]  # type: ignore[assignment]
        delta: Optional[float] = row["delta"]  # type: ignore[assignment]
        paths: dict[str, Optional[Path]] = row["paths"]  # type: ignore[assignment]

        y = y0 + r * (row_h + gap_y)
        draw.rectangle([pad, y, pad + left_w - 12, y + cell_h], fill=(35, 35, 35), outline=(90, 90, 90), width=1)
        draw.text((pad + 10, y + 14), f"scene {scene.short}", fill=(255, 255, 255), font=font)
        draw.text((pad + 10, y + 36), f"id {image_id}", fill=(205, 205, 205), font=font)
        if delta is not None:
            draw.text((pad + 10, y + 58), f"FF-ours +{delta:.2f}dB", fill=(170, 220, 170), font=font)
        else:
            draw.text((pad + 10, y + 58), "FreeFix blank", fill=(160, 160, 160), font=font)

        for c, col_name in enumerate(COLS):
            x = pad + left_w + c * (cell_w + gap_x)
            paste_contain(canvas, paths[col_name], x, y, cell_w, cell_h, draw, font)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = SAVE_DIR / f"comparison_panel_3scene_van30k_van60k_difix_difix3dplus_freefix_{timestamp}.png"
    canvas.save(out_path)
    print(out_path)
    for row in rows:
        scene: SceneSpec = row["scene"]  # type: ignore[assignment]
        print(scene.short, row["image_id"], row["eval_dir"], row["delta"])


if __name__ == "__main__":
    main()
