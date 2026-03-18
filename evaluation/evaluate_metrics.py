#!/usr/bin/env python3
"""Simple single-run image quality evaluation (PSNR/SSIM/LPIPS).

This script evaluates one prediction directory against one ground-truth directory.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one render folder against GT folder with PSNR/SSIM/LPIPS."
    )
    parser.add_argument("--pred-dir", required=True, type=Path, help="Prediction image directory.")
    parser.add_argument("--gt-dir", required=True, type=Path, help="Ground-truth image directory.")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON output path (no CSV is generated).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search image files recursively under pred/gt directories.",
    )
    parser.add_argument(
        "--allow-resize",
        action="store_true",
        help="Resize prediction to GT size if dimensions differ.",
    )
    parser.add_argument(
        "--lpips-net",
        default="alex",
        choices=("alex", "vgg"),
        help="LPIPS backbone.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto | cpu | cuda | cuda:0 ...",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional cap for number of evaluated image pairs (0 means no cap).",
    )
    parser.add_argument(
        "--include-per-image",
        action="store_true",
        help="Include per-image metrics in JSON output.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def list_images(root: Path, recursive: bool) -> List[Path]:
    if recursive:
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS]
    else:
        files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    return sorted(files)


def pair_by_relative_path(pred_dir: Path, gt_dir: Path, recursive: bool) -> Tuple[List[Tuple[Path, Path]], List[str]]:
    pairs: List[Tuple[Path, Path]] = []
    missing_gt: List[str] = []

    pred_files = list_images(pred_dir, recursive)
    for pred_path in pred_files:
        rel = pred_path.relative_to(pred_dir)
        gt_path = gt_dir / rel
        if gt_path.exists():
            pairs.append((pred_path, gt_path))
        else:
            missing_gt.append(str(rel))
    return pairs, missing_gt


def pair_by_filename_fallback(pred_dir: Path, gt_dir: Path, recursive: bool) -> List[Tuple[Path, Path]]:
    gt_files = list_images(gt_dir, recursive)
    gt_map: Dict[str, Path] = {}
    duplicate_names = set()
    for path in gt_files:
        key = path.name
        if key in gt_map:
            duplicate_names.add(key)
        else:
            gt_map[key] = path
    for dup in duplicate_names:
        gt_map.pop(dup, None)

    pairs: List[Tuple[Path, Path]] = []
    for pred_path in list_images(pred_dir, recursive):
        gt_path = gt_map.get(pred_path.name)
        if gt_path is not None:
            pairs.append((pred_path, gt_path))
    return pairs


def load_rgb_tensor(image_path: Path, device: str) -> torch.Tensor:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def evaluate_pairs(
    pairs: Sequence[Tuple[Path, Path]],
    device: str,
    lpips_net: str,
    allow_resize: bool,
    include_per_image: bool,
    max_images: int,
) -> Dict[str, object]:
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type=lpips_net, normalize=True
    ).to(device)
    lpips_metric.eval()

    rows: List[Dict[str, object]] = []
    psnr_vals: List[float] = []
    ssim_vals: List[float] = []
    lpips_vals: List[float] = []

    skipped_size_mismatch = 0

    for idx, (pred_path, gt_path) in enumerate(pairs):
        if max_images > 0 and idx >= max_images:
            break

        pred = load_rgb_tensor(pred_path, device)
        gt = load_rgb_tensor(gt_path, device)

        if pred.shape[-2:] != gt.shape[-2:]:
            if not allow_resize:
                skipped_size_mismatch += 1
                continue
            pred = F.interpolate(
                pred,
                size=gt.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        with torch.no_grad():
            psnr = to_float(psnr_metric(pred, gt))
            ssim = to_float(ssim_metric(pred, gt))
            lpips = to_float(lpips_metric(pred, gt))
        psnr_metric.reset()
        ssim_metric.reset()
        lpips_metric.reset()

        psnr_vals.append(psnr)
        ssim_vals.append(ssim)
        lpips_vals.append(lpips)

        if include_per_image:
            rows.append(
                {
                    "pred_image": str(pred_path),
                    "gt_image": str(gt_path),
                    "psnr": psnr,
                    "ssim": ssim,
                    "lpips": lpips,
                }
            )

    if not psnr_vals:
        raise RuntimeError(
            "No valid image pairs were evaluated. Check directory pairing and image sizes."
        )

    result: Dict[str, object] = {
        "num_pairs_evaluated": len(psnr_vals),
        "num_pairs_skipped_size_mismatch": skipped_size_mismatch,
        "metrics": {
            "psnr": float(np.mean(psnr_vals)),
            "ssim": float(np.mean(ssim_vals)),
            "lpips": float(np.mean(lpips_vals)),
        },
    }
    if include_per_image:
        result["per_image"] = rows
    return result


def main() -> None:
    args = parse_args()

    pred_dir = args.pred_dir.resolve()
    gt_dir = args.gt_dir.resolve()
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth directory not found: {gt_dir}")

    device = resolve_device(args.device)

    pairs, missing_gt = pair_by_relative_path(pred_dir, gt_dir, recursive=args.recursive)
    matching_mode = "relative_path"
    if not pairs:
        pairs = pair_by_filename_fallback(pred_dir, gt_dir, recursive=args.recursive)
        matching_mode = "filename_fallback"
        missing_gt = []

    if not pairs:
        raise RuntimeError("No image pairs found between pred-dir and gt-dir.")

    report = evaluate_pairs(
        pairs=pairs,
        device=device,
        lpips_net=args.lpips_net,
        allow_resize=args.allow_resize,
        include_per_image=args.include_per_image,
        max_images=args.max_images,
    )

    output = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pred_dir": str(pred_dir),
        "gt_dir": str(gt_dir),
        "matching_mode": matching_mode,
        "num_pred_images_found": len(list_images(pred_dir, args.recursive)),
        "num_pairs_found": len(pairs),
        "num_pred_missing_gt_relative_match": len(missing_gt),
        "device": device,
        "lpips_net": args.lpips_net,
        **report,
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(output, indent=2), encoding="utf-8")

    metrics = output["metrics"]
    print(f"Pred dir: {output['pred_dir']}")
    print(f"GT dir:   {output['gt_dir']}")
    print(f"Device:   {output['device']}")
    print(f"Pairs:    {output['num_pairs_evaluated']} / {output['num_pairs_found']}")
    print(f"PSNR:     {metrics['psnr']:.6f}")
    print(f"SSIM:     {metrics['ssim']:.6f}")
    print(f"LPIPS:    {metrics['lpips']:.6f}")
    if args.json_out:
        print(f"JSON:     {args.json_out.resolve()}")


if __name__ == "__main__":
    main()
