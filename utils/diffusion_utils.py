"""
Reusable diffusion helpers shared by FreeFix/DiFix wrappers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

ImageLike = Union[Image.Image, torch.Tensor, np.ndarray]
MaskLike = Union[ImageLike, Sequence[ImageLike]]


def resolve_hf_cache_root() -> str:
    cache_root = os.environ.get("DIFFUSION_IBR_CACHE_DIR", "/mntdatalora/src/Diffusion-IBR/cache_weights")
    hf_home = os.path.join(cache_root, "huggingface")
    hf_hub_cache = os.path.join(hf_home, "hub")
    transformers_cache = os.path.join(hf_home, "transformers")
    os.makedirs(hf_hub_cache, exist_ok=True)
    os.makedirs(transformers_cache, exist_ok=True)
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_hub_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", transformers_cache)
    return hf_hub_cache


def resolve_freefix_root() -> str:
    # Backward-compatible helper retained for older code paths.
    # The self-sufficient FreeFix path in this repo does not require this.
    return os.environ.get(
        "DIFFUSION_IBR_FREEFIX_ROOT",
        "/mntdatalora/src/Diffusion-IBR",
    )


def ensure_import_path(path: str) -> None:
    resolved = str(Path(path).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def to_pil_image(image: ImageLike) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().float().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
    elif isinstance(image, np.ndarray):
        arr = image.astype(np.float32)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    if arr.max() <= 1.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return Image.fromarray(arr).convert("RGB")


def to_mask_stack(
    mask: MaskLike,
    size: tuple[int, int],
    device: Union[str, torch.device],
) -> torch.Tensor:
    def _single_to_array(single_mask: ImageLike) -> np.ndarray:
        if isinstance(single_mask, Image.Image):
            arr = np.array(single_mask.convert("L"), dtype=np.float32)
        elif isinstance(single_mask, torch.Tensor):
            arr = single_mask.detach().cpu().float().numpy()
        elif isinstance(single_mask, np.ndarray):
            arr = single_mask.astype(np.float32)
        else:
            raise TypeError(f"Unsupported mask type: {type(single_mask)}")

        if arr.ndim == 3:
            if arr.shape[-1] in (1, 3):
                arr = arr[..., 0]
            elif arr.shape[0] in (1, 3):
                arr = arr[0]
            else:
                arr = arr[..., 0]
        if arr.max() > 1.0:
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)
        resized = Image.fromarray((arr * 255.0).astype(np.uint8), mode="L").resize(size, Image.LANCZOS)
        return np.array(resized, dtype=np.float32) / 255.0

    if isinstance(mask, Sequence) and not isinstance(mask, (Image.Image, np.ndarray, torch.Tensor)):
        arrays = [_single_to_array(m) for m in mask]
    else:
        arrays = [_single_to_array(mask)]  # type: ignore[arg-type]

    stacked = np.stack(arrays, axis=0)
    return torch.from_numpy(stacked).float().to(device)


def combine_mask_stack(
    mask_stack: torch.Tensor,
    *,
    mask_scheduler: Optional[Sequence[int]] = None,
    infer_steps: int,
) -> torch.Tensor:
    """
    Convert a stack of masks [N,H,W] into a single weighted mask [H,W].
    """
    if mask_stack.ndim != 3:
        raise ValueError(f"mask_stack must have shape [N,H,W], got {tuple(mask_stack.shape)}")
    n_masks = int(mask_stack.shape[0])
    if n_masks == 0:
        raise ValueError("mask_stack must contain at least one mask")

    if mask_scheduler is None or len(mask_scheduler) == 0:
        weights = torch.ones((n_masks,), dtype=mask_stack.dtype, device=mask_stack.device)
    else:
        raw = [max(1, int(v)) for v in mask_scheduler[:n_masks]]
        if len(raw) < n_masks:
            raw.extend([raw[-1]] * (n_masks - len(raw)))
        denom = float(max(1, int(infer_steps)))
        weights = torch.tensor(raw, dtype=mask_stack.dtype, device=mask_stack.device) / denom
        weights = torch.clamp(weights, 0.0, 1.0)

    weighted = (mask_stack * weights[:, None, None]).sum(dim=0)
    norm = torch.clamp(weights.sum(), min=1e-6)
    return torch.clamp(weighted / norm, 0.0, 1.0)


def blend_images(base: ImageLike, edited: ImageLike, weight_map: ImageLike) -> Image.Image:
    """
    Blend two RGB images with a scalar weight map in [0,1].
    output = edited * weight + base * (1 - weight)
    """
    base_img = np.array(to_pil_image(base).convert("RGB"), dtype=np.float32) / 255.0
    edited_img = np.array(to_pil_image(edited).convert("RGB"), dtype=np.float32) / 255.0
    if base_img.shape != edited_img.shape:
        raise ValueError(f"base and edited image shape mismatch: {base_img.shape} vs {edited_img.shape}")

    h, w = base_img.shape[:2]
    weight_tensor = to_mask_stack(weight_map, (w, h), device="cpu")[0].cpu().numpy().astype(np.float32)
    weight = np.clip(weight_tensor[..., None], 0.0, 1.0)
    out = edited_img * weight + base_img * (1.0 - weight)
    return Image.fromarray(np.clip(out * 255.0, 0, 255).astype(np.uint8), mode="RGB")
