from __future__ import annotations

import os
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
        if arr.shape == (size[1], size[0]):
            return arr.astype(np.float32)
        resized = Image.fromarray((arr * 255.0).astype(np.uint8), mode="L").resize(size, Image.LANCZOS)
        return np.array(resized, dtype=np.float32) / 255.0

    if isinstance(mask, Sequence) and not isinstance(mask, (Image.Image, np.ndarray, torch.Tensor)):
        arrays = [_single_to_array(m) for m in mask]
    else:
        arrays = [_single_to_array(mask)]  # type: ignore[arg-type]

    stacked = np.stack(arrays, axis=0)
    return torch.from_numpy(stacked).float().to(device)
