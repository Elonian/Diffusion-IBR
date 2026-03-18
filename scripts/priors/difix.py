import os
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image


def _resolve_hf_cache_root() -> str:
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


def _resolve_difix_root() -> str:
    return os.environ.get(
        "DIFFUSION_IBR_DIFIX_ROOT",
        "/mntdatalora/src/Diffusion-IBR/works/Difix3D",
    )


def _ensure_import_path(path: str) -> None:
    resolved = str(Path(path).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


class CustomDifixFixer:
    """
    Thin wrapper around pretrained DIFIX pipelines (e.g. nvidia/difix_ref).
    """

    def __init__(
        self,
        model_id: str = "nvidia/difix_ref",
        model_path: Optional[str] = None,
        device: str = "cuda",
        torch_dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[str] = None,
        difix_root: Optional[str] = None,
    ) -> None:
        self.device = device
        self.cache_dir = cache_dir or _resolve_hf_cache_root()
        self.difix_root = difix_root or _resolve_difix_root()
        _ensure_import_path(self.difix_root)
        if torch_dtype is None:
            torch_dtype = torch.float16 if "cuda" in device else torch.float32

        source = model_path if model_path is not None else model_id
        try:
            # Prefer the official Difix3D custom pipeline class.
            from src.pipeline_difix import DifixPipeline  # type: ignore

            self.pipe = DifixPipeline.from_pretrained(
                source,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                cache_dir=self.cache_dir,
            )
        except Exception:
            # Fallback for environments where official class import is unavailable.
            from diffusers import DiffusionPipeline

            self.pipe = DiffusionPipeline.from_pretrained(
                source,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                cache_dir=self.cache_dir,
            )
        self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    @staticmethod
    def _to_pil(image: Union[Image.Image, torch.Tensor, np.ndarray]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, torch.Tensor):
            arr = image.detach().cpu().float().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            return Image.fromarray(arr).convert("RGB")
        if isinstance(image, np.ndarray):
            arr = image
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            return Image.fromarray(arr).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")

    def __call__(
        self,
        prompt: str,
        image: Union[Image.Image, torch.Tensor, np.ndarray],
        ref_image: Optional[Union[Image.Image, torch.Tensor, np.ndarray]] = None,
        num_inference_steps: int = 1,
        timestep: int = 199,
        guidance_scale: float = 0.0,
    ) -> Image.Image:
        # -------- 1) Normalize inputs --------
        image_pil = self._to_pil(image)
        ref_image_pil = None
        if ref_image is not None:
            ref_image_pil = self._to_pil(ref_image).resize(image_pil.size, Image.LANCZOS)

        # -------- 2) Build DIFIX kwargs --------
        kwargs = {
            "prompt": prompt,
            "image": image_pil,
            "num_inference_steps": num_inference_steps,
            "timesteps": [int(timestep)],
            "guidance_scale": guidance_scale,
        }
        if ref_image_pil is not None:
            kwargs["ref_image"] = ref_image_pil

        # -------- 3) Run pretrained DIFIX model --------
        output = self.pipe(**kwargs).images[0]

        # -------- 4) Match original size --------
        if output.size != image_pil.size:
            output = output.resize(image_pil.size, Image.LANCZOS)

        return output
