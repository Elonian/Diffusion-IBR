from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str in sys.path:
    sys.path.remove(project_root_str)
sys.path.insert(0, project_root_str)

from scripts.priors._utils import resolve_hf_cache_root, to_mask_stack, to_pil_image


class CustomSDXLFixer:
    """
    Local FreeFix SDXL backend using the vendored official FreeFix pipeline.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-refiner-1.0",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.device = device
        self.cache_dir = cache_dir or resolve_hf_cache_root()
        self.model_id = model_id

        from scripts.priors.src.freefix_euler_discrete_scheduler import EulerDiscreteScheduler
        from scripts.priors.src.freefix_sdxl_pipeline import StableDiffusionXLImg2ImgPipeline

        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            cache_dir=self.cache_dir,
        )
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    def __call__(
        self,
        prompt: str,
        image: Union[Image.Image, torch.Tensor, np.ndarray],
        negative_prompt: Optional[str] = None,
        mask: Optional[
            Union[
                Image.Image,
                torch.Tensor,
                np.ndarray,
                Sequence[Union[Image.Image, torch.Tensor, np.ndarray]],
            ]
        ] = None,
        mask_scheduler: Optional[list[int]] = None,
        guide_until: Optional[int] = None,
        warp_image: Optional[Union[Image.Image, torch.Tensor, np.ndarray]] = None,
        warp_until: Optional[int] = None,
        warp_mask: Optional[Union[Image.Image, torch.Tensor, np.ndarray]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        strength: float = 0.6,
    ) -> Image.Image:
        input_image = to_pil_image(image)
        if width is None or height is None:
            width, height = input_image.size
        width = int(width)
        height = int(height)
        input_image = input_image.resize((width, height), Image.LANCZOS)

        mask_tensor = None
        if mask is not None:
            if mask_scheduler is None:
                raise ValueError("mask_scheduler is required when mask is provided for FreeFix SDXL.")
            mask_tensor = to_mask_stack(mask, (width, height), device=self.device)

        warp_image_pil = None
        if warp_image is not None:
            warp_image_pil = to_pil_image(warp_image).resize((width, height), Image.LANCZOS)

        warp_mask_tensor = None
        if warp_mask is not None:
            warp_mask_tensor = to_mask_stack(warp_mask, (width, height), device=self.device)[0]

        output = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            mask=mask_tensor,
            mask_scheduler=mask_scheduler,
            warp_image=warp_image_pil,
            warp_mask=warp_mask_tensor,
            guide_until=0 if guide_until is None else int(guide_until),
            warp_until=0 if warp_until is None else int(warp_until),
            height=height,
            width=width,
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(num_inference_steps),
            generator=generator,
            strength=float(strength),
        )
        return output.images[0].resize((width, height), Image.LANCZOS)
