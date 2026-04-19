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


def _image_size(image: Union[Image.Image, torch.Tensor, np.ndarray]) -> tuple[int, int]:
    if isinstance(image, Image.Image):
        return image.size
    if isinstance(image, torch.Tensor):
        if image.ndim == 3 and image.shape[0] in (1, 3):
            return int(image.shape[2]), int(image.shape[1])
        return int(image.shape[1]), int(image.shape[0])
    return int(image.shape[1]), int(image.shape[0])


class CustomFluxFixer:
    """
    Local FreeFix Flux backend using the vendored official FreeFix pipeline.
    """

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-dev",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.device = device
        self.cache_dir = cache_dir or resolve_hf_cache_root()
        self.model_id = model_id

        from scripts.priors.src.freefix_flux_pipeline import FluxPipeline
        from scripts.priors.src.freefix_flow_match_euler_discrete_scheduler import (
            FlowMatchEulerDiscreteScheduler,
        )

        self.pipe = FluxPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            cache_dir=self.cache_dir,
        )
        self.pipe = self.pipe.to(self.device)
        self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
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
        guide_until: Optional[float] = None,
        warp_image: Optional[Union[Image.Image, torch.Tensor, np.ndarray]] = None,
        warp_until: Optional[float] = None,
        warp_mask: Optional[Union[Image.Image, torch.Tensor, np.ndarray]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        strength: float = 0.5,
    ) -> Image.Image:
        input_image = image if isinstance(image, torch.Tensor) else to_pil_image(image)
        if width is None or height is None:
            width, height = _image_size(input_image)
        width = int(width)
        height = int(height)
        if isinstance(input_image, Image.Image):
            input_image = input_image.resize((width, height), Image.LANCZOS)

        mask_tensor = None
        if mask is not None:
            if mask_scheduler is None:
                raise ValueError("mask_scheduler is required when mask is provided for FreeFix Flux.")
            mask_tensor = to_mask_stack(mask, (width, height), device=self.device)

        warp_image_pil = None
        if warp_image is not None:
            if isinstance(warp_image, torch.Tensor):
                warp_image_pil = warp_image
            else:
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
            guide_until=0 if guide_until is None else guide_until,
            warp_until=0 if warp_until is None else warp_until,
            height=height,
            width=width,
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(num_inference_steps),
            generator=generator,
            strength=float(strength),
        )
        return output.images[0].resize((width, height), Image.LANCZOS)
