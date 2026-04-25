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

from utils.diffusion_utils import (
    resolve_hf_cache_root,
    to_mask_stack,
    to_pil_image,
)

FREEFIX_IMPL_ROOT = PROJECT_ROOT / "scripts" / "freefix_impl"
freefix_impl_root_str = str(FREEFIX_IMPL_ROOT)
if freefix_impl_root_str in sys.path:
    sys.path.remove(freefix_impl_root_str)
sys.path.insert(0, freefix_impl_root_str)


def _evict_foreign_freefix_modules() -> None:
    impl_root = FREEFIX_IMPL_ROOT.resolve()

    def _under_impl(path: object) -> bool:
        try:
            Path(str(path)).resolve().relative_to(impl_root)
        except ValueError:
            return False
        return True

    for module_name, module in list(sys.modules.items()):
        if not (
            module_name in {"ours", "recon", "schedulers"}
            or module_name.startswith(("ours.", "recon.", "schedulers."))
        ):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            module_paths = list(getattr(module, "__path__", []))
            if not module_paths or not all(_under_impl(path) for path in module_paths):
                sys.modules.pop(module_name, None)
            continue
        if not _under_impl(module_file):
            sys.modules.pop(module_name, None)


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
    Local FreeFix Flux backend.

    This uses the embedded FreeFix pipeline/scheduler implementation under
    scripts/freefix_impl, not the external official worktree.
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

        try:
            _evict_foreign_freefix_modules()
            from ours.pipelines.flux_pipeline import FluxPipeline
            from ours.schedulers.flow_match_euler_discrete_scheduler import FlowMatchEulerDiscreteScheduler
        except Exception as exc:
            raise RuntimeError(
                "CustomFluxFixer requires the embedded FreeFix Flux pipeline under "
                f"{FREEFIX_IMPL_ROOT}."
            ) from exc

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
        else:
            guide_until = 0

        warp_image_pil = None
        if warp_image is not None:
            if isinstance(warp_image, torch.Tensor):
                warp_image_pil = warp_image
            else:
                warp_image_pil = to_pil_image(warp_image).resize((width, height), Image.LANCZOS)

        warp_mask_tensor = None
        if warp_mask is not None:
            warp_mask_tensor = to_mask_stack(warp_mask, (width, height), device=self.device)[0]
        effective_guide_until = 0.0 if guide_until is None else float(guide_until)
        effective_warp_until = 0.0 if warp_mask_tensor is None else (0.0 if warp_until is None else float(warp_until))

        output = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            mask=mask_tensor,
            mask_scheduler=mask_scheduler,
            guide_until=effective_guide_until,
            warp_image=warp_image_pil,
            warp_until=effective_warp_until,
            warp_mask=warp_mask_tensor,
            height=height,
            width=width,
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(num_inference_steps),
            generator=generator,
            strength=float(strength),
        )
        return output.images[0].resize((width, height), Image.LANCZOS)
