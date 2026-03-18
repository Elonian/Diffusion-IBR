from __future__ import annotations

import inspect
from typing import Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

from utils import blend_images, combine_mask_stack, resolve_hf_cache_root, to_mask_stack, to_pil_image


class CustomFluxFixer:
    """
    Self-contained Flux-style fixer.

    The implementation intentionally stays inside this repository and uses
    standard diffusers img2img pipelines with FreeFix-style mask/warp blending.
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

        from diffusers import AutoPipelineForImage2Image

        try:
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                cache_dir=self.cache_dir,
            )
            self.active_model_id = self.model_id
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize Flux backend. Install a Flux-compatible diffusers build "
                "and ensure Flux weights are accessible, or switch to backend='sdxl'. "
                f"Original error: {exc}"
            )

        self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    @staticmethod
    def _invoke_pipe(pipe: object, kwargs: dict) -> Image.Image:
        call_sig = inspect.signature(pipe.__call__)  # type: ignore[attr-defined]
        accepted = {k: v for k, v in kwargs.items() if k in call_sig.parameters}
        output = pipe(**accepted)  # type: ignore[misc]
        return output.images[0]

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
        input_image = input_image.resize((width, height), Image.LANCZOS)

        strength = float(strength)
        if strength <= 0.0:
            return input_image
        strength = float(min(strength, 1.0))
        safe_steps = max(1, int(num_inference_steps))
        safe_steps = max(safe_steps, int(np.ceil(1.0 / max(strength, 1e-6))))

        warp_image_pil = None
        if warp_image is not None:
            warp_image_pil = to_pil_image(warp_image).resize((width, height), Image.LANCZOS)

        infer_steps = max(1, int(safe_steps * strength))
        if guide_until is None:
            guide_until = infer_steps
        if warp_until is None:
            warp_until = infer_steps if warp_image_pil is not None else 0

        guide_ratio = float(max(0.0, min(1.0, guide_until / max(1, infer_steps))))
        effective_guidance = float(guidance_scale * guide_ratio)

        refined = self._invoke_pipe(
            self.pipe,
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": input_image,
                "height": int(height),
                "width": int(width),
                "guidance_scale": effective_guidance,
                "num_inference_steps": int(safe_steps),
                "generator": generator,
                "strength": float(strength),
            },
        ).resize((width, height), Image.LANCZOS)

        if mask is not None:
            mask_stack = to_mask_stack(mask, (width, height), device="cpu")
            mask_weight = combine_mask_stack(
                mask_stack,
                mask_scheduler=mask_scheduler,
                infer_steps=infer_steps,
            )
            refined = blend_images(input_image, refined, mask_weight)

        if warp_image_pil is not None and warp_until > 0:
            if warp_mask is None:
                warp_weight = np.ones((height, width), dtype=np.float32)
            else:
                warp_weight = to_mask_stack(warp_mask, (width, height), device="cpu")[0].cpu().numpy()
            warp_ratio = float(max(0.0, min(1.0, warp_until / max(1, infer_steps))))
            refined = blend_images(refined, warp_image_pil, np.clip(warp_weight * warp_ratio, 0.0, 1.0))

        return refined
