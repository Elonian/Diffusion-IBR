import argparse
import os
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

from scripts.priors.difix import CustomDifixFixer
from scripts.priors.flux import CustomFluxFixer
from scripts.priors.sdxl import CustomSDXLFixer
from utils.diffusion_utils import to_pil_image

ImageInput = Union[Image.Image, torch.Tensor, np.ndarray]
MaskInput = Union[ImageInput, Sequence[ImageInput]]


class DiffusionFixer:
    """Entry-point wrapper that selects Flux, SDXL, or DIFIX custom fixer."""

    def __init__(
        self,
        backend: str = "flux",
        cache_dir: Optional[str] = None,
        device: str = "cuda",
        difix_model_id: str = "nvidia/difix_ref",
        difix_model_path: Optional[str] = None,
    ) -> None:
        backend = backend.lower().strip()
        if backend not in {"flux", "sdxl", "difix"}:
            raise ValueError("backend must be one of: flux, sdxl, difix")
        self.backend = backend
        self.device = device
        if backend == "difix" and difix_model_path is None:
            local_difix = PROJECT_ROOT / "cache_weights" / "difix_ref"
            if local_difix.exists():
                difix_model_path = str(local_difix)

        if backend == "flux":
            self.model = CustomFluxFixer(cache_dir=cache_dir, device=device)
        elif backend == "sdxl":
            self.model = CustomSDXLFixer(cache_dir=cache_dir, device=device)
        else:
            self.model = CustomDifixFixer(
                model_id=difix_model_id,
                model_path=difix_model_path,
                cache_dir=cache_dir,
                device=device,
            )

    def _build_generator(self, seed: int) -> torch.Generator:
        device = self.device
        if device == "mps":
            device = "cpu"
        try:
            generator = torch.Generator(device=device)
        except (RuntimeError, TypeError):
            generator = torch.Generator()
        generator.manual_seed(int(seed))
        return generator

    def __call__(
        self,
        prompt: str,
        image: ImageInput,
        negative_prompt: Optional[str] = None,
        ref_image: Optional[ImageInput] = None,
        mask: Optional[MaskInput] = None,
        warp_image: Optional[ImageInput] = None,
        warp_mask: Optional[ImageInput] = None,
        mask_scheduler: Optional[list[int]] = None,
        guide_until: Optional[float] = None,
        warp_until: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        strength: float = 0.5,
        guidance_scale: Optional[float] = None,
        timestep: int = 199,
        seed: Optional[int] = 64,
        generator: Optional[torch.Generator] = None,
    ) -> Image.Image:
        # -------- 1) Dispatch to selected custom fixer backend --------
        if self.backend == "difix":
            if guidance_scale is None:
                guidance_scale = 0.0
            if num_inference_steps is None:
                num_inference_steps = 1
            return self.model(
                prompt=prompt,
                image=image,
                ref_image=ref_image if ref_image is not None else warp_image,
                num_inference_steps=num_inference_steps,
                timestep=timestep,
                guidance_scale=guidance_scale,
                generator=None,
            )

        if generator is None:
            generator = self._build_generator(64 if seed is None else seed)

        image_for_size = image if isinstance(image, Image.Image) else to_pil_image(image)

        # -------- 2) Run deterministic non-DIFIX fixer backend --------
        if num_inference_steps is None:
            num_inference_steps = 50
        if guidance_scale is None:
            guidance_scale = 3.5
        return self.model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask=mask,
            mask_scheduler=mask_scheduler,
            guide_until=guide_until,
            warp_image=warp_image if warp_image is not None else ref_image,
            warp_until=warp_until
            if warp_until is not None
            else (num_inference_steps // 2 if (warp_image is not None or ref_image is not None) else None),
            warp_mask=warp_mask,
            height=image_for_size.height,
            width=image_for_size.width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            strength=strength,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple Diffusion fixer entrypoint (Flux, SDXL, or DIFIX).")
    parser.add_argument("--backend", type=str, default="flux", choices=["flux", "sdxl", "difix"], help="Fixer backend")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input RGB image")
    parser.add_argument("--output_image", type=str, required=True, help="Path to save fixed RGB image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt")
    parser.add_argument("--ref_image", type=str, default=None, help="Optional reference image path (for DIFIX)")
    parser.add_argument("--mask", type=str, default=None, help="Optional grayscale mask path")
    parser.add_argument("--warp_image", type=str, default=None, help="Optional warp/reference image path")
    parser.add_argument("--warp_mask", type=str, default=None, help="Optional warp mask path")
    parser.add_argument("--steps", type=int, default=None, help="Number of denoising steps")
    parser.add_argument("--timestep", type=int, default=199, help="Single-step diffusion timestep (DIFIX)")
    parser.add_argument("--strength", type=float, default=0.5, help="Img2img strength")
    parser.add_argument("--guidance_scale", type=float, default=None, help="CFG scale; backend default when omitted")
    parser.add_argument("--seed", type=int, default=64, help="Random seed")
    parser.add_argument("--difix_model_id", type=str, default="nvidia/difix_ref", help="DIFIX model id")
    parser.add_argument("--difix_model_path", type=str, default=None, help="Optional local DIFIX model path")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.path.join("/mntdatalora/src/Diffusion-IBR/cache_weights", "huggingface", "hub"),
        help="Hugging Face cache dir",
    )
    args = parser.parse_args()

    image = Image.open(args.input_image).convert("RGB")
    ref_image = Image.open(args.ref_image).convert("RGB") if args.ref_image else None
    mask = Image.open(args.mask).convert("L") if args.mask else None
    warp_image = Image.open(args.warp_image).convert("RGB") if args.warp_image else None
    warp_mask = Image.open(args.warp_mask).convert("L") if args.warp_mask else None

    fixer = DiffusionFixer(
        backend=args.backend,
        cache_dir=args.cache_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        difix_model_id=args.difix_model_id,
        difix_model_path=args.difix_model_path,
    )
    output = fixer(
        prompt=args.prompt,
        image=image,
        negative_prompt=args.negative_prompt,
        ref_image=ref_image,
        mask=mask,
        warp_image=warp_image,
        warp_mask=warp_mask,
        num_inference_steps=args.steps,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        timestep=args.timestep,
        seed=args.seed,
    )
    os.makedirs(os.path.dirname(args.output_image) or ".", exist_ok=True)
    output.save(args.output_image)


if __name__ == "__main__":
    main()
