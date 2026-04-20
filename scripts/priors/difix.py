import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str in sys.path:
    sys.path.remove(project_root_str)
sys.path.insert(0, project_root_str)

from utils.diffusion_utils import ensure_import_path, resolve_hf_cache_root, to_pil_image


class CustomDifixFixer:
    """
    Thin wrapper around the local DIFIX pipeline loader and released weights.
    """

    def __init__(
        self,
        model_id: str = "nvidia/difix_ref",
        model_path: Optional[str] = None,
        device: str = "cuda",
        torch_dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.device = device
        self.cache_dir = cache_dir or resolve_hf_cache_root()
        ensure_import_path(str(PROJECT_ROOT))

        source = model_path if model_path is not None else model_id
        try:
            from scripts.priors.src.pipeline_difix import DifixPipeline
        except Exception as exc:
            raise RuntimeError(
                "Failed to import the local DifixPipeline from scripts/priors/src/pipeline_difix.py."
            ) from exc

        load_kwargs = {
            "cache_dir": self.cache_dir,
        }
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        self.pipe = DifixPipeline.from_pretrained(source, **load_kwargs)
        self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    @staticmethod
    def _to_pil(image: Union[Image.Image, torch.Tensor, np.ndarray]) -> Image.Image:
        return to_pil_image(image)

    def __call__(
        self,
        prompt: str,
        image: Union[Image.Image, torch.Tensor, np.ndarray],
        ref_image: Optional[Union[Image.Image, torch.Tensor, np.ndarray]] = None,
        num_inference_steps: int = 1,
        timestep: int = 199,
        guidance_scale: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> Image.Image:
        # -------- 1) Normalize inputs --------
        image_pil = self._to_pil(image)
        ref_image_pil = None
        if ref_image is not None:
            ref_image_pil = self._to_pil(ref_image)

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
        if generator is not None:
            kwargs["generator"] = generator

        # -------- 3) Run pretrained DIFIX model --------
        output = self.pipe(**kwargs).images[0]

        # -------- 4) Match original size --------
        if output.size != image_pil.size:
            output = output.resize(image_pil.size, Image.LANCZOS)

        return output
