"""
Backward-compatible diffusion helper imports.

New shared helpers live in `utils.diffusion_utils`.
"""

from utils.diffusion_utils import ImageLike, MaskLike, resolve_hf_cache_root, to_mask_stack, to_pil_image

__all__ = [
    "ImageLike",
    "MaskLike",
    "resolve_hf_cache_root",
    "to_mask_stack",
    "to_pil_image",
]
