"""
Stable local utility exports for scripts that must avoid third-party `utils`
package collisions.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UTILS_DIR = PROJECT_ROOT / "utils"
MODULE_NAME = "diffusion_ibr_local_utils"

module = sys.modules.get(MODULE_NAME)
if module is None:
    spec = importlib.util.spec_from_file_location(
        MODULE_NAME,
        UTILS_DIR / "__init__.py",
        submodule_search_locations=[str(UTILS_DIR)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load local utils package from {UTILS_DIR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)


CameraPoseInterpolator = module.CameraPoseInterpolator
ColmapImageDataset = module.ColmapImageDataset
ColmapParser = module.ColmapParser
blend_images = module.blend_images
combine_mask_stack = module.combine_mask_stack
compute_psnr = module.compute_psnr
ensure_import_path = module.ensure_import_path
interpolate_pose = module.interpolate_pose
knn = module.knn
parse_float_csv = module.parse_float_csv
parse_name_csv = module.parse_name_csv
parse_steps_csv = module.parse_steps_csv
normalize = module.normalize
generate_ellipse_path_y = module.generate_ellipse_path_y
generate_ellipse_path_z = module.generate_ellipse_path_z
generate_interpolated_path = module.generate_interpolated_path
generate_spiral_path = module.generate_spiral_path
resolve_freefix_root = module.resolve_freefix_root
resolve_hf_cache_root = module.resolve_hf_cache_root
rgb_to_sh = module.rgb_to_sh
rotation_matrix_to_quaternion = module.rotation_matrix_to_quaternion
set_random_seed = module.set_random_seed
simple_ssim = module.simple_ssim
soft_sigmoid = module.soft_sigmoid
to_mask_stack = module.to_mask_stack
to_pil_image = module.to_pil_image

__all__ = [
    "CameraPoseInterpolator",
    "ColmapImageDataset",
    "ColmapParser",
    "blend_images",
    "combine_mask_stack",
    "compute_psnr",
    "ensure_import_path",
    "interpolate_pose",
    "knn",
    "parse_float_csv",
    "parse_name_csv",
    "parse_steps_csv",
    "normalize",
    "generate_ellipse_path_y",
    "generate_ellipse_path_z",
    "generate_interpolated_path",
    "generate_spiral_path",
    "resolve_freefix_root",
    "resolve_hf_cache_root",
    "rgb_to_sh",
    "rotation_matrix_to_quaternion",
    "set_random_seed",
    "simple_ssim",
    "soft_sigmoid",
    "to_mask_stack",
    "to_pil_image",
]
