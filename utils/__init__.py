"""
Project utility namespace.

Exports are resolved lazily so lightweight helpers can be imported without
loading optional COLMAP/OpenCV/Torch dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "ColmapParser": ("colmap_data", "ColmapParser"),
    "Parser": ("colmap_data", "Parser"),
    "ColmapImageDataset": ("colmap_dataset", "ColmapImageDataset"),
    "Dataset": ("colmap_dataset", "Dataset"),
    "similarity_from_cameras": ("camera_normalization", "similarity_from_cameras"),
    "align_principle_axes": ("camera_normalization", "align_principle_axes"),
    "transform_cameras": ("camera_normalization", "transform_cameras"),
    "transform_points": ("camera_normalization", "transform_points"),
    "normalize": ("camera_normalization", "normalize"),
    "normalize_cameras_and_points": ("camera_normalization", "normalize_cameras_and_points"),
    "set_random_seed": ("training_utils", "set_random_seed"),
    "rgb_to_sh": ("training_utils", "rgb_to_sh"),
    "knn": ("training_utils", "knn"),
    "compute_psnr": ("training_utils", "compute_psnr"),
    "simple_ssim": ("training_utils", "simple_ssim"),
    "parse_steps_csv": ("training_utils", "parse_steps_csv"),
    "parse_float_csv": ("training_utils", "parse_float_csv"),
    "parse_name_csv": ("training_utils", "parse_name_csv"),
    "soft_sigmoid": ("training_utils", "soft_sigmoid"),
    "rotation_matrix_to_quaternion": ("pose_utils", "rotation_matrix_to_quaternion"),
    "quaternion_to_rotation_matrix": ("pose_utils", "quaternion_to_rotation_matrix"),
    "slerp_quaternion": ("pose_utils", "slerp_quaternion"),
    "CameraPoseInterpolator": ("pose_utils", "CameraPoseInterpolator"),
    "interpolate_pose": ("pose_utils", "interpolate_pose"),
    "generate_interpolated_path": ("traj_utils", "generate_interpolated_path"),
    "generate_ellipse_path_y": ("traj_utils", "generate_ellipse_path_y"),
    "generate_ellipse_path_z": ("traj_utils", "generate_ellipse_path_z"),
    "generate_spiral_path": ("traj_utils", "generate_spiral_path"),
    "resolve_hf_cache_root": ("diffusion_utils", "resolve_hf_cache_root"),
    "resolve_freefix_root": ("diffusion_utils", "resolve_freefix_root"),
    "ensure_import_path": ("diffusion_utils", "ensure_import_path"),
    "to_pil_image": ("diffusion_utils", "to_pil_image"),
    "to_mask_stack": ("diffusion_utils", "to_mask_stack"),
    "combine_mask_stack": ("diffusion_utils", "combine_mask_stack"),
    "blend_images": ("diffusion_utils", "blend_images"),
    "FREEFIX_DEFAULT_PROMPT": ("freefix_support", "DEFAULT_PROMPT"),
    "FREEFIX_DEFAULT_NEGATIVE_PROMPT": ("freefix_support", "DEFAULT_NEGATIVE_PROMPT"),
    "FreeFixSceneAssets": ("freefix_support", "FreeFixSceneAssets"),
    "generate_freefix_scene_assets": ("freefix_support", "generate_freefix_scene_assets"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(f".{module_name}", __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
