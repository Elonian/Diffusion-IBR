#!/usr/bin/env python3
"""
Render videos from gsplat/3DGS checkpoints.

Examples:
  python scripts/rendering/render.py \
    --ckpt outputs/gaussian_splatting_baseline/<scene>/ckpts/ckpt_59999_rank0.pt \
    --data_dir data/DL3DV-10K-Benchmark/<scene>/gaussian_splat \
    --trajectory_mode circle \
    --output outputs/renders/<scene>_circle.mp4

  python scripts/rendering/render.py \
    --ckpt outputs/gaussian_splatting_baseline/<scene>/ckpts/ckpt_59999_rank0.pt \
    --data_dir data/DL3DV-10K-Benchmark/<scene>/gaussian_splat \
    --trajectory cameras.npy \
    --output outputs/renders/<scene>_custom.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str in sys.path:
    sys.path.remove(project_root_str)
sys.path.insert(0, project_root_str)

imageio = None
torch = None
rasterization = None
ColmapParser = None
focus_point_fn = None
generate_ellipse_path_z = None
generate_interpolated_path = None
generate_spiral_path = None
viewmatrix = None


def _load_runtime_modules() -> None:
    global imageio
    global torch
    global rasterization
    global ColmapParser
    global focus_point_fn
    global generate_ellipse_path_z
    global generate_interpolated_path
    global generate_spiral_path
    global viewmatrix

    import imageio.v2 as imageio_mod
    import torch as torch_mod
    from gsplat.rendering import rasterization as rasterization_mod
    from utils.colmap_data import ColmapParser as ColmapParserMod
    from utils.traj_utils import (
        focus_point_fn as focus_point_fn_mod,
        generate_ellipse_path_z as generate_ellipse_path_z_mod,
        generate_interpolated_path as generate_interpolated_path_mod,
        generate_spiral_path as generate_spiral_path_mod,
        viewmatrix as viewmatrix_mod,
    )

    imageio = imageio_mod
    torch = torch_mod
    rasterization = rasterization_mod
    ColmapParser = ColmapParserMod
    focus_point_fn = focus_point_fn_mod
    generate_ellipse_path_z = generate_ellipse_path_z_mod
    generate_interpolated_path = generate_interpolated_path_mod
    generate_spiral_path = generate_spiral_path_mod
    viewmatrix = viewmatrix_mod


@dataclass
class LoadedTrajectory:
    camtoworlds: np.ndarray
    ks: Optional[np.ndarray] = None
    width: Optional[int] = None
    height: Optional[int] = None


def _normalize_vec(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(x))
    if norm < eps:
        return x
    return x / norm


def _as_camtoworlds(array: Any) -> np.ndarray:
    poses = np.asarray(array, dtype=np.float32)
    if poses.ndim == 2:
        poses = poses[None]
    if poses.ndim != 3:
        raise ValueError(f"Camera trajectory must have shape [N, 3, 4] or [N, 4, 4], got {poses.shape}.")
    if poses.shape[1:] == (3, 4):
        bottom = np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]], dtype=np.float32), len(poses), axis=0)
        poses = np.concatenate([poses, bottom], axis=1)
    elif poses.shape[1:] != (4, 4):
        raise ValueError(f"Camera trajectory must have shape [N, 3, 4] or [N, 4, 4], got {poses.shape}.")
    return poses.astype(np.float32)


def _poses_3x4_to_4x4(poses: np.ndarray) -> np.ndarray:
    return _as_camtoworlds(poses)


def _safe_focus_point(poses: np.ndarray) -> np.ndarray:
    try:
        return focus_point_fn(poses[:, :3, :4]).astype(np.float32)
    except Exception:
        return poses[:, :3, 3].mean(axis=0).astype(np.float32)


def _dominant_up_axis(poses: np.ndarray) -> np.ndarray:
    avg_up = poses[:, :3, 1].mean(axis=0)
    avg_up = _normalize_vec(avg_up)
    if float(np.linalg.norm(avg_up)) < 1e-8:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    dominant = int(np.argmax(np.abs(avg_up)))
    up = np.zeros(3, dtype=np.float32)
    up[dominant] = 1.0 if avg_up[dominant] >= 0.0 else -1.0
    return up


def _orbit_axes(poses: np.ndarray, center: np.ndarray, up: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    base = poses[0, :3, 3] - center
    base = base - up * float(np.dot(base, up))
    if float(np.linalg.norm(base)) < 1e-8:
        base = poses[:, :3, 3].mean(axis=0) - center
        base = base - up * float(np.dot(base, up))
    if float(np.linalg.norm(base)) < 1e-8:
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(fallback, up))) > 0.9:
            fallback = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        base = fallback - up * float(np.dot(fallback, up))
    x_axis = _normalize_vec(base).astype(np.float32)
    y_axis = _normalize_vec(np.cross(up, x_axis)).astype(np.float32)
    return x_axis, y_axis


def generate_circular_path(
    poses: np.ndarray,
    n_frames: int,
    radius_scale: float = 1.0,
    height_offset: float = 0.0,
) -> np.ndarray:
    poses = _poses_3x4_to_4x4(poses)
    center = _safe_focus_point(poses)
    up = _dominant_up_axis(poses)
    x_axis, y_axis = _orbit_axes(poses, center, up)
    radii = np.linalg.norm(poses[:, :3, 3] - center[None], axis=-1)
    radius = max(float(np.percentile(radii, 70)) * radius_scale, 1e-4)
    orbit_center = center + up * float(height_offset)

    render_poses = []
    for theta in np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False):
        position = orbit_center + radius * (math.cos(theta) * x_axis + math.sin(theta) * y_axis)
        render_poses.append(viewmatrix(orbit_center - position, up, position))
    return np.stack(render_poses, axis=0).astype(np.float32)


def generate_spherical_path(
    poses: np.ndarray,
    n_frames: int,
    radius_scale: float = 1.0,
    elevation_deg: float = 20.0,
) -> np.ndarray:
    poses = _poses_3x4_to_4x4(poses)
    center = _safe_focus_point(poses)
    up = _dominant_up_axis(poses)
    x_axis, y_axis = _orbit_axes(poses, center, up)
    radii = np.linalg.norm(poses[:, :3, 3] - center[None], axis=-1)
    radius = max(float(np.percentile(radii, 70)) * radius_scale, 1e-4)
    elevation_amp = math.radians(float(elevation_deg))

    render_poses = []
    for theta in np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False):
        elevation = elevation_amp * math.sin(2.0 * theta)
        planar = math.cos(elevation) * (math.cos(theta) * x_axis + math.sin(theta) * y_axis)
        vertical = math.sin(elevation) * up
        position = center + radius * (planar + vertical)
        render_poses.append(viewmatrix(center - position, up, position))
    return np.stack(render_poses, axis=0).astype(np.float32)


def _find_first(mapping: Dict[str, Any], names: Iterable[str]) -> Optional[Any]:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _intrinsics_from_fov(width: int, height: int, fov: float) -> np.ndarray:
    fov = float(fov)
    fov_rad = math.radians(fov) if fov > math.pi else fov
    focal = 0.5 * float(width) / max(math.tan(0.5 * fov_rad), 1e-8)
    return np.array(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _parse_json_trajectory(data: Any) -> LoadedTrajectory:
    if isinstance(data, list):
        return LoadedTrajectory(camtoworlds=_as_camtoworlds(data))
    if not isinstance(data, dict):
        raise ValueError("JSON trajectory must be either a list of poses or an object.")

    if "camera_path" in data:
        frames = data["camera_path"]
        poses = []
        ks = []
        width = data.get("render_width") or data.get("width") or data.get("w")
        height = data.get("render_height") or data.get("height") or data.get("h")
        for frame in frames:
            pose = _find_first(frame, ("camera_to_world", "camera_to_worlds", "camtoworld", "c2w", "transform_matrix"))
            if pose is None:
                raise ValueError("A camera_path frame is missing camera_to_world/c2w data.")
            pose_array = np.asarray(pose, dtype=np.float32)
            if pose_array.size == 16:
                pose_array = pose_array.reshape(4, 4)
            poses.append(pose_array)

            fl_x = frame.get("fl_x") or frame.get("fx")
            fl_y = frame.get("fl_y") or frame.get("fy") or fl_x
            cx = frame.get("cx")
            cy = frame.get("cy")
            frame_w = frame.get("w") or frame.get("width") or width
            frame_h = frame.get("h") or frame.get("height") or height
            if fl_x is not None and fl_y is not None and cx is not None and cy is not None:
                ks.append(
                    np.array(
                        [[float(fl_x), 0.0, float(cx)], [0.0, float(fl_y), float(cy)], [0.0, 0.0, 1.0]],
                        dtype=np.float32,
                    )
                )
            elif frame.get("fov") is not None and frame_w is not None and frame_h is not None:
                ks.append(_intrinsics_from_fov(int(frame_w), int(frame_h), float(frame["fov"])))

        return LoadedTrajectory(
            camtoworlds=_as_camtoworlds(poses),
            ks=np.stack(ks, axis=0) if len(ks) == len(poses) else None,
            width=int(width) if width is not None else None,
            height=int(height) if height is not None else None,
        )

    poses = _find_first(data, ("camtoworlds", "camera_to_worlds", "camera_to_world", "c2ws", "poses", "trajectory"))
    if poses is None:
        raise ValueError("JSON trajectory needs camtoworlds/c2ws/poses/trajectory or camera_path.")

    k_data = _find_first(data, ("Ks", "ks", "K", "intrinsics"))
    ks = None if k_data is None else np.asarray(k_data, dtype=np.float32)
    if ks is not None and ks.ndim == 2:
        ks = ks[None]
    return LoadedTrajectory(
        camtoworlds=_as_camtoworlds(poses),
        ks=ks,
        width=int(data["width"]) if data.get("width") is not None else None,
        height=int(data["height"]) if data.get("height") is not None else None,
    )


def load_trajectory(path: Path) -> LoadedTrajectory:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return LoadedTrajectory(camtoworlds=_as_camtoworlds(np.load(path)))
    if suffix == ".npz":
        data = np.load(path)
        pose_key = next(
            (key for key in ("camtoworlds", "camera_to_worlds", "c2ws", "poses", "trajectory") if key in data),
            None,
        )
        if pose_key is None:
            raise ValueError(f"{path} does not contain camtoworlds/c2ws/poses/trajectory.")
        k_key = next((key for key in ("Ks", "ks", "K", "intrinsics") if key in data), None)
        ks = None if k_key is None else np.asarray(data[k_key], dtype=np.float32)
        if ks is not None and ks.ndim == 2:
            ks = ks[None]
        width = int(np.asarray(data["width"]).item()) if "width" in data else None
        height = int(np.asarray(data["height"]).item()) if "height" in data else None
        return LoadedTrajectory(camtoworlds=_as_camtoworlds(data[pose_key]), ks=ks, width=width, height=height)
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return _parse_json_trajectory(json.load(f))
    raise ValueError(f"Unsupported trajectory format: {path}. Use .npy, .npz, or .json.")


def select_source_poses(parser: ColmapParser, source: str, test_every: int, trim_ends: int) -> np.ndarray:
    n = len(parser.camtoworlds)
    indices = np.arange(n)
    image_names = list(getattr(parser, "image_names", []))
    tagged = test_every == 1 and any("_train_" in name or "_eval_" in name for name in image_names)

    if source == "train":
        if tagged:
            indices = np.array([i for i in indices if "_train_" in image_names[i]], dtype=np.int64)
        elif test_every > 0:
            indices = indices[indices % max(test_every, 1) == 0]
    elif source in {"test", "val"}:
        if tagged:
            indices = np.array([i for i in indices if "_eval_" in image_names[i]], dtype=np.int64)
        elif test_every > 0:
            indices = indices[indices % max(test_every, 1) != 0]
    elif source != "all":
        raise ValueError(f"Unknown source camera split: {source}")

    poses = parser.camtoworlds[indices].astype(np.float32)
    if trim_ends > 0 and len(poses) > 2 * trim_ends:
        poses = poses[trim_ends:-trim_ends]
    if len(poses) == 0:
        raise ValueError("No source camera poses available for trajectory generation.")
    return poses


def build_generated_trajectory(args: argparse.Namespace, parser: ColmapParser, scene_scale: float) -> np.ndarray:
    keyframes = select_source_poses(parser, args.source_cameras, args.test_every, args.trim_ends)
    mode = args.trajectory_mode.lower().strip()
    if mode in {"interpolate", "interp"}:
        if len(keyframes) < 2:
            poses = keyframes[:, :3, :4]
        else:
            interp_steps = args.interp_steps
            if interp_steps is None:
                interp_steps = max(1, int(math.ceil(args.frames / max(len(keyframes) - 1, 1))))
            poses = generate_interpolated_path(keyframes, interp_steps)
            if args.frames > 0 and len(poses) > args.frames:
                sample = np.linspace(0, len(poses) - 1, args.frames).round().astype(np.int64)
                poses = poses[sample]
    elif mode in {"ellipse", "elliptical"}:
        height = float(keyframes[:, 2, 3].mean()) + float(args.height_offset)
        poses = generate_ellipse_path_z(
            keyframes,
            n_frames=args.frames,
            variation=args.variation,
            height=height,
        )
    elif mode in {"circle", "circular", "orbit"}:
        poses = generate_circular_path(
            keyframes,
            n_frames=args.frames,
            radius_scale=args.radius_scale,
            height_offset=args.height_offset,
        )
    elif mode in {"sphere", "spherical"}:
        poses = generate_spherical_path(
            keyframes,
            n_frames=args.frames,
            radius_scale=args.radius_scale,
            elevation_deg=args.elevation_deg,
        )
    elif mode == "spiral":
        bounds = np.asarray(getattr(parser, "bounds", np.array([0.01, 1.0])), dtype=np.float32)
        extconf = getattr(parser, "extconf", {"spiral_radius_scale": 1.0})
        poses = generate_spiral_path(
            keyframes,
            bounds=bounds * scene_scale,
            n_frames=args.frames,
            n_rots=args.rotations,
            spiral_scale_r=float(extconf.get("spiral_radius_scale", 1.0)) * args.radius_scale,
        )
    else:
        raise ValueError(f"Unknown trajectory mode: {args.trajectory_mode}")
    return _as_camtoworlds(poses)


def _torch_load(path: Path, device: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_splats(ckpt_paths: List[Path], device: str) -> Dict[str, torch.Tensor]:
    loaded = []
    for ckpt_path in ckpt_paths:
        ckpt = _torch_load(ckpt_path, device)
        splats = ckpt.get("splats", ckpt)
        if not isinstance(splats, dict):
            raise ValueError(f"Checkpoint does not contain a splats dict: {ckpt_path}")
        loaded.append({key: value.to(device) for key, value in splats.items() if torch.is_tensor(value)})

    required = {"means", "quats", "scales", "opacities", "sh0", "shN"}
    missing = required.difference(loaded[0].keys())
    if missing:
        raise ValueError(f"Checkpoint is missing required 3DGS tensors: {sorted(missing)}")
    if len(loaded) == 1:
        return loaded[0]

    merged = {}
    for key in required:
        merged[key] = torch.cat([splats[key] for splats in loaded], dim=0)
    return merged


def infer_scene_id_from_ckpt(ckpt_path: Path) -> Optional[str]:
    ckpt_path = ckpt_path.resolve()
    if ckpt_path.parent.name == "ckpts" and ckpt_path.parent.parent.name:
        return ckpt_path.parent.parent.name
    parts = list(ckpt_path.parts)
    for marker in ("gaussian_splatting_baseline", "difix3d", "ours_difix3dplus", "freefix_self"):
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return None


def resolve_data_dir(args: argparse.Namespace) -> Path:
    if args.data_dir is not None:
        return Path(args.data_dir)
    scene_id = args.scene_id or infer_scene_id_from_ckpt(args.ckpt[0])
    if scene_id is None:
        raise ValueError("Could not infer scene id from --ckpt. Pass --data_dir or --scene_id explicitly.")
    data_dir = Path(args.dataset_root) / scene_id / "gaussian_splat"
    if not data_dir.exists():
        raise FileNotFoundError(f"Inferred data_dir does not exist: {data_dir}")
    print(f"[render] inferred data_dir={data_dir}")
    return data_dir


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return Path(args.output)
    result_dir = args.ckpt[0].resolve().parent.parent if args.ckpt[0].parent.name == "ckpts" else PROJECT_ROOT / "outputs" / "renders"
    stem = args.trajectory.stem if args.trajectory is not None else args.trajectory_mode
    match = re.search(r"ckpt_(\d+)", args.ckpt[0].name)
    step = match.group(1) if match is not None else "unknown"
    output = result_dir / "videos" / f"render_{stem}_step{step}.mp4"
    print(f"[render] inferred output={output}")
    return output


def _scale_intrinsics(k_mat: np.ndarray, src_size: Tuple[int, int], dst_size: Tuple[int, int]) -> np.ndarray:
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    scaled = k_mat.astype(np.float32).copy()
    scaled[0, :] *= float(dst_w) / max(float(src_w), 1.0)
    scaled[1, :] *= float(dst_h) / max(float(src_h), 1.0)
    return scaled


def resolve_intrinsics(
    args: argparse.Namespace,
    parser: ColmapParser,
    trajectory: LoadedTrajectory,
    n_frames: int,
) -> Tuple[np.ndarray, int, int]:
    parser_ks = getattr(parser, "Ks_dict", parser.ks_dict)
    first_camera_id = int(parser.camera_ids[0])
    base_k = np.asarray(parser_ks[first_camera_id], dtype=np.float32)
    base_w, base_h = parser.imsize_dict[first_camera_id]

    width = int(args.width or trajectory.width or base_w)
    height = int(args.height or trajectory.height or base_h)

    if args.fov_deg is not None:
        k_mat = _intrinsics_from_fov(width, height, args.fov_deg)
        ks = np.repeat(k_mat[None], n_frames, axis=0)
    elif args.fx is not None:
        fx = float(args.fx)
        fy = float(args.fy) if args.fy is not None else fx
        cx = float(args.cx) if args.cx is not None else width * 0.5
        cy = float(args.cy) if args.cy is not None else height * 0.5
        k_mat = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        ks = np.repeat(k_mat[None], n_frames, axis=0)
    elif trajectory.ks is not None:
        ks = trajectory.ks.astype(np.float32)
        if ks.shape[0] == 1:
            ks = np.repeat(ks, n_frames, axis=0)
        if ks.shape[0] != n_frames:
            raise ValueError(f"Trajectory has {ks.shape[0]} intrinsics but {n_frames} camera poses.")
        src_w = int(trajectory.width or width)
        src_h = int(trajectory.height or height)
        if (src_w, src_h) != (width, height):
            ks = np.stack([_scale_intrinsics(k, (src_w, src_h), (width, height)) for k in ks], axis=0)
    else:
        k_mat = _scale_intrinsics(base_k, (base_w, base_h), (width, height))
        ks = np.repeat(k_mat[None], n_frames, axis=0)

    return ks.astype(np.float32), width, height


def _depth_to_uint8(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    if not np.any(finite):
        return np.zeros((*depth.shape[:2], 3), dtype=np.uint8)
    lo = float(np.percentile(depth[finite], 2.0))
    hi = float(np.percentile(depth[finite], 98.0))
    if hi <= lo:
        hi = lo + 1e-6
    depth = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    gray = (depth * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def render_video(
    args: argparse.Namespace,
    splats: Dict[str, torch.Tensor],
    camtoworlds: np.ndarray,
    ks: np.ndarray,
    width: int,
    height: int,
) -> None:
    with torch.no_grad():
        device = args.device
        means = splats["means"]
        quats = splats["quats"]
        scales = torch.exp(splats["scales"])
        opacities = torch.sigmoid(splats["opacities"])
        colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
        inferred_sh_degree = int(round(math.sqrt(colors.shape[1]) - 1))
        sh_degree = args.sh_degree if args.sh_degree is not None else inferred_sh_degree

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(output), fps=args.fps, codec=args.codec, quality=args.quality, macro_block_size=1)

        rasterize_mode = "antialiased" if args.antialiased else "classic"
        render_mode = "RGB+ED" if args.depth else "RGB"
        print(
            f"[render] gaussians={means.shape[0]} frames={len(camtoworlds)} size={width}x{height} "
            f"sh_degree={sh_degree} mode={render_mode}"
        )

        try:
            for start in range(0, len(camtoworlds), args.batch_size):
                end = min(start + args.batch_size, len(camtoworlds))
                c2w = torch.from_numpy(camtoworlds[start:end]).float().to(device)
                k_batch = torch.from_numpy(ks[start:end]).float().to(device)
                raster_kwargs: Dict[str, Any] = {}
                if args.radius_clip > 0.0:
                    raster_kwargs["radius_clip"] = args.radius_clip
                renders, _, _ = rasterization(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=torch.linalg.inv(c2w),
                    Ks=k_batch,
                    width=width,
                    height=height,
                    packed=args.packed,
                    sparse_grad=False,
                    rasterize_mode=rasterize_mode,
                    distributed=False,
                    camera_model=args.camera_model,
                    sh_degree=sh_degree,
                    near_plane=args.near_plane,
                    far_plane=args.far_plane,
                    render_mode=render_mode,
                    **raster_kwargs,
                )
                renders = renders.detach().float().cpu().numpy()
                for j in range(renders.shape[0]):
                    rgb = np.clip(renders[j, ..., :3], 0.0, 1.0)
                    rgb_u8 = (rgb * 255.0).astype(np.uint8)
                    if args.depth:
                        depth_u8 = _depth_to_uint8(renders[j, ..., 3])
                        frame = np.concatenate([rgb_u8, depth_u8], axis=1)
                    else:
                        frame = rgb_u8
                    writer.append_data(frame)
        finally:
            writer.close()

    print(f"[render] wrote {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an MP4 from a 3DGS/gsplat checkpoint.")
    parser.add_argument("--ckpt", type=Path, nargs="+", required=True, help="One or more ckpt_*.pt files.")
    parser.add_argument("--data_dir", type=Path, default=None, help="COLMAP scene directory used for the checkpoint.")
    parser.add_argument("--scene_id", type=str, default=None, help="Optional scene id for DL3DV data_dir inference.")
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=PROJECT_ROOT / "data" / "DL3DV-10K-Benchmark",
        help="Dataset root used when --data_dir is omitted.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output .mp4 path.")
    parser.add_argument("--trajectory", type=Path, default=None, help="Optional .npy/.npz/.json camera path.")
    parser.add_argument(
        "--trajectory_mode",
        type=str,
        default="circle",
        choices=["interp", "interpolate", "ellipse", "elliptical", "circle", "circular", "orbit", "sphere", "spherical", "spiral"],
        help="Generated trajectory to use when --trajectory is not provided.",
    )
    parser.add_argument("--source_cameras", choices=["all", "train", "test", "val"], default="all")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--interp_steps", type=int, default=None)
    parser.add_argument("--trim_ends", type=int, default=5)
    parser.add_argument("--rotations", type=int, default=2)
    parser.add_argument("--radius_scale", type=float, default=1.0)
    parser.add_argument("--height_offset", type=float, default=0.0)
    parser.add_argument("--elevation_deg", type=float, default=20.0)
    parser.add_argument("--variation", type=float, default=0.0)
    parser.add_argument("--data_factor", type=int, default=4)
    parser.add_argument("--test_every", type=int, default=8)
    parser.add_argument("--global_scale", type=float, default=1.0)
    parser.add_argument("--normalize_world", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--align_principal_axes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fov_deg", type=float, default=None)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--camera_model", choices=["pinhole", "ortho", "fisheye"], default="pinhole")
    parser.add_argument("--near_plane", type=float, default=0.01)
    parser.add_argument("--far_plane", type=float, default=1e10)
    parser.add_argument("--sh_degree", type=int, default=None)
    parser.add_argument("--antialiased", action="store_true")
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--radius_clip", type=float, default=0.0)
    parser.add_argument("--depth", action="store_true", help="Write RGB and expected-depth side by side.")
    parser.add_argument("--codec", type=str, default="libx264")
    parser.add_argument("--quality", type=int, default=8)
    parser.add_argument("--save_trajectory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trajectory_out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_runtime_modules()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[render] CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"
    args.batch_size = max(1, int(args.batch_size))
    args.data_dir = resolve_data_dir(args)
    args.output = resolve_output_path(args)

    parser = ColmapParser(
        data_dir=str(args.data_dir),
        factor=args.data_factor,
        normalize=args.normalize_world,
        test_every=args.test_every,
        align_principal_axes=args.align_principal_axes,
    )
    scene_scale = parser.scene_scale * 1.1 * args.global_scale

    if args.trajectory is not None:
        trajectory = load_trajectory(args.trajectory)
    else:
        trajectory = LoadedTrajectory(camtoworlds=build_generated_trajectory(args, parser, scene_scale))

    camtoworlds = _as_camtoworlds(trajectory.camtoworlds)
    ks, width, height = resolve_intrinsics(args, parser, trajectory, len(camtoworlds))

    if args.save_trajectory:
        traj_out = args.trajectory_out or Path(args.output).with_suffix(".traj.npz")
        traj_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(traj_out, camtoworlds=camtoworlds, Ks=ks, width=width, height=height)
        print(f"[render] wrote trajectory {traj_out}")

    splats = load_splats(args.ckpt, args.device)
    render_video(args, splats, camtoworlds, ks, width, height)


if __name__ == "__main__":
    main()
