"""
Render trajectory helpers aligned with the official Difix3D gsplat code.
"""

from __future__ import annotations

import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    return np.stack([vec0, vec1, vec2, position], axis=1)


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    return np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]


def average_pose(poses: np.ndarray) -> np.ndarray:
    position = poses[:, :3, 3].mean(0)
    z_axis = poses[:, :3, 2].mean(0)
    up = poses[:, :3, 1].mean(0)
    return viewmatrix(z_axis, up, position)


def generate_spiral_path(
    poses: np.ndarray,
    bounds: np.ndarray,
    n_frames: int = 120,
    n_rots: int = 2,
    zrate: float = 0.5,
    spiral_scale_f: float = 1.0,
    spiral_scale_r: float = 1.0,
    focus_distance: float = 0.75,
) -> np.ndarray:
    near_bound = bounds.min()
    far_bound = bounds.max()
    focal = 1 / (((1 - focus_distance) / near_bound + focus_distance / far_bound))
    focal = focal * spiral_scale_f

    positions = poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), 90, axis=0)
    radii = radii * spiral_scale_r
    radii = np.concatenate([radii, [1.0]])

    render_poses = []
    cam2world = average_pose(poses)
    up = poses[:, :3, 1].mean(0)
    for theta in np.linspace(0.0, 2.0 * np.pi * n_rots, n_frames, endpoint=False):
        t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0]
        position = cam2world @ t
        lookat = cam2world @ [0, 0, -focal, 1.0]
        z_axis = position - lookat
        render_poses.append(viewmatrix(z_axis, up, position))
    return np.stack(render_poses, axis=0)


def generate_ellipse_path_z(
    poses: np.ndarray,
    n_frames: int = 120,
    variation: float = 0.0,
    phase: float = 0.0,
    height: float = 0.0,
) -> np.ndarray:
    center = focus_point_fn(poses)
    offset = np.array([center[0], center[1], height])

    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    low = -sc + offset
    high = sc + offset
    z_low = np.percentile(poses[:, :3, 3], 10, axis=0)
    z_high = np.percentile(poses[:, :3, 3], 90, axis=0)

    def get_positions(theta: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5),
                low[1] + (high - low)[1] * (np.sin(theta) * 0.5 + 0.5),
                variation
                * (
                    z_low[2]
                    + (z_high - z_low)[2] * (np.cos(theta + 2 * np.pi * phase) * 0.5 + 0.5)
                )
                + height,
            ],
            axis=-1,
        )

    theta = np.linspace(0, 2.0 * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)[:-1]

    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = int(np.argmax(np.abs(avg_up)))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return np.stack([viewmatrix(center - p, up, p) for p in positions], axis=0)


def generate_ellipse_path_y(
    poses: np.ndarray,
    n_frames: int = 120,
    variation: float = 0.0,
    phase: float = 0.0,
    height: float = 0.0,
) -> np.ndarray:
    center = focus_point_fn(poses)
    offset = np.array([center[0], height, center[2]])

    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    low = -sc + offset
    high = sc + offset
    y_low = np.percentile(poses[:, :3, 3], 10, axis=0)
    y_high = np.percentile(poses[:, :3, 3], 90, axis=0)

    def get_positions(theta: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5),
                variation
                * (
                    y_low[1]
                    + (y_high - y_low)[1] * (np.cos(theta + 2 * np.pi * phase) * 0.5 + 0.5)
                )
                + height,
                low[2] + (high - low)[2] * (np.sin(theta) * 0.5 + 0.5),
            ],
            axis=-1,
        )

    theta = np.linspace(0, 2.0 * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)[:-1]

    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = int(np.argmax(np.abs(avg_up)))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return np.stack([viewmatrix(p - center, up, p) for p in positions], axis=0)


def generate_interpolated_path(
    poses: np.ndarray,
    n_interp: int,
    spline_degree: int = 5,
    smoothness: float = 0.03,
    rot_weight: float = 0.1,
) -> np.ndarray:
    def poses_to_points(pose_mats: np.ndarray, dist: float) -> np.ndarray:
        pos = pose_mats[:, :3, -1]
        lookat = pose_mats[:, :3, -1] - dist * pose_mats[:, :3, 2]
        up = pose_mats[:, :3, -1] + dist * pose_mats[:, :3, 1]
        return np.stack([pos, lookat, up], axis=1)

    def points_to_poses(points: np.ndarray) -> np.ndarray:
        return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])

    def interp(points: np.ndarray, n: int, k: int, s: float) -> np.ndarray:
        from scipy import interpolate

        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        degree = min(k, sh[0] - 1)
        tck, _ = interpolate.splprep(pts.T, k=degree, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(interpolate.splev(u, tck))
        return np.reshape(new_points.T, (n, sh[1], sh[2]))

    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(
        points,
        n_interp * (points.shape[0] - 1),
        k=spline_degree,
        s=smoothness,
    )
    return points_to_poses(new_points)


__all__ = [
    "generate_ellipse_path_y",
    "generate_ellipse_path_z",
    "generate_interpolated_path",
    "generate_spiral_path",
]
