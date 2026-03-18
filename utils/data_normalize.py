"""
Geometry normalization helpers for COLMAP-style datasets.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def similarity_from_cameras(
    camtoworlds: np.ndarray,
    strict_scaling: bool = False,
    center_method: str = "focus",
) -> np.ndarray:
    """
    Estimate a similarity transform from camera poses.

    Args:
        camtoworlds: (N, 4, 4) camera-to-world matrices.
        strict_scaling: Use max camera distance instead of median.
        center_method: "focus" or "poses".

    Returns:
        (4, 4) similarity transform that recenters and rescales the scene.
    """
    t = camtoworlds[:, :3, 3]
    r = camtoworlds[:, :3, :3]

    # Rotate world so average camera up axis maps to y- (OpenCV-style).
    ups = np.sum(r * np.array([0.0, -1.0, 0.0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up) + 1e-8

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = float((up_camspace * world_up).sum())
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    if c > -1.0:
        r_align = np.eye(3, dtype=np.float64) + skew + (skew @ skew) * (1.0 / (1.0 + c))
    else:
        r_align = np.array(
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    r = r_align @ r
    fwds = np.sum(r * np.array([0.0, 0.0, 1.0]), axis=-1)
    t = (r_align @ t[..., None])[..., 0]

    if center_method == "focus":
        nearest = t + (fwds * -t).sum(-1)[:, None] * fwds
        translate = -np.median(nearest, axis=0)
    elif center_method == "poses":
        translate = -np.median(t, axis=0)
    else:
        raise ValueError(f"Unknown center_method: {center_method}")

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = r_align
    transform[:3, 3] = translate

    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / (scale_fn(np.linalg.norm(t + translate, axis=-1)) + 1e-8)
    transform[:3, :] *= scale
    return transform


def align_principle_axes(point_cloud: np.ndarray) -> np.ndarray:
    """
    Build an SE(3) transform that aligns principal axes of a point cloud.
    """
    if point_cloud.shape[0] == 0:
        return np.eye(4, dtype=np.float64)

    centroid = np.median(point_cloud, axis=0)
    centered = point_cloud - centroid
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Descending eigenvalues => dominant axis first.
    order = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, order]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1

    rotation = eigenvectors.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ centroid
    return transform


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Apply an SE(3)/similarity transform to 3D points.
    """
    if points.shape[0] == 0:
        return points.copy()
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def transform_cameras(matrix: np.ndarray, camtoworlds: np.ndarray) -> np.ndarray:
    """
    Apply an SE(3)/similarity transform to camera-to-world matrices.
    """
    out = np.einsum("nij,ki->nkj", camtoworlds, matrix)
    scaling = np.linalg.norm(out[:, 0, :3], axis=1)
    scaling = np.where(scaling > 1e-8, scaling, 1.0)
    out[:, :3, :3] = out[:, :3, :3] / scaling[:, None, None]
    return out


def normalize_cameras_and_points(
    camtoworlds: np.ndarray,
    points: Optional[np.ndarray] = None,
    align_axes: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Normalize camera poses, optionally with 3D points for principal-axis alignment.

    Returns:
        normalized_c2w, normalized_points (or None), total_transform
    """
    t1 = similarity_from_cameras(camtoworlds)
    camtoworlds = transform_cameras(t1, camtoworlds)
    total = t1

    if points is None:
        return camtoworlds, None, total

    points = transform_points(t1, points)
    if align_axes:
        t2 = align_principle_axes(points)
        camtoworlds = transform_cameras(t2, camtoworlds)
        points = transform_points(t2, points)
        total = t2 @ total
    return camtoworlds, points, total
