"""
COLMAP parsing utilities for standalone training scripts.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import pycolmap

from .data_normalize import normalize_cameras_and_points


def _get_rel_paths(path_dir: str) -> List[str]:
    paths = []
    for dp, _, fn in os.walk(path_dir):
        for name in fn:
            paths.append(os.path.relpath(os.path.join(dp, name), path_dir))
    return paths


def _get_camera_model_name(camera: object) -> str:
    model_name = getattr(camera, "model_name", None)
    if model_name is None:
        model_name = str(getattr(camera, "model", ""))
    return str(model_name).upper()


def _intrinsics_from_camera(camera: object) -> Tuple[float, float, float, float]:
    if (
        hasattr(camera, "focal_length_x")
        and hasattr(camera, "focal_length_y")
        and hasattr(camera, "principal_point_x")
        and hasattr(camera, "principal_point_y")
    ):
        fx = float(camera.focal_length_x)
        fy = float(camera.focal_length_y)
        cx = float(camera.principal_point_x)
        cy = float(camera.principal_point_y)
        return fx, fy, cx, cy

    params = np.array(getattr(camera, "params", []), dtype=np.float64)
    model = _get_camera_model_name(camera)
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[:3]
        return float(f), float(f), float(cx), float(cy)
    if model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
        return float(fx), float(fy), float(cx), float(cy)
    if model in {"SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"}:
        f, cx, cy = params[:3]
        return float(f), float(f), float(cx), float(cy)
    if model in {"RADIAL", "RADIAL_FISHEYE"}:
        f, cx, cy = params[:3]
        return float(f), float(f), float(cx), float(cy)
    if model in {"OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE"}:
        fx, fy, cx, cy = params[:4]
        return float(fx), float(fy), float(cx), float(cy)
    raise ValueError(f"Unsupported COLMAP camera model: {model}")


def _distortion_from_camera(camera: object) -> np.ndarray:
    model = _get_camera_model_name(camera)
    params = np.array(getattr(camera, "params", []), dtype=np.float32)

    if model in {"SIMPLE_PINHOLE", "PINHOLE"}:
        return np.empty(0, dtype=np.float32)
    if model in {"SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"}:
        if params.size < 4:
            return np.empty(0, dtype=np.float32)
        return np.array([params[3], 0.0, 0.0, 0.0], dtype=np.float32)
    if model in {"RADIAL", "RADIAL_FISHEYE"}:
        if params.size < 5:
            return np.empty(0, dtype=np.float32)
        return np.array([params[3], params[4], 0.0, 0.0], dtype=np.float32)
    if model in {"OPENCV", "FULL_OPENCV"}:
        if params.size < 8:
            return np.empty(0, dtype=np.float32)
        return np.array([params[4], params[5], params[6], params[7]], dtype=np.float32)
    # Fisheye and other unsupported undistortion models are left as-is.
    return np.empty(0, dtype=np.float32)


def _image_world_to_camera(image: object) -> np.ndarray:
    bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64).reshape(1, 4)

    def _to_pose(candidate: object) -> Optional[object]:
        if candidate is None:
            return None
        if callable(candidate):
            try:
                candidate = candidate()
            except TypeError:
                return None
        return candidate

    def _pose_to_matrix(pose: object) -> Optional[np.ndarray]:
        if pose is None:
            return None
        if hasattr(pose, "matrix"):
            mat_like = pose.matrix() if callable(pose.matrix) else pose.matrix
            mat = np.asarray(mat_like, dtype=np.float64)
            if mat.shape == (4, 4):
                return mat
            if mat.shape == (3, 4):
                return np.concatenate([mat, bottom], axis=0)
        if hasattr(pose, "rotation") and hasattr(pose.rotation, "matrix") and hasattr(pose, "translation"):
            rot = pose.rotation.matrix()
            trans = np.asarray(pose.translation, dtype=np.float64).reshape(3, 1)
            return np.concatenate([np.concatenate([rot, trans], axis=1), bottom], axis=0)
        return None

    pose_candidates: List[object] = []
    if hasattr(image, "cam_from_world"):
        pose_candidates.append(getattr(image, "cam_from_world"))
    if hasattr(image, "frame"):
        frame = getattr(image, "frame")
        for attr in ("rig_from_world", "sensor_from_world", "cam_from_world"):
            if hasattr(frame, attr):
                pose_candidates.append(getattr(frame, attr))
    for pose_candidate in pose_candidates:
        pose = _to_pose(pose_candidate)
        mat = _pose_to_matrix(pose)
        if mat is not None:
            return mat

    if hasattr(image, "R") and hasattr(image, "tvec"):
        rot = image.R()
        trans = np.asarray(image.tvec, dtype=np.float64).reshape(3, 1)
        return np.concatenate([np.concatenate([rot, trans], axis=1), bottom], axis=0)

    raise ValueError("Could not read COLMAP image pose from pycolmap image object.")


class ColmapParser:
    """
    Parse COLMAP reconstructions and image metadata into training-ready arrays.
    """

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = True,
        test_every: int = 8,
        align_principal_axes: bool = False,
    ) -> None:
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        self.align_principal_axes = align_principal_axes

        colmap_dir = os.path.join(data_dir, "sparse/0")
        if not os.path.exists(colmap_dir):
            alt = os.path.join(data_dir, "colmap/sparse/0")
            if os.path.exists(alt):
                colmap_dir = alt
        if not os.path.exists(colmap_dir):
            alt = os.path.join(data_dir, "sparse")
            if os.path.exists(alt):
                colmap_dir = alt
        if not os.path.exists(colmap_dir):
            raise FileNotFoundError(f"COLMAP directory not found under: {data_dir}")

        rec = pycolmap.Reconstruction(colmap_dir)

        images = rec.images
        cameras = rec.cameras
        w2c_mats: List[np.ndarray] = []
        camera_ids: List[int] = []
        image_names: List[str] = []
        ks_dict: Dict[int, np.ndarray] = {}
        params_dict: Dict[int, np.ndarray] = {}
        model_name_dict: Dict[int, str] = {}
        imsize_dict: Dict[int, Tuple[int, int]] = {}

        for image_id in images:
            im = images[image_id]
            w2c_mats.append(_image_world_to_camera(im))
            image_names.append(im.name)
            cam_id = int(im.camera_id)
            camera_ids.append(cam_id)

            cam = cameras[cam_id]
            fx, fy, cx, cy = _intrinsics_from_camera(cam)
            k_mat = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
            k_mat[:2, :] /= max(factor, 1)
            ks_dict[cam_id] = k_mat

            model_name = _get_camera_model_name(cam)
            model_name_dict[cam_id] = model_name
            params_dict[cam_id] = _distortion_from_camera(cam)

            width = int(getattr(cam, "width"))
            height = int(getattr(cam, "height"))
            imsize_dict[cam_id] = (max(width // max(factor, 1), 1), max(height // max(factor, 1), 1))

        if len(w2c_mats) == 0:
            raise ValueError("No images found in COLMAP reconstruction.")

        w2c_mats = np.stack(w2c_mats, axis=0)
        camtoworlds = np.linalg.inv(w2c_mats)

        # Ensure deterministic order by image name.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Resolve image directories and map COLMAP names to downsampled images.
        image_dir_suffix = f"_{factor}" if factor > 1 else ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        if not os.path.exists(image_dir):
            image_dir = colmap_image_dir
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.exists(colmap_image_dir):
            colmap_image_dir = image_dir

        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image.get(name, name)) for name in image_names]

        # 3D points.
        points3d = rec.points3D
        point_ids = list(points3d.keys())
        point_id_to_idx = {pid: i for i, pid in enumerate(point_ids)}
        if len(point_ids) == 0:
            points = np.empty((0, 3), dtype=np.float32)
            points_rgb = np.empty((0, 3), dtype=np.uint8)
            points_err = np.empty((0,), dtype=np.float32)
        else:
            points = np.array([points3d[pid].xyz for pid in point_ids], dtype=np.float32)
            points_rgb = np.array([points3d[pid].color for pid in point_ids], dtype=np.uint8)
            points_err = np.array([points3d[pid].error for pid in point_ids], dtype=np.float32)

        image_id_to_name = {int(img.image_id): img.name for img in rec.images.values()}
        point_indices: Dict[str, np.ndarray] = {}
        for pid in point_ids:
            point = points3d[pid]
            point_idx = point_id_to_idx[pid]
            for elem in point.track.elements:
                name = image_id_to_name.get(int(elem.image_id))
                if name is not None:
                    point_indices.setdefault(name, []).append(point_idx)
        point_indices = {k: np.array(v, dtype=np.int32) for k, v in point_indices.items()}

        transform = np.eye(4, dtype=np.float64)
        if normalize:
            camtoworlds, points, transform = normalize_cameras_and_points(
                camtoworlds,
                points=points,
                align_axes=align_principal_axes,
            )

        # Reconcile intrinsics with actual image sizes after optional downsampling.
        observed_size: Dict[int, Tuple[int, int]] = {}
        for img_path, cam_id in zip(image_paths, camera_ids):
            if cam_id in observed_size:
                continue
            arr = imageio.imread(img_path)[..., :3]
            observed_size[cam_id] = (arr.shape[1], arr.shape[0])

        for cam_id, (actual_w, actual_h) in observed_size.items():
            expected_w, expected_h = imsize_dict[cam_id]
            sx = actual_w / max(expected_w, 1)
            sy = actual_h / max(expected_h, 1)
            k_mat = ks_dict[cam_id].copy()
            k_mat[0, :] *= sx
            k_mat[1, :] *= sy
            ks_dict[cam_id] = k_mat
            imsize_dict[cam_id] = (actual_w, actual_h)

        # Optional undistortion maps for supported models.
        mapx_dict: Dict[int, np.ndarray] = {}
        mapy_dict: Dict[int, np.ndarray] = {}
        roi_undist_dict: Dict[int, Tuple[int, int, int, int]] = {}
        mask_dict: Dict[int, Optional[np.ndarray]] = {cam_id: None for cam_id in ks_dict.keys()}
        for cam_id, params in params_dict.items():
            if len(params) == 0:
                continue
            model = model_name_dict.get(cam_id, "")
            if model not in {"SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV"}:
                continue

            k_mat = ks_dict[cam_id]
            width, height = imsize_dict[cam_id]
            k_undist, roi = cv2.getOptimalNewCameraMatrix(k_mat, params, (width, height), 0)
            mapx, mapy = cv2.initUndistortRectifyMap(
                k_mat,
                params,
                None,
                k_undist,
                (width, height),
                cv2.CV_32FC1,
            )
            ks_dict[cam_id] = k_undist.astype(np.float32)
            mapx_dict[cam_id] = mapx
            mapy_dict[cam_id] = mapy
            roi_undist_dict[cam_id] = tuple(int(v) for v in roi)

        self.image_names = image_names
        self.image_paths = image_paths
        self.alpha_mask_paths = None
        self.camtoworlds = camtoworlds.astype(np.float32)
        self.camera_ids = camera_ids
        self.num_cameras = len(set(camera_ids))
        self.ks_dict = ks_dict
        self.Ks_dict = ks_dict  # FreeFix-style alias
        self.params_dict = params_dict
        self.model_name_dict = model_name_dict
        self.imsize_dict = imsize_dict
        self.points = points.astype(np.float32)
        self.points_rgb = points_rgb
        self.points_err = points_err
        self.point_indices = point_indices
        self.transform = transform.astype(np.float32)
        self.mapx_dict = mapx_dict
        self.mapy_dict = mapy_dict
        self.roi_undist_dict = roi_undist_dict
        self.mask_dict = mask_dict

        cam_positions = self.camtoworlds[:, :3, 3]
        center = np.mean(cam_positions, axis=0)
        dists = np.linalg.norm(cam_positions - center, axis=1)
        self.scene_scale = float(np.max(dists) + 1e-8)


# Compatibility aliases.
Parser = ColmapParser
