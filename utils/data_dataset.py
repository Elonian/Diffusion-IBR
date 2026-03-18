"""
Dataset wrappers built on top of ColmapParser.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset

from .data_colmap import ColmapParser


class ColmapImageDataset(Dataset):
    """
    Image dataset for COLMAP scenes with train/test split, random crop, and optional depth samples.
    """

    def __init__(
        self,
        parser: ColmapParser,
        split: str = "train",
        patch_size: Optional[int] = None,
        train_split_all: bool = True,
        partition_file: Optional[str] = None,
        load_depths: bool = False,
        split_strategy: str = "auto",
    ) -> None:
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths

        idx = np.arange(len(parser.image_names))
        if partition_file is not None:
            with open(partition_file, "r", encoding="utf-8") as f:
                partition = json.load(f)
            if split == "train":
                self.indices = np.array(partition["train"], dtype=np.int64)
            elif split in {"test", "val"}:
                self.indices = np.array(partition["test"], dtype=np.int64)
            else:
                raise ValueError(f"Unknown split: {split}")
        else:
            if split_strategy == "auto":
                image_names = list(getattr(parser, "image_names", []))
                if (
                    int(getattr(parser, "test_every", 8)) == 1
                    and any("_train_" in name or "_eval_" in name for name in image_names)
                ):
                    split_strategy = "train_eval_tags"
                elif train_split_all:
                    split_strategy = "freefix"
                else:
                    split_strategy = "difix3d"

            if split_strategy not in {"freefix", "difix3d", "all", "train_eval_tags"}:
                raise ValueError(f"Unknown split_strategy: {split_strategy}")

            test_every = int(getattr(parser, "test_every", 8))
            if test_every == 0 or split_strategy == "all":
                self.indices = idx
            elif split_strategy == "train_eval_tags":
                names = list(getattr(parser, "image_names", []))
                if split == "train":
                    self.indices = np.array([i for i in idx if "_train_" in names[i]], dtype=np.int64)
                elif split in {"test", "val"}:
                    self.indices = np.array([i for i in idx if "_eval_" in names[i]], dtype=np.int64)
                else:
                    raise ValueError(f"Unknown split: {split}")
            elif split_strategy == "difix3d":
                if split == "train":
                    self.indices = idx[idx % max(test_every, 1) == 0]
                elif split in {"test", "val"}:
                    self.indices = idx[idx % max(test_every, 1) != 0]
                else:
                    raise ValueError(f"Unknown split: {split}")
            elif split == "train":
                # FreeFix-style COLMAP split: train on all images by default.
                self.indices = idx
            elif split in {"test", "val"}:
                self.indices = idx[idx % max(test_every, 1) == 0]
            else:
                raise ValueError(f"Unknown split: {split}")

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = int(self.indices[item])
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = int(self.parser.camera_ids[index])
        k_mat = self.parser.ks_dict[camera_id].copy()
        camtoworld = self.parser.camtoworlds[index]
        x0, y0 = 0, 0

        if (
            camera_id in self.parser.mapx_dict
            and camera_id in self.parser.mapy_dict
            and camera_id in self.parser.roi_undist_dict
        ):
            mapx = self.parser.mapx_dict[camera_id]
            mapy = self.parser.mapy_dict[camera_id]
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]
            k_mat[0, 2] -= x
            k_mat[1, 2] -= y

        if self.patch_size is not None:
            h, w = image.shape[:2]
            x0 = int(np.random.randint(0, max(w - self.patch_size, 1)))
            y0 = int(np.random.randint(0, max(h - self.patch_size, 1)))
            image = image[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
            k_mat[0, 2] -= x0
            k_mat[1, 2] -= y0

        data: Dict[str, Any] = {
            "K": torch.from_numpy(k_mat).float(),
            "camtoworld": torch.from_numpy(camtoworld).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": torch.tensor(item, dtype=torch.long),
            "camera_idx": torch.tensor(camera_id, dtype=torch.long),
            "image_name": str(self.parser.image_names[index]),
            "image_path": str(self.parser.image_paths[index]),
        }

        mask_dict = getattr(self.parser, "mask_dict", {})
        if isinstance(mask_dict, dict):
            mask = mask_dict.get(camera_id)
            if mask is not None:
                if self.patch_size is not None:
                    mask = mask[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
                data["mask"] = torch.from_numpy(mask).bool()

        alpha_mask_paths = getattr(self.parser, "alpha_mask_paths", None)
        if alpha_mask_paths is not None and index < len(alpha_mask_paths):
            alpha_path = alpha_mask_paths[index]
            if alpha_path is not None and os.path.exists(alpha_path):
                alpha_mask = imageio.imread(alpha_path)
                if alpha_mask.ndim == 2:
                    alpha_mask = alpha_mask[..., None]
                else:
                    alpha_mask = alpha_mask[..., :1]
                alpha_mask = alpha_mask.astype(np.float32) / 255.0
                if self.patch_size is not None:
                    alpha_mask = alpha_mask[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
                data["alpha_mask"] = torch.from_numpy(alpha_mask).float()

        if self.load_depths:
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices.get(image_name, np.empty((0,), dtype=np.int32))
            if point_indices.shape[0] > 0:
                world_to_cam = np.linalg.inv(camtoworld)
                points_world = self.parser.points[point_indices]
                points_cam = (world_to_cam[:3, :3] @ points_world.T + world_to_cam[:3, 3:4]).T
                points_proj = (k_mat @ points_cam.T).T
                points_2d = points_proj[:, :2] / np.maximum(points_proj[:, 2:3], 1e-8)
                depths = points_cam[:, 2]
                if self.patch_size is not None:
                    points_2d[:, 0] -= x0
                    points_2d[:, 1] -= y0

                valid = (
                    (points_2d[:, 0] >= 0)
                    & (points_2d[:, 0] < image.shape[1])
                    & (points_2d[:, 1] >= 0)
                    & (points_2d[:, 1] < image.shape[0])
                    & (depths > 0)
                )
                points_2d = points_2d[valid]
                depths = depths[valid]
            else:
                points_2d = np.empty((0, 2), dtype=np.float32)
                depths = np.empty((0,), dtype=np.float32)

            data["points"] = torch.from_numpy(points_2d.astype(np.float32))
            data["depths"] = torch.from_numpy(depths.astype(np.float32))

        return data


# Compatibility alias.
Dataset = ColmapImageDataset
