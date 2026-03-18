"""
Standalone 3DGS trainer with optional FreeFix- and Difix3D+-style iterative refinement.

The implementation is self-contained in this repo, while exposing training recipes
that mirror the core iterative behaviors used by FreeFix and Difix3D+.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy
from PIL import Image
from scipy.spatial.transform import Rotation
from torch import Tensor
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.priors.fixer import DiffusionFixer
from utils import (
    ColmapImageDataset,
    ColmapParser,
    rotation_matrix_to_quaternion,
    compute_psnr as _compute_psnr,
    interpolate_pose as _interpolate_pose,
    knn,
    parse_float_csv as _parse_float_csv,
    parse_name_csv as _parse_name_csv,
    parse_steps_csv as _parse_steps_csv,
    rgb_to_sh,
    set_random_seed,
    simple_ssim as _simple_ssim,
    soft_sigmoid as _soft_sigmoid,
)


@dataclass
class Config:
    data_dir: str
    result_dir: str
    data_factor: int = 4
    test_every: int = 8
    normalize_world: bool = True
    normalize_align_axes: bool = False
    partition_file: Optional[str] = None
    patch_size: Optional[int] = None
    batch_size: int = 1
    num_workers: int = 4
    max_steps: int = 30000
    eval_every: int = 10000
    save_every: int = 10000
    eval_steps: Optional[str] = None
    save_steps: Optional[str] = None
    train_split_all: bool = True
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    ssim_lambda: float = 0.2
    random_bkgd: bool = False
    strategy_prune_opa: float = 0.005
    strategy_grow_grad2d: float = 0.0002
    strategy_grow_scale3d: float = 0.01
    strategy_prune_scale3d: float = 0.1
    strategy_refine_start_iter: int = 500
    strategy_refine_stop_iter: int = 15000
    strategy_reset_every: int = 3000
    strategy_refine_every: int = 100
    near_plane: float = 0.01
    far_plane: float = 1e10
    packed: bool = False
    sparse_grad: bool = False
    absgrad: bool = False
    antialiased: bool = False
    init_type: str = "sfm"
    init_num_pts: int = 100000
    init_extent: float = 3.0
    init_opa: float = 0.1
    init_scale: float = 1.0
    global_scale: float = 1.0
    means_lr: float = 1.6e-4 / 10
    scales_lr: float = 5e-3 / 5
    quats_lr: float = 1e-3 / 5
    opacities_lr: float = 5e-2 / 5
    sh0_lr: float = 2.5e-3 / 50
    shN_lr: float = 2.5e-3 / 20 / 50
    seed: int = 42
    device: str = "cuda"
    mode: str = "train"  # train | eval | render | train_eval
    ckpt: Optional[str] = None
    render_frames: int = 120
    render_fps: int = 24

    training_recipe: str = "vanilla"  # vanilla | freefix | difix3d
    use_freefix: bool = False
    fix_steps: Optional[str] = None
    fix_cache_dir: Optional[str] = None
    lazy_fixer_init: bool = True

    use_difix: bool = False
    difix_model_id: str = "nvidia/difix_ref"
    difix_model_path: Optional[str] = None
    difix_cache_dir: Optional[str] = None
    difix_prompt: str = "remove degradation"
    difix_start_step: int = 3000
    difix_fix_every: int = 3000
    difix_num_views: int = 0
    difix_num_inference_steps: int = 1
    difix_timestep: int = 199
    difix_guidance_scale: float = 0.0
    difix_novel_prob: float = 0.3
    difix_novel_lambda: float = 0.3
    difix_progressive_updates: bool = True
    difix_progressive_pose_step: float = 0.5
    difix_post_render: bool = False
    difix_post_prompt: Optional[str] = None
    difix_post_num_inference_steps: int = 1
    difix_post_timestep: int = 199
    difix_post_guidance_scale: float = 0.0

    freefix_backend: str = "flux"  # flux | sdxl
    freefix_prompt: str = (
        "A photorealistic real-world scene with consistent geometry, detailed textures, and natural lighting."
    )
    freefix_negative_prompt: Optional[str] = (
        "blurry, low quality, foggy, overall gray, subtitles, incomplete, ghost image, too close to camera"
    )
    freefix_start_step: int = 30000
    freefix_fix_every: int = 400
    freefix_num_views: int = 0
    freefix_num_inference_steps: int = 50
    freefix_strength: float = 0.6
    freefix_guidance_scale: float = 3.5
    freefix_novel_prob: float = 0.1
    freefix_novel_lambda: float = 0.2
    freefix_real_lambda: float = 1.0
    freefix_certainty_scales: str = "0.001,0.01,0.1"
    freefix_hessian_attrs: str = "means,quats,scales"
    freefix_mask_center: float = 0.5
    freefix_mask_softness: float = 10.0
    freefix_mask_scheduler: str = "0.3,0.9,1.0"
    freefix_guide_ratio: float = 1.0
    freefix_warp_ratio: float = 0.5


def create_splats_with_optimizers(
    parser: ColmapParser,
    cfg: Config,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if cfg.init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif cfg.init_type == "random":
        points = cfg.init_extent * parser.scene_scale * (torch.rand((cfg.init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((cfg.init_num_pts, 3))
    else:
        raise ValueError("init_type must be sfm or random")

    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(dist2_avg + 1e-9)
    scales = torch.log(dist_avg * cfg.init_scale).unsqueeze(-1).repeat(1, 3)
    n = points.shape[0]
    quats = torch.rand((n, 4))
    opacities = torch.logit(torch.full((n,), cfg.init_opa))

    colors = torch.zeros((n, (cfg.sh_degree + 1) ** 2, 3))
    colors[:, 0, :] = rgb_to_sh(rgbs)

    params = [
        ("means", torch.nn.Parameter(points), cfg.means_lr * parser.scene_scale * cfg.global_scale),
        ("scales", torch.nn.Parameter(scales), cfg.scales_lr),
        ("quats", torch.nn.Parameter(quats), cfg.quats_lr),
        ("opacities", torch.nn.Parameter(opacities), cfg.opacities_lr),
        ("sh0", torch.nn.Parameter(colors[:, :1, :]), cfg.sh0_lr),
        ("shN", torch.nn.Parameter(colors[:, 1:, :]), cfg.shN_lr),
    ]

    splats = torch.nn.ParameterDict({n_: p for n_, p, _ in params}).to(cfg.device)
    optimizers = {
        name: (torch.optim.SparseAdam if cfg.sparse_grad else torch.optim.Adam)(
            [{"params": splats[name], "lr": lr * math.sqrt(cfg.batch_size)}],
            eps=1e-15 / math.sqrt(cfg.batch_size),
            betas=(1 - cfg.batch_size * (1 - 0.9), 1 - cfg.batch_size * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


class Trainer:
    def __init__(self, cfg: Config) -> None:
        set_random_seed(cfg.seed)
        self.cfg = cfg
        self.device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
        self.cfg.device = self.device

        self.result_dir = Path(cfg.result_dir)
        self.ckpt_dir = self.result_dir / "ckpts"
        self.stats_dir = self.result_dir / "stats"
        self.render_dir = self.result_dir / "renders"
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        self.render_dir.mkdir(parents=True, exist_ok=True)

        self.training_recipe = cfg.training_recipe.lower().strip()
        if self.training_recipe == "vanilla":
            if cfg.use_difix:
                self.training_recipe = "difix3d"
            elif cfg.use_freefix:
                self.training_recipe = "freefix"
        if self.training_recipe not in {"vanilla", "freefix", "difix3d"}:
            raise ValueError("training_recipe must be one of: vanilla, freefix, difix3d")
        self.cfg.training_recipe = self.training_recipe
        split_strategy = "difix3d" if self.training_recipe == "difix3d" else "freefix"

        self.parser = ColmapParser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world,
            test_every=cfg.test_every,
            align_principal_axes=cfg.normalize_align_axes,
        )
        self.trainset = ColmapImageDataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            train_split_all=cfg.train_split_all,
            partition_file=cfg.partition_file,
            split_strategy=split_strategy,
        )
        self.testset = ColmapImageDataset(
            self.parser,
            split="test",
            patch_size=None,
            train_split_all=cfg.train_split_all,
            partition_file=cfg.partition_file,
            split_strategy=split_strategy,
        )
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        self.train_indices = np.array(self.trainset.indices, dtype=np.int64)
        self.train_pose_bank = self.parser.camtoworlds[self.train_indices].astype(np.float32)
        # Official Difix3D progressive pose update starts from train poses.
        self.current_novel_poses = self.train_pose_bank.copy()
        self.difix_progressive_pose_bank: Dict[int, np.ndarray] = {}
        self.ref_image_cache: Dict[Tuple[int, int, int], Image.Image] = {}

        self.splats, self.optimizers = create_splats_with_optimizers(self.parser, cfg)
        self.strategy = DefaultStrategy(
            verbose=True,
            prune_opa=cfg.strategy_prune_opa,
            grow_grad2d=cfg.strategy_grow_grad2d,
            grow_scale3d=cfg.strategy_grow_scale3d,
            prune_scale3d=cfg.strategy_prune_scale3d,
            refine_start_iter=cfg.strategy_refine_start_iter,
            refine_stop_iter=cfg.strategy_refine_stop_iter,
            reset_every=cfg.strategy_reset_every,
            refine_every=cfg.strategy_refine_every,
            absgrad=cfg.absgrad,
            revised_opacity=False,
        )
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.strategy.initialize_state(scene_scale=self.scene_scale)

        self.novel_loader: Optional[DataLoader] = None
        self.novel_loader_iter = None
        self.novel_image_paths: List[str] = []
        self.novel_c2ws: List[np.ndarray] = []
        self.novel_camera_ids: List[int] = []
        self.fix_steps = _parse_steps_csv(cfg.fix_steps)
        self.eval_steps = _parse_steps_csv(cfg.eval_steps)
        self.save_steps = _parse_steps_csv(cfg.save_steps)
        self.freefix_certainty_scales = _parse_float_csv(
            cfg.freefix_certainty_scales,
            default=[0.001, 0.01, 0.1],
        )
        self.freefix_hessian_attrs = _parse_name_csv(
            cfg.freefix_hessian_attrs,
            default=["means", "quats", "scales"],
        )
        self.freefix_mask_schedule_ratios = _parse_float_csv(
            cfg.freefix_mask_scheduler,
            default=[0.3, 0.9, 1.0],
        )

        self.fixer: Optional[DiffusionFixer] = None
        self.fixer_enabled = self.training_recipe in {"difix3d", "freefix"}
        if self.training_recipe == "difix3d":
            print(
                f"[trainer] recipe=difix3d model={cfg.difix_model_path or cfg.difix_model_id} "
                f"start={cfg.difix_start_step} every={cfg.difix_fix_every} "
                f"progressive={int(cfg.difix_progressive_updates)} "
                f"post_render={int(cfg.difix_post_render)} "
                f"lazy_init={int(cfg.lazy_fixer_init)}"
            )
        elif self.training_recipe == "freefix":
            print(
                f"[trainer] recipe=freefix backend={cfg.freefix_backend} "
                f"start={cfg.freefix_start_step} every={cfg.freefix_fix_every} "
                f"lazy_init={int(cfg.lazy_fixer_init)}"
            )

        if self.fixer_enabled and not cfg.lazy_fixer_init:
            self._ensure_fixer()

    def _get_novel_sampling_prob(self) -> float:
        if self.training_recipe == "difix3d":
            return self.cfg.difix_novel_prob
        if self.training_recipe == "freefix":
            return self.cfg.freefix_novel_prob
        return 0.0

    def _get_novel_loss_scale(self) -> float:
        if self.training_recipe == "difix3d":
            return self.cfg.difix_novel_lambda
        if self.training_recipe == "freefix":
            return self.cfg.freefix_novel_lambda
        return 1.0

    def _get_real_loss_scale(self) -> float:
        if self.training_recipe == "freefix":
            return self.cfg.freefix_real_lambda
        if self.training_recipe == "difix3d":
            return 1.5
        return 1.0

    def _get_fix_trigger(self) -> Tuple[int, int]:
        if self.training_recipe == "difix3d":
            return self.cfg.difix_start_step, self.cfg.difix_fix_every
        if self.training_recipe == "freefix":
            return self.cfg.freefix_start_step, self.cfg.freefix_fix_every
        return 0, 0

    def _ensure_fixer(self) -> Optional[DiffusionFixer]:
        if not self.fixer_enabled:
            return None
        if self.fixer is not None:
            return self.fixer

        cfg = self.cfg
        cache_dir = cfg.fix_cache_dir or cfg.difix_cache_dir
        if self.training_recipe == "difix3d":
            self.fixer = DiffusionFixer(
                backend="difix",
                cache_dir=cache_dir,
                device=self.device,
                difix_model_id=cfg.difix_model_id,
                difix_model_path=cfg.difix_model_path,
            )
        elif self.training_recipe == "freefix":
            self.fixer = DiffusionFixer(
                backend=cfg.freefix_backend,
                cache_dir=cache_dir,
                device=self.device,
                difix_model_id=cfg.difix_model_id,
                difix_model_path=cfg.difix_model_path,
            )
        return self.fixer

    def _build_freefix_infer_steps(self) -> int:
        return max(1, int(self.cfg.freefix_num_inference_steps * self.cfg.freefix_strength))

    def _build_freefix_mask_scheduler(self, mask_count: int) -> List[int]:
        infer_steps = self._build_freefix_infer_steps()
        scheduler = [
            max(1, min(infer_steps, int(infer_steps * ratio)))
            for ratio in self.freefix_mask_schedule_ratios
        ]
        if len(scheduler) < mask_count:
            scheduler.extend([infer_steps] * (mask_count - len(scheduler)))
        return scheduler[:mask_count]

    def _build_freefix_guide_until(self) -> int:
        infer_steps = self._build_freefix_infer_steps()
        return max(0, min(infer_steps, int(infer_steps * self.cfg.freefix_guide_ratio)))

    def _build_freefix_warp_until(self) -> int:
        infer_steps = self._build_freefix_infer_steps()
        return max(0, min(infer_steps, int(infer_steps * self.cfg.freefix_warp_ratio)))

    @staticmethod
    def _matches_schedule(step: int, schedule: set[int]) -> bool:
        # Step lists are provided in human-readable 1-based indexing.
        return (step + 1) in schedule

    def _should_run_fix(self, step: int) -> bool:
        if not self.fixer_enabled:
            return False
        # When an explicit schedule is provided, only scheduled steps trigger.
        if self.fix_steps:
            return self._matches_schedule(step, self.fix_steps)
        start_step, fix_every = self._get_fix_trigger()
        if fix_every <= 0:
            return False
        return (step + 1) >= start_step and ((step + 1 - start_step) % fix_every == 0)

    @staticmethod
    def _pose_rotation_distance(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
        q1 = rotation_matrix_to_quaternion(pose_a[:3, :3]).astype(np.float64)
        q2 = rotation_matrix_to_quaternion(pose_b[:3, :3]).astype(np.float64)
        if np.dot(q1, q2) < 0:
            q2 = -q2
        return float(np.arccos(np.clip(2.0 * np.dot(q1, q2) ** 2 - 1.0, -1.0, 1.0)))

    def _pose_distance(self, pose_a: np.ndarray, pose_b: np.ndarray) -> float:
        translation_dist = float(np.linalg.norm(pose_a[:3, 3] - pose_b[:3, 3]))
        rotation_dist = self._pose_rotation_distance(pose_a, pose_b)
        return translation_dist + rotation_dist

    def _nearest_train_global_index(self, c2w: np.ndarray) -> int:
        if self.train_pose_bank.shape[0] == 0:
            return 0
        distances = [self._pose_distance(pose, c2w) for pose in self.train_pose_bank]
        local_idx = int(np.argmin(distances))
        return int(self.train_indices[local_idx])

    def _load_train_reference_image(self, c2w: np.ndarray, width: int, height: int) -> Image.Image:
        train_idx = self._nearest_train_global_index(c2w)
        cache_key = (train_idx, width, height)
        cached = self.ref_image_cache.get(cache_key)
        if cached is None:
            image_path = self.parser.image_paths[train_idx]
            with Image.open(image_path) as image:
                cached = image.convert("RGB").resize((width, height), Image.LANCZOS)
            self.ref_image_cache[cache_key] = cached
        return cached.copy()

    def _run_difix_fixer(
        self,
        pred_pil: Image.Image,
        ref_pil: Image.Image,
        *,
        seed: int,
        post_render: bool,
    ) -> Image.Image:
        fixer = self._ensure_fixer()
        if fixer is None:
            return pred_pil
        if post_render:
            prompt = self.cfg.difix_post_prompt if self.cfg.difix_post_prompt else self.cfg.difix_prompt
            num_inference_steps = self.cfg.difix_post_num_inference_steps
            timestep = self.cfg.difix_post_timestep
            guidance_scale = self.cfg.difix_post_guidance_scale
        else:
            prompt = self.cfg.difix_prompt
            num_inference_steps = self.cfg.difix_num_inference_steps
            timestep = self.cfg.difix_timestep
            guidance_scale = self.cfg.difix_guidance_scale
        return fixer(
            prompt=prompt,
            image=pred_pil,
            ref_image=ref_pil,
            num_inference_steps=num_inference_steps,
            timestep=timestep,
            guidance_scale=guidance_scale,
            seed=seed,
        )

    def _use_difix_post_render(self) -> bool:
        return self.training_recipe == "difix3d" and self.cfg.difix_post_render

    def _apply_difix_post_render(
        self,
        pred_u8: np.ndarray,
        c2w: np.ndarray,
        *,
        step: int,
        frame_index: int,
    ) -> np.ndarray:
        if not self._use_difix_post_render():
            return pred_u8
        pred_pil = Image.fromarray(pred_u8, mode="RGB")
        height, width = pred_u8.shape[0], pred_u8.shape[1]
        ref_pil = self._load_train_reference_image(c2w=c2w, width=width, height=height)
        seed = self.cfg.seed + step * 1000 + frame_index
        fixed_pil = self._run_difix_fixer(
            pred_pil=pred_pil,
            ref_pil=ref_pil,
            seed=seed,
            post_render=True,
        )
        return np.array(fixed_pil.convert("RGB"), dtype=np.uint8)

    def _get_progressive_render_pose(self, target_idx: int) -> np.ndarray:
        target_c2w = self.parser.camtoworlds[target_idx].astype(np.float32)
        if self.training_recipe != "difix3d" or not self.cfg.difix_progressive_updates:
            return target_c2w.copy()
        pose_step = float(self.cfg.difix_progressive_pose_step)
        if pose_step <= 0.0:
            return target_c2w.copy()
        if pose_step >= 1.0:
            self.difix_progressive_pose_bank[target_idx] = target_c2w.copy()
            return target_c2w.copy()

        if target_idx not in self.difix_progressive_pose_bank:
            nearest_idx = self._nearest_train_global_index(target_c2w)
            current_pose = self.parser.camtoworlds[nearest_idx].astype(np.float32)
        else:
            current_pose = self.difix_progressive_pose_bank[target_idx]

        pose_distance = self._pose_distance(current_pose, target_c2w)
        if pose_distance <= pose_step:
            self.difix_progressive_pose_bank[target_idx] = target_c2w.copy()
            return target_c2w.copy()

        translation_vec = target_c2w[:3, 3] - current_pose[:3, 3]
        translation_norm = float(np.linalg.norm(translation_vec))
        if translation_norm > 1e-6:
            translation_step = (translation_vec / translation_norm) * pose_step
            new_translation = current_pose[:3, 3] + translation_step
            interp_ratio = min(pose_step / translation_norm, 1.0)
        else:
            new_translation = target_c2w[:3, 3].copy()
            interp_ratio = 1.0

        if np.dot(new_translation - current_pose[:3, 3], translation_vec) <= 0.0 or np.linalg.norm(
            new_translation - target_c2w[:3, 3]
        ) <= pose_step:
            new_translation = target_c2w[:3, 3]

        next_pose = _interpolate_pose(current_pose, target_c2w, interp_ratio)
        next_pose[:3, 3] = new_translation
        self.difix_progressive_pose_bank[target_idx] = next_pose
        return next_pose

    @staticmethod
    def _interpolate_rotation(rot_a: np.ndarray, rot_b: np.ndarray, t: float) -> np.ndarray:
        q1 = Rotation.from_matrix(rot_a).as_quat()
        q2 = Rotation.from_matrix(rot_b).as_quat()
        if np.dot(q1, q2) < 0:
            q2 = -q2
        dot_product = float(np.clip(np.dot(q1, q2), -1.0, 1.0))
        theta = float(np.arccos(dot_product))
        if abs(theta) < 1e-6:
            q_interp = (1.0 - t) * q1 + t * q2
        else:
            q_interp = (np.sin((1.0 - t) * theta) * q1 + np.sin(t * theta) * q2) / np.sin(theta)
        q_interp = q_interp / np.linalg.norm(q_interp)
        return Rotation.from_quat(q_interp).as_matrix().astype(np.float32)

    def _find_nearest_assignments(self, source_poses: np.ndarray, target_poses: np.ndarray) -> List[int]:
        assignments: List[int] = []
        for target_pose in target_poses:
            distances = [self._pose_distance(src_pose, target_pose) for src_pose in source_poses]
            assignments.append(int(np.argmin(distances)))
        return assignments

    def _shift_poses_toward_targets(
        self,
        source_poses: np.ndarray,
        target_poses: np.ndarray,
        distance: float,
    ) -> np.ndarray:
        assignments = self._find_nearest_assignments(source_poses, target_poses)
        shifted: List[np.ndarray] = []
        for target_idx, source_idx in enumerate(assignments):
            src = source_poses[source_idx]
            tgt = target_poses[target_idx]
            if self._pose_distance(src, tgt) <= distance:
                shifted.append(tgt.astype(np.float32))
                continue

            t_src, t_tgt = src[:3, 3], tgt[:3, 3]
            translation_direction = t_tgt - t_src
            translation_norm = float(np.linalg.norm(translation_direction))
            if translation_norm > 1e-6:
                translation_step = (translation_direction / translation_norm) * distance
                new_translation = t_src + translation_step
            else:
                new_translation = t_tgt.copy()

            if (
                np.dot(new_translation - t_src, t_tgt - t_src) <= 0.0
                or np.linalg.norm(new_translation - t_tgt) <= distance
            ):
                new_translation = t_tgt

            if translation_norm > 1e-6:
                interp_t = min(distance / translation_norm, 1.0)
                rot_interp = self._interpolate_rotation(src[:3, :3], tgt[:3, :3], interp_t)
            else:
                rot_interp = tgt[:3, :3].astype(np.float32)

            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = rot_interp
            pose[:3, 3] = new_translation.astype(np.float32)
            shifted.append(pose)
        return np.stack(shifted, axis=0).astype(np.float32)

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        ks: Tensor,
        width: int,
        height: int,
        sh_degree: Optional[int],
        override_color: Optional[Tensor] = None,
        render_mode: str = "RGB",
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]
        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"])
        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1) if override_color is None else override_color
        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=ks,
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=self.cfg.absgrad,
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            sh_degree=sh_degree,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            render_mode=render_mode,
        )
        return render_colors, render_alphas, info

    def _zero_splat_grads(self) -> None:
        for param in self.splats.values():
            param.grad = None

    def render_with_certainty(
        self,
        camtoworlds: Tensor,
        ks: Tensor,
        width: int,
        height: int,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        with torch.enable_grad():
            self._zero_splat_grads()
            rgbs, alphas, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                ks=ks,
                width=width,
                height=height,
                sh_degree=self.cfg.sh_degree,
                render_mode="RGB+ED",
            )
            colors = torch.clamp(rgbs[..., :3], 0.0, 1.0)
            alpha = alphas[..., 0]
            colors.backward(gradient=torch.ones_like(colors))

            hessian_terms: List[Tensor] = []
            for attr in self.freefix_hessian_attrs:
                if attr not in self.splats:
                    continue
                grad = self.splats[attr].grad
                if grad is None:
                    continue
                hessian_terms.append(grad.detach() ** 2)

            if not hessian_terms:
                fallback = torch.clamp(alpha.detach()[0], 0.0, 1.0)
                self._zero_splat_grads()
                return colors.detach()[0], alpha.detach()[0], [fallback]

            h_per_gaussian = torch.cat(hessian_terms, dim=-1)
            self._zero_splat_grads()

            certainty_masks: List[Tensor] = []
            alpha_2d = alpha.detach()[0]
            for scale in self.freefix_certainty_scales:
                inv_h = torch.exp(-scale * h_per_gaussian)
                certainty_rgb, _, _ = self.rasterize_splats(
                    camtoworlds=camtoworlds,
                    ks=ks,
                    width=width,
                    height=height,
                    sh_degree=None,
                    override_color=inv_h,
                )
                certainty = certainty_rgb[0].detach().mean(dim=-1)
                certainty = (alpha_2d * certainty).clamp(0.0, 1.0)
                certainty = _soft_sigmoid(
                    certainty - self.cfg.freefix_mask_center,
                    self.cfg.freefix_mask_softness,
                )
                certainty_masks.append(certainty)

            return colors.detach()[0], alpha_2d, certainty_masks

    def save_checkpoint(self, step: int) -> Path:
        out = self.ckpt_dir / f"ckpt_{step}_rank0.pt"
        torch.save({"step": step, "splats": self.splats.state_dict()}, out)
        return out

    def load_checkpoint(self, ckpt_path: str) -> int:
        ckpt = torch.load(ckpt_path, map_location=self.device)
        for k in self.splats.keys():
            self.splats[k].data = ckpt["splats"][k].to(self.device)
        return int(ckpt.get("step", -1))

    @torch.no_grad()
    def run_fixer_update(self, step: int) -> None:
        fixer = self._ensure_fixer()
        if fixer is None:
            return
        if len(self.testset) == 0:
            return

        cfg = self.cfg
        if self.training_recipe == "difix3d":
            n_views = len(self.testset) if cfg.difix_num_views <= 0 else min(cfg.difix_num_views, len(self.testset))
            fix_tag = "novel"
        elif self.training_recipe == "freefix":
            n_views = len(self.testset) if cfg.freefix_num_views <= 0 else min(cfg.freefix_num_views, len(self.testset))
            fix_tag = f"freefix_{cfg.freefix_backend}_fix"
        else:
            return
        if n_views <= 0:
            return

        out_dir = self.render_dir / fix_tag / str(step)
        pred_dir = out_dir / "Pred"
        ref_dir = out_dir / "Ref"
        fixed_dir = out_dir / "Fixed"
        pred_dir.mkdir(parents=True, exist_ok=True)
        ref_dir.mkdir(parents=True, exist_ok=True)
        fixed_dir.mkdir(parents=True, exist_ok=True)

        render_cam_id = int(self.parser.camera_ids[0])
        render_width, render_height = self.parser.imsize_dict[render_cam_id]
        render_k = torch.from_numpy(self.parser.ks_dict[render_cam_id]).float().to(self.device)

        test_indices_all = np.array(self.testset.indices, dtype=np.int64)
        target_poses_all = self.parser.camtoworlds[test_indices_all].astype(np.float32)
        if self.training_recipe == "difix3d":
            if len(self.fix_steps) == 1 or not self.cfg.difix_progressive_updates:
                render_poses_all = target_poses_all.copy()
            else:
                render_poses_all = self._shift_poses_toward_targets(
                    source_poses=self.current_novel_poses,
                    target_poses=target_poses_all,
                    distance=float(self.cfg.difix_progressive_pose_step),
                )
        else:
            render_poses_all = target_poses_all.copy()

        # Uniformly sample test poses for pseudo supervision updates.
        if len(self.testset) <= n_views:
            local_indices = np.arange(len(self.testset), dtype=np.int64)
        else:
            local_indices = np.linspace(0, len(self.testset) - 1, n_views, dtype=np.int64)
        novel_indices = np.array([int(self.testset.indices[i]) for i in local_indices], dtype=np.int64)
        target_poses = target_poses_all[local_indices]
        render_poses = render_poses_all[local_indices]

        fixed_paths: List[str] = []
        fixed_c2ws: List[np.ndarray] = []
        fixed_camera_ids: List[int] = []
        fixed_alpha_mask_paths: List[str] = []
        pose_metadata: List[Dict[str, Any]] = []
        mask_dir = out_dir / "Mask"
        alpha_dir = out_dir / "Alpha"
        if self.training_recipe == "freefix":
            mask_dir.mkdir(parents=True, exist_ok=True)
            alpha_dir.mkdir(parents=True, exist_ok=True)
        elif self.training_recipe == "difix3d":
            alpha_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{self.training_recipe}] step={step} creating {len(novel_indices)} fixed novel views")
        for i, novel_idx in enumerate(novel_indices):
            target_c2w = target_poses[i].astype(np.float32)
            render_c2w_np = render_poses[i].astype(np.float32)
            c2w = torch.from_numpy(render_c2w_np[None, ...]).float().to(self.device)
            k_mat = render_k.unsqueeze(0).to(self.device)

            if self.training_recipe == "freefix":
                colors, alphas, certainty_masks = self.render_with_certainty(
                    camtoworlds=c2w,
                    ks=k_mat,
                    width=render_width,
                    height=render_height,
                )
                pred_u8 = (colors.cpu().numpy() * 255.0).astype(np.uint8)
                alpha_u8 = (torch.clamp(alphas, 0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
                mask_pils: List[Image.Image] = []
                for mask_idx, certainty_mask in enumerate(certainty_masks):
                    mask_u8 = (torch.clamp(certainty_mask, 0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
                    Image.fromarray(mask_u8, mode="L").save(mask_dir / f"{i:04d}_{mask_idx}.png")
                    mask_pils.append(Image.fromarray(mask_u8, mode="L"))

                avg_mask = torch.stack(certainty_masks, dim=0).mean(dim=0)
                avg_mask_u8 = (torch.clamp(avg_mask, 0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
                Image.fromarray(avg_mask_u8, mode="L").save(mask_dir / f"{i:04d}.png")
                alpha_path = alpha_dir / f"{i:04d}.png"
                Image.fromarray(alpha_u8, mode="L").save(alpha_path)
                fixed_alpha_mask_paths.append(str(alpha_path))
                mask_pil = mask_pils
                alpha_pil = Image.fromarray(alpha_u8, mode="L")
            else:
                colors, alphas, _ = self.rasterize_splats(
                    camtoworlds=c2w,
                    ks=k_mat,
                    width=render_width,
                    height=render_height,
                    sh_degree=self.cfg.sh_degree,
                    render_mode="RGB+ED",
                )
                # RGB+ED includes depth in channel 4; only RGB should be passed to DIFIX.
                pred_u8 = (torch.clamp(colors[0, ..., :3], 0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
                alpha_u8 = (torch.clamp(alphas[..., 0], 0.0, 1.0).squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)
                alpha_path = alpha_dir / f"{i:04d}.png"
                Image.fromarray(alpha_u8, mode="L").save(alpha_path)
                fixed_alpha_mask_paths.append(str(alpha_path))
                alpha_pil = Image.fromarray(alpha_u8, mode="L")
                mask_pil = None
            pred_pil = Image.fromarray(pred_u8)
            pred_path = pred_dir / f"{i:04d}.png"
            pred_pil.save(pred_path)

            seed = cfg.seed + step * 1000 + i
            if self.training_recipe == "difix3d":
                ref_pil = self._load_train_reference_image(
                    c2w=render_c2w_np,
                    width=render_width,
                    height=render_height,
                )
                ref_pil.save(ref_dir / f"{i:04d}.png")
                fixed_pil = self._run_difix_fixer(
                    pred_pil=pred_pil,
                    ref_pil=ref_pil,
                    seed=seed,
                    post_render=False,
                )
                pose_metadata.append(
                    {
                        "target_index": int(novel_idx),
                        "camera_id": render_cam_id,
                        "translation_error_to_target": float(
                            np.linalg.norm(render_c2w_np[:3, 3] - target_c2w[:3, 3])
                        ),
                    }
                )
            elif self.training_recipe == "freefix":
                fixed_pil = fixer(
                    prompt=cfg.freefix_prompt,
                    negative_prompt=cfg.freefix_negative_prompt,
                    image=pred_pil,
                    mask=mask_pil,
                    mask_scheduler=self._build_freefix_mask_scheduler(mask_count=len(mask_pil)),
                    guide_until=self._build_freefix_guide_until(),
                    warp_image=pred_pil,
                    warp_until=self._build_freefix_warp_until(),
                    warp_mask=alpha_pil,
                    num_inference_steps=cfg.freefix_num_inference_steps,
                    strength=cfg.freefix_strength,
                    guidance_scale=cfg.freefix_guidance_scale,
                    seed=seed,
                )
            else:
                continue
            fixed_path = fixed_dir / f"{i:04d}.png"
            fixed_pil.save(fixed_path)

            fixed_paths.append(str(fixed_path))
            fixed_c2ws.append(render_c2w_np.astype(np.float32))
            fixed_camera_ids.append(render_cam_id)

        if len(fixed_paths) == 0:
            return

        self.novel_image_paths = list(fixed_paths)
        self.novel_c2ws = [pose.copy() for pose in fixed_c2ws]
        self.novel_camera_ids = list(fixed_camera_ids)
        if self.training_recipe == "difix3d":
            self.current_novel_poses = render_poses_all.copy()

        if pose_metadata:
            with open(out_dir / "pose_progress.json", "w", encoding="utf-8") as f:
                json.dump(pose_metadata, f, indent=2)

        novel_parser = SimpleNamespace(
            image_names=[Path(p).name for p in self.novel_image_paths],
            image_paths=list(self.novel_image_paths),
            camtoworlds=np.stack(self.novel_c2ws, axis=0).astype(np.float32),
            camera_ids=list(self.novel_camera_ids),
            ks_dict=self.parser.ks_dict,
            Ks_dict=getattr(self.parser, "Ks_dict", self.parser.ks_dict),
            test_every=0,
            imsize_dict=self.parser.imsize_dict,
            mapx_dict=getattr(self.parser, "mapx_dict", {}),
            mapy_dict=getattr(self.parser, "mapy_dict", {}),
            roi_undist_dict=getattr(self.parser, "roi_undist_dict", {}),
            mask_dict=getattr(self.parser, "mask_dict", {}),
            point_indices={},
            points=np.empty((0, 3), dtype=np.float32),
            points_rgb=np.empty((0, 3), dtype=np.uint8),
            points_err=np.empty((0,), dtype=np.float32),
            alpha_mask_paths=fixed_alpha_mask_paths or None,
        )

        novel_dataset = ColmapImageDataset(
            novel_parser,
            split="train",
            patch_size=None,
            train_split_all=True,
        )
        self.novel_loader = DataLoader(
            novel_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=cfg.num_workers > 0,
        )
        self.novel_loader_iter = iter(self.novel_loader)
        print(f"[{self.training_recipe}] rebuilt novel loader with {len(novel_dataset)} fixed images")

    def train(self, start_step: int = 0) -> None:
        cfg = self.cfg
        with open(self.result_dir / "cfg.json", "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2)

        train_loader = DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=cfg.num_workers > 0,
        )
        train_iter = iter(train_loader)

        lr_sched = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizers["means"],
            gamma=0.01 ** (1.0 / max(cfg.max_steps, 1)),
        )

        t0 = time.time()
        for step in range(start_step, cfg.max_steps):
            is_novel_data = False
            use_novel_data = (
                self.fixer is not None
                and self.novel_loader_iter is not None
                and random.random() < self._get_novel_sampling_prob()
            )

            if use_novel_data:
                try:
                    data = next(self.novel_loader_iter)
                except StopIteration:
                    assert self.novel_loader is not None
                    self.novel_loader_iter = iter(self.novel_loader)
                    data = next(self.novel_loader_iter)
                is_novel_data = True
            else:
                try:
                    data = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    data = next(train_iter)

            c2w = data["camtoworld"].to(self.device)
            ks = data["K"].to(self.device)
            pixels = data["image"].to(self.device) / 255.0
            h, w = pixels.shape[1:3]

            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            renders, alphas, info = self.rasterize_splats(
                camtoworlds=c2w,
                ks=ks,
                width=w,
                height=h,
                sh_degree=sh_degree_to_use,
            )
            colors = torch.clamp(renders, 0.0, 1.0)
            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=self.device)
                colors = colors + bkgd * (1.0 - alphas)
            alpha_masks = data["alpha_mask"].to(self.device) if "alpha_mask" in data else None  # [1, H, W, 1]
            if is_novel_data and alpha_masks is not None:
                colors = colors * (alpha_masks > 0.5).float()
                pixels = pixels * (alpha_masks > 0.5).float()

            self.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            l1 = F.l1_loss(colors, pixels)
            ssim_loss = 1.0 - _simple_ssim(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2))
            loss = l1 * (1.0 - cfg.ssim_lambda) + ssim_loss * cfg.ssim_lambda
            if is_novel_data:
                loss = loss * self._get_novel_loss_scale()
            else:
                loss = loss * self._get_real_loss_scale()
            loss.backward()

            if cfg.sparse_grad:
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],
                        values=grad[gaussian_ids],
                        size=self.splats[k].size(),
                        is_coalesced=len(ks) == 1,
                    )

            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            lr_sched.step()

            # Keep strategy update order consistent with gsplat training flow:
            # run post-backward after optimizer/scheduler steps.
            self.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                packed=cfg.packed,
            )

            if (step + 1) % 50 == 0 or step == 0:
                psnr = _compute_psnr(colors, pixels).item()
                elapsed = time.time() - t0
                print(
                    f"[step {step+1:06d}/{cfg.max_steps}] "
                    f"loss={loss.item():.4f} l1={l1.item():.4f} ssim={ssim_loss.item():.4f} "
                    f"psnr={psnr:.2f} n_gs={len(self.splats['means'])} "
                    f"novel={int(is_novel_data)} t={elapsed:.1f}s"
                )

            should_save = (
                bool(self.save_steps)
                and (self._matches_schedule(step, self.save_steps) or step == cfg.max_steps - 1)
            ) or (
                not self.save_steps
                and ((step + 1) % cfg.save_every == 0 or step + 1 == cfg.max_steps)
            )
            if should_save:
                ckpt = self.save_checkpoint(step)
                with open(self.stats_dir / f"train_step{step:06d}.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "step": step,
                            "loss": float(loss.item()),
                            "num_gs": int(len(self.splats["means"])),
                            "elapsed_sec": float(time.time() - t0),
                            "checkpoint": str(ckpt),
                        },
                        f,
                        indent=2,
                    )

            should_eval = (
                bool(self.eval_steps)
                and (self._matches_schedule(step, self.eval_steps) or step == cfg.max_steps - 1)
            ) or (
                not self.eval_steps
                and ((step + 1) % cfg.eval_every == 0 or step + 1 == cfg.max_steps)
            )
            if should_eval:
                self.eval(step)

            if self._should_run_fix(step):
                self.run_fixer_update(step)

    @torch.no_grad()
    def eval(self, step: int) -> Dict[str, float]:
        loader = DataLoader(self.testset, batch_size=1, shuffle=False, num_workers=1)
        out_dir = self.render_dir / f"eval_{step:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        psnrs: List[float] = []
        psnrs_raw: List[float] = []
        use_post_render = self._use_difix_post_render()
        t0 = time.time()
        for i, data in enumerate(loader):
            c2w = data["camtoworld"].to(self.device)
            ks = data["K"].to(self.device)
            gt = data["image"].to(self.device) / 255.0
            h, w = gt.shape[1:3]

            colors, _, _ = self.rasterize_splats(
                camtoworlds=c2w,
                ks=ks,
                width=w,
                height=h,
                sh_degree=self.cfg.sh_degree,
            )
            colors = torch.clamp(colors, 0.0, 1.0)
            raw_psnr = _compute_psnr(colors, gt).item()
            if use_post_render:
                pred_raw = (colors[0].cpu().numpy() * 255).astype(np.uint8)
                c2w_np = c2w[0].detach().cpu().numpy().astype(np.float32)
                pred = self._apply_difix_post_render(
                    pred_u8=pred_raw,
                    c2w=c2w_np,
                    step=step,
                    frame_index=i,
                )
                colors_eval = torch.from_numpy(pred).float().to(self.device).unsqueeze(0) / 255.0
                psnr = _compute_psnr(colors_eval, gt).item()
                psnrs_raw.append(raw_psnr)
            else:
                pred = (colors[0].cpu().numpy() * 255).astype(np.uint8)
                psnr = raw_psnr
            psnrs.append(psnr)

            gt_u8 = (gt[0].cpu().numpy() * 255).astype(np.uint8)
            if use_post_render:
                imageio.imwrite(out_dir / f"{i:04d}_pred_raw.png", pred_raw)
            imageio.imwrite(out_dir / f"{i:04d}_pred.png", pred)
            imageio.imwrite(out_dir / f"{i:04d}_gt.png", gt_u8)

        stats = {
            "step": float(step),
            "psnr": float(np.mean(psnrs) if psnrs else 0.0),
            "n_images": float(len(psnrs)),
            "sec_per_image": float((time.time() - t0) / max(len(psnrs), 1)),
            "num_gs": float(len(self.splats["means"])),
        }
        if psnrs_raw:
            stats["psnr_raw"] = float(np.mean(psnrs_raw))
        with open(self.stats_dir / f"eval_step{step:06d}.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(
            f"[eval step {step:06d}] psnr={stats['psnr']:.3f}"
            + (f" psnr_raw={stats['psnr_raw']:.3f}" if "psnr_raw" in stats else "")
            + " "
            f"sec/img={stats['sec_per_image']:.3f} n_gs={int(stats['num_gs'])}"
        )
        return {k: float(v) for k, v in stats.items()}

    @torch.no_grad()
    def render_interpolated_video(self, step: int, n_frames: int, fps: int) -> Path:
        if len(self.testset) == 0:
            raise RuntimeError("Test split is empty; cannot render trajectory.")

        cam0 = self.parser.camtoworlds[self.testset.indices[0]]
        cam1 = self.parser.camtoworlds[self.testset.indices[-1]]
        cam_interp = []
        for t in np.linspace(0.0, 1.0, n_frames):
            c2w = cam0 * (1.0 - t) + cam1 * t
            c2w[3, :] = np.array([0, 0, 0, 1], dtype=np.float32)
            cam_interp.append(c2w)
        cam_interp_np = np.stack(cam_interp, axis=0).astype(np.float32)
        cam_interp = torch.from_numpy(cam_interp_np).float().to(self.device)

        cam_id = self.parser.camera_ids[self.testset.indices[0]]
        k_mat = torch.from_numpy(self.parser.ks_dict[cam_id]).float().to(self.device)
        width, height = self.parser.imsize_dict[cam_id]
        ks = k_mat.unsqueeze(0).repeat(cam_interp.shape[0], 1, 1)

        out_mp4 = self.render_dir / f"traj_step{step:06d}.mp4"
        writer = imageio.get_writer(str(out_mp4), fps=fps)
        use_post_render = self._use_difix_post_render()
        for i in range(0, cam_interp.shape[0], 8):
            c2w = cam_interp[i : i + 8]
            k_batch = ks[i : i + 8]
            colors, _, _ = self.rasterize_splats(
                camtoworlds=c2w,
                ks=k_batch,
                width=width,
                height=height,
                sh_degree=self.cfg.sh_degree,
            )
            colors = torch.clamp(colors, 0.0, 1.0).cpu().numpy()
            for j in range(colors.shape[0]):
                frame = (colors[j] * 255).astype(np.uint8)
                if use_post_render:
                    frame = self._apply_difix_post_render(
                        pred_u8=frame,
                        c2w=cam_interp_np[i + j],
                        step=step,
                        frame_index=i + j,
                    )
                writer.append_data(frame)
        writer.close()
        print(f"[render] wrote {out_mp4}")
        return out_mp4


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Standalone 3DGS Trainer with optional FreeFix and Difix3D+ style iterative fixing."
    )
    p.add_argument("--config", type=str, default=None, help="Optional JSON config file.")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--result_dir", type=str, default=None)
    p.add_argument("--data_factor", type=int, default=4)
    p.add_argument("--test_every", type=int, default=8)
    p.add_argument("--normalize_world", action="store_true")
    p.add_argument("--normalize_align_axes", action="store_true")
    p.add_argument("--partition_file", type=str, default=None)
    p.add_argument("--patch_size", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=30000)
    p.add_argument("--eval_every", type=int, default=10000)
    p.add_argument("--save_every", type=int, default=10000)
    p.add_argument("--eval_steps", type=str, default=None, help="Comma-separated eval steps.")
    p.add_argument("--save_steps", type=str, default=None, help="Comma-separated checkpoint steps.")
    p.add_argument("--train_split_all", action="store_true")
    p.add_argument("--sh_degree", type=int, default=3)
    p.add_argument("--sh_degree_interval", type=int, default=1000)
    p.add_argument("--ssim_lambda", type=float, default=0.2)
    p.add_argument("--random_bkgd", action="store_true")
    p.add_argument("--near_plane", type=float, default=0.01)
    p.add_argument("--far_plane", type=float, default=1e10)
    p.add_argument("--packed", action="store_true")
    p.add_argument("--sparse_grad", action="store_true")
    p.add_argument("--absgrad", action="store_true")
    p.add_argument("--antialiased", action="store_true")
    p.add_argument("--init_type", type=str, default="sfm", choices=["sfm", "random"])
    p.add_argument("--init_num_pts", type=int, default=100000)
    p.add_argument("--init_extent", type=float, default=3.0)
    p.add_argument("--init_opa", type=float, default=0.1)
    p.add_argument("--init_scale", type=float, default=1.0)
    p.add_argument("--global_scale", type=float, default=1.0)
    p.add_argument("--means_lr", type=float, default=1.6e-4 / 10)
    p.add_argument("--scales_lr", type=float, default=5e-3 / 5)
    p.add_argument("--quats_lr", type=float, default=1e-3 / 5)
    p.add_argument("--opacities_lr", type=float, default=5e-2 / 5)
    p.add_argument("--sh0_lr", type=float, default=2.5e-3 / 50)
    p.add_argument("--shN_lr", type=float, default=2.5e-3 / 20 / 50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--mode", type=str, default="train", choices=["train", "eval", "render", "train_eval"])
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--render_frames", type=int, default=120)
    p.add_argument("--render_fps", type=int, default=24)
    p.add_argument("--training_recipe", type=str, default="vanilla", choices=["vanilla", "freefix", "difix3d"])
    p.add_argument("--use_freefix", action="store_true", help="Compatibility alias to enable FreeFix recipe.")
    p.add_argument("--fix_steps", type=str, default=None, help="Comma-separated fix steps, e.g. 3000,6000,9000")
    p.add_argument("--fix_cache_dir", type=str, default=None, help="Shared cache dir for all fixer backends")
    p.add_argument("--lazy_fixer_init", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_difix", action="store_true")
    p.add_argument("--difix_model_id", type=str, default="nvidia/difix_ref")
    p.add_argument("--difix_model_path", type=str, default=None)
    p.add_argument("--difix_cache_dir", type=str, default=None)
    p.add_argument("--difix_prompt", type=str, default="remove degradation")
    p.add_argument("--difix_start_step", type=int, default=3000)
    p.add_argument("--difix_fix_every", type=int, default=3000)
    p.add_argument("--difix_num_views", type=int, default=0)
    p.add_argument("--difix_num_inference_steps", type=int, default=1)
    p.add_argument("--difix_timestep", type=int, default=199)
    p.add_argument("--difix_guidance_scale", type=float, default=0.0)
    p.add_argument("--difix_novel_prob", type=float, default=0.3)
    p.add_argument("--difix_novel_lambda", type=float, default=0.3)
    p.add_argument(
        "--difix_progressive_updates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Progressively move pseudo-view poses toward targets during fixer updates.",
    )
    p.add_argument("--difix_progressive_pose_step", type=float, default=0.5)
    p.add_argument(
        "--difix_post_render",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply DIFIX as a post-render neural enhancer during eval/render.",
    )
    p.add_argument("--difix_post_prompt", type=str, default=None)
    p.add_argument("--difix_post_num_inference_steps", type=int, default=1)
    p.add_argument("--difix_post_timestep", type=int, default=199)
    p.add_argument("--difix_post_guidance_scale", type=float, default=0.0)
    p.add_argument("--freefix_backend", type=str, default="flux", choices=["flux", "sdxl"])
    p.add_argument(
        "--freefix_prompt",
        type=str,
        default="A photorealistic real-world scene with consistent geometry, detailed textures, and natural lighting.",
    )
    p.add_argument(
        "--freefix_negative_prompt",
        type=str,
        default="blurry, low quality, foggy, overall gray, subtitles, incomplete, ghost image, too close to camera",
    )
    p.add_argument("--freefix_start_step", type=int, default=30000)
    p.add_argument("--freefix_fix_every", type=int, default=400)
    p.add_argument("--freefix_num_views", type=int, default=0)
    p.add_argument("--freefix_num_inference_steps", type=int, default=50)
    p.add_argument("--freefix_strength", type=float, default=0.6)
    p.add_argument("--freefix_guidance_scale", type=float, default=3.5)
    p.add_argument("--freefix_novel_prob", type=float, default=0.1)
    p.add_argument("--freefix_novel_lambda", type=float, default=0.2)
    p.add_argument("--freefix_real_lambda", type=float, default=1.0)
    p.add_argument("--freefix_certainty_scales", type=str, default="0.001,0.01,0.1")
    p.add_argument("--freefix_hessian_attrs", type=str, default="means,quats,scales")
    p.add_argument("--freefix_mask_center", type=float, default=0.5)
    p.add_argument("--freefix_mask_softness", type=float, default=10.0)
    p.add_argument("--freefix_mask_scheduler", type=str, default="0.3,0.9,1.0")
    p.add_argument("--freefix_guide_ratio", type=float, default=1.0)
    p.add_argument("--freefix_warp_ratio", type=float, default=0.5)
    args_pre, _ = p.parse_known_args()
    if args_pre.config is not None:
        with open(args_pre.config, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
        if not isinstance(cfg_data, dict):
            raise ValueError("Config JSON must be an object.")
        valid_fields = {f.name for f in fields(Config)}
        unknown_fields = sorted(str(k) for k in cfg_data.keys() if str(k) not in valid_fields)
        if unknown_fields:
            raise ValueError(f"Unknown config keys: {', '.join(unknown_fields)}")
        p.set_defaults(**cfg_data)

    args = p.parse_args()
    args_dict = vars(args)
    args_dict.pop("config", None)

    cfg = Config(**args_dict)
    missing: List[str] = []
    if cfg.data_dir is None or len(str(cfg.data_dir).strip()) == 0:
        missing.append("data_dir")
    if cfg.result_dir is None or len(str(cfg.result_dir).strip()) == 0:
        missing.append("result_dir")
    if missing:
        raise ValueError(
            "Missing required arguments: "
            + ", ".join(missing)
            + ". Provide them with CLI flags or in --config JSON."
        )
    return cfg


def main() -> None:
    cfg = parse_args()
    trainer = Trainer(cfg)

    step = 0
    if cfg.ckpt is not None:
        step = trainer.load_checkpoint(cfg.ckpt)
        print(f"[load] checkpoint={cfg.ckpt} step={step}")

    if cfg.mode == "train":
        trainer.train(start_step=max(step, 0))
    elif cfg.mode == "eval":
        trainer.eval(step=max(step, 0))
    elif cfg.mode == "render":
        trainer.render_interpolated_video(step=max(step, 0), n_frames=cfg.render_frames, fps=cfg.render_fps)
    elif cfg.mode == "train_eval":
        trainer.train(start_step=max(step, 0))
        last_ckpt = trainer.ckpt_dir / f"ckpt_{cfg.max_steps - 1}_rank0.pt"
        if last_ckpt.exists():
            trainer.load_checkpoint(str(last_ckpt))
        trainer.eval(step=cfg.max_steps - 1)
        trainer.render_interpolated_video(
            step=cfg.max_steps - 1,
            n_frames=cfg.render_frames,
            fps=cfg.render_fps,
        )
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()
