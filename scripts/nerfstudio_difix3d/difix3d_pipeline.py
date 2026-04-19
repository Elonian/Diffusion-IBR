from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type

import numpy as np
import torch
import tqdm
from PIL import Image
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig

from scripts.priors.src.pipeline_difix import DifixPipeline
from utils.diffusion_utils import resolve_hf_cache_root
from utils.pose_utils import CameraPoseInterpolator

from .difix3d_datamanager import Difix3DDataManagerConfig

HF_HUB_CACHE = resolve_hf_cache_root()


@dataclass
class Difix3DPipelineConfig(VanillaPipelineConfig):
    _target: Type = field(default_factory=lambda: Difix3DPipeline)
    datamanager: Difix3DDataManagerConfig = field(default_factory=Difix3DDataManagerConfig)
    steps_per_fix: int = 2000
    steps_per_val: int = 5000


class Difix3DPipeline(VanillaPipeline):
    config: Difix3DPipelineConfig

    def __init__(
        self,
        config: Difix3DPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
        render_dir: str = "renders",
    ) -> None:
        del grad_scaler
        super().__init__(config, device, test_mode, world_size, local_rank)

        self.render_dir = Path(render_dir)
        self.difix = None
        self.difix_available = True

        train_c2w = self.datamanager.train_dataparser_outputs.cameras.camera_to_worlds
        train_pad = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=train_c2w.dtype, device=train_c2w.device).reshape(1, 1, 4)
        self.training_poses = torch.cat([train_c2w, train_pad.repeat(train_c2w.shape[0], 1, 1)], dim=1)

        test_outputs = self.datamanager.dataparser.get_dataparser_outputs(split=self.datamanager.test_split)
        test_c2w = test_outputs.cameras.camera_to_worlds
        test_pad = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=test_c2w.dtype, device=test_c2w.device).reshape(1, 1, 4)
        self.testing_poses = torch.cat([test_c2w, test_pad.repeat(test_c2w.shape[0], 1, 1)], dim=1)

        self.current_novel_poses = self.training_poses
        self.current_novel_cameras = self.datamanager.train_dataparser_outputs.cameras
        self.interpolator = CameraPoseInterpolator(rotation_weight=1.0, translation_weight=1.0)
        self.novel_datamanagers = []

    def _load_difix(self):
        if self.difix is not None:
            return self.difix
        if not self.difix_available:
            return None

        try:
            difix = DifixPipeline.from_pretrained(
                "nvidia/difix_ref",
                trust_remote_code=True,
                cache_dir=HF_HUB_CACHE,
            )
            difix.set_progress_bar_config(disable=True)
            difix.to(self.device)
            self.difix = difix
            return difix
        except Exception as exc:
            self.difix_available = False
            print(f"[difix3d] Disabling fixer because DifixPipeline failed to load: {type(exc).__name__}: {exc}")
            return None

    def get_train_loss_dict(self, step: int):
        if len(self.novel_datamanagers) == 0 or random.random() < 0.6:
            ray_bundle, batch = self.datamanager.next_train(step)
        else:
            ray_bundle, batch = self.novel_datamanagers[-1].next_train(step)

        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        if self.config.steps_per_fix > 0 and step > 0 and step % self.config.steps_per_fix == 0:
            self.fix(step)
        if self.config.steps_per_val > 0 and step > 0 and step % self.config.steps_per_val == 0:
            self.val(step)

        return model_outputs, loss_dict, metrics_dict

    def forward(self):
        raise NotImplementedError

    @torch.no_grad()
    def render_traj(self, step: int, cameras, tag: str = "novel") -> None:
        was_training = self.training
        self.eval()
        try:
            for i in tqdm.trange(0, len(cameras), desc="Rendering trajectory"):
                outputs = self.model.get_outputs_for_camera(cameras[i])
                rgb_path = self.render_dir / tag / str(step) / "Pred" / f"{i:04d}.png"
                rgb_path.parent.mkdir(parents=True, exist_ok=True)
                rgb_canvas = outputs["rgb"].detach().cpu().numpy()
                rgb_canvas = np.clip(rgb_canvas, 0.0, 1.0)
                Image.fromarray((rgb_canvas * 255.0).astype(np.uint8)).save(rgb_path)
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def val(self, step: int) -> None:
        was_training = self.training
        self.eval()
        try:
            cameras = self.datamanager.dataparser.get_dataparser_outputs(split=self.datamanager.test_split).cameras
            for i in tqdm.trange(0, len(cameras), desc="Running evaluation"):
                outputs = self.model.get_outputs_for_camera(cameras[i])
                rgb_path = self.render_dir / "val" / str(step) / f"{i:04d}.png"
                rgb_path.parent.mkdir(parents=True, exist_ok=True)
                rgb_canvas = outputs["rgb"].detach().cpu().numpy()
                rgb_canvas = np.clip(rgb_canvas, 0.0, 1.0)
                Image.fromarray((rgb_canvas * 255.0).astype(np.uint8)).save(rgb_path)
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def fix(self, step: int) -> None:
        difix = self._load_difix()
        if difix is None:
            return

        novel_poses_np = self.interpolator.shift_poses(
            self.current_novel_poses.detach().cpu().numpy(),
            self.testing_poses.detach().cpu().numpy(),
            distance=0.5,
        )
        novel_poses = torch.from_numpy(novel_poses_np).to(self.testing_poses.device, dtype=self.testing_poses.dtype)

        ref_indices = self.interpolator.find_nearest_assignments(
            self.training_poses.detach().cpu().numpy(),
            novel_poses.detach().cpu().numpy(),
        )
        ref_image_filenames = np.array(self.datamanager.train_dataparser_outputs.image_filenames)[ref_indices].tolist()

        cameras = self.datamanager.train_dataparser_outputs.cameras
        cameras = Cameras(
            fx=cameras.fx[0].repeat(len(novel_poses), 1),
            fy=cameras.fy[0].repeat(len(novel_poses), 1),
            cx=cameras.cx[0].repeat(len(novel_poses), 1),
            cy=cameras.cy[0].repeat(len(novel_poses), 1),
            distortion_params=cameras.distortion_params[0].repeat(len(novel_poses), 1),
            height=cameras.height[0].repeat(len(novel_poses), 1),
            width=cameras.width[0].repeat(len(novel_poses), 1),
            camera_to_worlds=novel_poses[:, :3, :4],
            camera_type=cameras.camera_type[0].repeat(len(novel_poses), 1),
            metadata=cameras.metadata,
        )

        self.render_traj(step, cameras)

        image_filenames = []
        fixed_dir = self.render_dir / "novel" / str(step) / "Fixed"
        ref_dir = self.render_dir / "novel" / str(step) / "Ref"
        fixed_dir.mkdir(parents=True, exist_ok=True)
        ref_dir.mkdir(parents=True, exist_ok=True)
        for i in tqdm.trange(0, len(novel_poses), desc="Fixing artifacts..."):
            image = Image.open(self.render_dir / "novel" / str(step) / "Pred" / f"{i:04d}.png").convert("RGB")
            ref_image = Image.open(ref_image_filenames[i]).convert("RGB")
            output_image = difix(
                prompt="remove degradation",
                image=image,
                ref_image=ref_image,
                num_inference_steps=1,
                timesteps=[199],
                guidance_scale=0.0,
            ).images[0]
            output_image = output_image.resize(image.size, Image.LANCZOS)
            fixed_path = fixed_dir / f"{i:04d}.png"
            output_image.save(fixed_path)
            ref_image.save(ref_dir / f"{i:04d}.png")
            image_filenames.append(fixed_path)

        train_outputs = self.datamanager.train_dataparser_outputs
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=train_outputs.scene_box,
            mask_filenames=None,
            dataparser_scale=train_outputs.dataparser_scale,
            dataparser_transform=train_outputs.dataparser_transform,
            metadata=train_outputs.metadata,
        )

        datamanager_config = Difix3DDataManagerConfig(
            dataparser=self.config.datamanager.dataparser,
            train_num_rays_per_batch=self.config.datamanager.train_num_rays_per_batch,
            eval_num_rays_per_batch=self.config.datamanager.eval_num_rays_per_batch,
            patch_size=self.config.datamanager.patch_size,
            cache_num_workers=self.config.datamanager.cache_num_workers,
        )
        datamanager = datamanager_config.setup(
            device=self.datamanager.device,
            test_mode=self.datamanager.test_mode,
            world_size=self.datamanager.world_size,
            local_rank=self.datamanager.local_rank,
        )
        datamanager.train_dataparser_outputs = dataparser_outputs
        datamanager.train_dataset = datamanager.create_train_dataset()
        datamanager.setup_train()

        self.novel_datamanagers.append(datamanager)
        self.current_novel_poses = novel_poses
        self.current_novel_cameras = cameras
