from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type

from rich.progress import Console

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.data.utils.dataloaders import CacheDataloader
from nerfstudio.model_components.ray_generators import RayGenerator

CONSOLE = Console(width=120)


@dataclass
class Difix3DDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: Difix3DDataManager)
    patch_size: int = 32
    cache_num_workers: int = 0


class Difix3DDataManager(VanillaDataManager):
    config: Difix3DDataManagerConfig

    def setup_train(self) -> None:
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        cache_num_workers = max(0, int(self.config.cache_num_workers))
        use_pin_memory = str(self.device).startswith("cuda")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=cache_num_workers,
            pin_memory=use_pin_memory,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_ray_generator = RayGenerator(self.train_dataset.cameras.to(self.device))

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        del step
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch
