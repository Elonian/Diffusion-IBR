#!/usr/bin/env python3
"""Standalone vanilla Nerfacto trainer for DL3DV scenes.

This script is independent from project-specific training code and uses official
Nerfstudio trainer/config classes directly.

Examples:
  python scripts/trainers/nerfacto_vanilla_trainer.py \
      --scene_id 06da796666297fe4c683c231edf56ec00148a6a52ab5bb159fe1be31f53a58df \
      --dl3dv_root /mntdatalora/src/Diffusion-IBR/data/DL3DV-10K-Benchmark \
      --output_dir /mntdatalora/src/Diffusion-IBR/outputs \
      --max_num_iterations 30000

  python scripts/trainers/nerfacto_vanilla_trainer.py \
      --data /mntdatalora/src/Diffusion-IBR/data/DL3DV-10K-Benchmark/<scene_id>/nerfstudio
"""

from __future__ import annotations

import argparse
import importlib
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Nerfacto trainer for DL3DV scenes.")

    parser.add_argument("--data", type=Path, default=None, help="Path to dataset dir/json or DL3DV scene dir.")
    parser.add_argument(
        "--scene_id",
        type=str,
        default=None,
        help="DL3DV scene id. Used with --dl3dv_root when --data is not provided.",
    )
    parser.add_argument(
        "--dl3dv_root",
        type=Path,
        default=Path("/mntdatalora/src/Diffusion-IBR/data/DL3DV-10K-Benchmark"),
        help="Root path containing DL3DV scene folders.",
    )
    parser.add_argument(
        "--scene_source",
        choices=["auto", "nerfstudio", "gaussian_splat"],
        default="auto",
        help="Which scene subfolder to use when a scene root is provided.",
    )

    parser.add_argument("--output_dir", type=Path, default=Path("/mntdatalora/src/Diffusion-IBR/outputs"))
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--method_name", type=str, default="nerfacto")
    parser.add_argument(
        "--timestamp",
        type=str,
        default="{timestamp}",
        help='Timestamp folder name. Use "{timestamp}" for auto time or "" for no timestamp folder.',
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device_type", choices=["cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--num_machines", type=int, default=1)
    parser.add_argument("--machine_rank", type=int, default=0)
    parser.add_argument("--dist_url", type=str, default="auto")

    parser.add_argument("--max_num_iterations", type=int, default=30000)
    parser.add_argument("--steps_per_eval_batch", type=int, default=500)
    parser.add_argument("--steps_per_save", type=int, default=2000)
    parser.add_argument("--disable_mixed_precision", action="store_true")
    parser.add_argument("--save_all_checkpoints", action="store_true")
    parser.add_argument(
        "--vis",
        choices=[
            "viewer",
            "wandb",
            "tensorboard",
            "comet",
            "viewer+wandb",
            "viewer+tensorboard",
            "viewer+comet",
            "viewer_legacy",
        ],
        default="viewer",
    )

    parser.add_argument("--downscale_factor", type=int, default=None, help="If omitted, Nerfstudio auto-selects.")
    parser.add_argument("--eval_mode", choices=["fraction", "filename", "interval", "all"], default="filename")
    parser.add_argument("--train_split_fraction", type=float, default=0.9)
    parser.add_argument("--eval_interval", type=int, default=8)
    parser.add_argument("--orientation_method", choices=["pca", "up", "vertical", "none"], default="up")
    parser.add_argument("--center_method", choices=["poses", "focus", "none"], default="poses")

    parser.add_argument("--train_num_rays_per_batch", type=int, default=4096)
    parser.add_argument("--eval_num_rays_per_batch", type=int, default=4096)
    parser.add_argument("--eval_num_rays_per_chunk", type=int, default=1 << 15)
    parser.add_argument("--average_init_density", type=float, default=0.01)
    parser.add_argument("--camera_optimizer_mode", choices=["off", "SO3xR3", "SE3"], default="SO3xR3")
    parser.add_argument("--implementation", choices=["tcnn", "torch"], default="torch")
    parser.add_argument("--prefer_parallel_datamanager", action="store_true")

    parser.add_argument("--proposal_lr", type=float, default=1e-2)
    parser.add_argument("--fields_lr", type=float, default=1e-2)
    parser.add_argument("--camera_opt_lr", type=float, default=1e-3)
    parser.add_argument("--proposal_lr_final", type=float, default=1e-4)
    parser.add_argument("--fields_lr_final", type=float, default=1e-4)
    parser.add_argument("--camera_opt_lr_final", type=float, default=1e-4)
    parser.add_argument("--proposal_lr_decay_steps", type=int, default=200000)
    parser.add_argument("--fields_lr_decay_steps", type=int, default=200000)
    parser.add_argument("--camera_opt_lr_decay_steps", type=int, default=5000)

    parser.add_argument(
        "--nerfstudio_path",
        type=Path,
        default=None,
        help="Optional directory that contains the `nerfstudio` package (added to PYTHONPATH).",
    )
    parser.add_argument("--dry_run", action="store_true", help="Only resolve config and print paths; no training.")

    return parser.parse_args()


def _candidate_sources(scene_source: str) -> List[str]:
    if scene_source == "nerfstudio":
        return ["nerfstudio"]
    if scene_source == "gaussian_splat":
        return ["gaussian_splat"]
    return ["nerfstudio", "gaussian_splat"]


def _resolve_data_path(
    data: Optional[Path], scene_id: Optional[str], dl3dv_root: Path, scene_source: str
) -> Tuple[Path, Optional[str]]:
    if data is None and not scene_id:
        raise ValueError("Provide either --data or --scene_id.")

    if data is not None:
        candidate = data.expanduser().resolve()
    else:
        candidate = (dl3dv_root.expanduser().resolve() / scene_id).resolve()

    if not candidate.exists():
        raise FileNotFoundError(f"Input path does not exist: {candidate}")

    if candidate.is_file():
        if candidate.suffix != ".json":
            raise ValueError(f"--data file must be a transforms json file, got: {candidate}")
        return candidate, scene_id

    # Direct nerfstudio-style folder.
    if (candidate / "transforms.json").exists():
        inferred_scene_id = scene_id if scene_id else candidate.parent.name if candidate.name in {"nerfstudio", "gaussian_splat"} else candidate.name
        return candidate, inferred_scene_id

    # Scene root that contains nerfstudio/ or gaussian_splat/.
    for src in _candidate_sources(scene_source):
        subdir = candidate / src
        if (subdir / "transforms.json").exists():
            inferred_scene_id = scene_id if scene_id else candidate.name
            return subdir, inferred_scene_id

    looked_for = ", ".join(_candidate_sources(scene_source))
    raise FileNotFoundError(
        f"Could not find transforms.json in {candidate} or in its [{looked_for}] subfolders."
    )


def _default_experiment_name(scene_id: Optional[str], data_path: Path) -> str:
    if scene_id:
        return scene_id
    if data_path.is_file():
        return data_path.parent.name
    if data_path.name in {"nerfstudio", "gaussian_splat"}:
        return data_path.parent.name
    return data_path.name


def _ensure_nerfstudio_import_path(nerfstudio_path: Optional[Path]) -> None:
    if nerfstudio_path is None:
        return
    path = nerfstudio_path.expanduser().resolve()
    if not (path / "nerfstudio").is_dir():
        raise FileNotFoundError(
            f"--nerfstudio_path must contain a `nerfstudio` package directory. Got: {path}"
        )
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _import_symbol(module_name: str, symbol_name: str) -> Any:
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def _load_nerfstudio_symbols(prefer_parallel_datamanager: bool) -> Dict[str, Any]:
    TrainerConfig = _import_symbol("nerfstudio.engine.trainer", "TrainerConfig")
    ViewerConfig = _import_symbol("nerfstudio.configs.base_config", "ViewerConfig")
    CameraOptimizerConfig = _import_symbol("nerfstudio.cameras.camera_optimizers", "CameraOptimizerConfig")
    NerfstudioDataParserConfig = _import_symbol(
        "nerfstudio.data.dataparsers.nerfstudio_dataparser", "NerfstudioDataParserConfig"
    )
    VanillaPipelineConfig = _import_symbol("nerfstudio.pipelines.base_pipeline", "VanillaPipelineConfig")
    NerfactoModelConfig = _import_symbol("nerfstudio.models.nerfacto", "NerfactoModelConfig")
    AdamOptimizerConfig = _import_symbol("nerfstudio.engine.optimizers", "AdamOptimizerConfig")
    ExponentialDecaySchedulerConfig = _import_symbol("nerfstudio.engine.schedulers", "ExponentialDecaySchedulerConfig")
    DataManagerConfigClass = None
    if prefer_parallel_datamanager:
        try:
            DataManagerConfigClass = _import_symbol(
                "nerfstudio.data.datamanagers.parallel_datamanager", "ParallelDataManagerConfig"
            )
        except Exception:
            DataManagerConfigClass = None
    if DataManagerConfigClass is None:
        DataManagerConfigClass = _import_symbol(
            "nerfstudio.data.datamanagers.base_datamanager", "VanillaDataManagerConfig"
        )

    return {
        "TrainerConfig": TrainerConfig,
        "ViewerConfig": ViewerConfig,
        "CameraOptimizerConfig": CameraOptimizerConfig,
        "NerfstudioDataParserConfig": NerfstudioDataParserConfig,
        "VanillaPipelineConfig": VanillaPipelineConfig,
        "NerfactoModelConfig": NerfactoModelConfig,
        "AdamOptimizerConfig": AdamOptimizerConfig,
        "ExponentialDecaySchedulerConfig": ExponentialDecaySchedulerConfig,
        "DataManagerConfigClass": DataManagerConfigClass,
    }


def _build_config(args: argparse.Namespace, data_path: Path, experiment_name: str, ns: Dict[str, Any]) -> Any:
    dataparser_cfg = ns["NerfstudioDataParserConfig"](
        data=data_path,
        downscale_factor=args.downscale_factor,
        eval_mode=args.eval_mode,
        train_split_fraction=args.train_split_fraction,
        eval_interval=args.eval_interval,
        orientation_method=args.orientation_method,
        center_method=args.center_method,
    )
    datamanager_cfg = ns["DataManagerConfigClass"](
        dataparser=dataparser_cfg,
        train_num_rays_per_batch=args.train_num_rays_per_batch,
        eval_num_rays_per_batch=args.eval_num_rays_per_batch,
    )
    model_cfg = ns["NerfactoModelConfig"](
        eval_num_rays_per_chunk=args.eval_num_rays_per_chunk,
        average_init_density=args.average_init_density,
        camera_optimizer=ns["CameraOptimizerConfig"](mode=args.camera_optimizer_mode),
        implementation=args.implementation,
    )
    pipeline_cfg = ns["VanillaPipelineConfig"](
        datamanager=datamanager_cfg,
        model=model_cfg,
    )
    cfg = ns["TrainerConfig"](
        method_name=args.method_name,
        output_dir=args.output_dir.expanduser().resolve(),
        experiment_name=experiment_name,
        timestamp=args.timestamp,
        steps_per_eval_batch=args.steps_per_eval_batch,
        steps_per_save=args.steps_per_save,
        max_num_iterations=args.max_num_iterations,
        mixed_precision=not args.disable_mixed_precision,
        save_only_latest_checkpoint=not args.save_all_checkpoints,
        pipeline=pipeline_cfg,
        optimizers={
            "proposal_networks": {
                "optimizer": ns["AdamOptimizerConfig"](lr=args.proposal_lr, eps=1e-15),
                "scheduler": ns["ExponentialDecaySchedulerConfig"](
                    lr_final=args.proposal_lr_final, max_steps=args.proposal_lr_decay_steps
                ),
            },
            "fields": {
                "optimizer": ns["AdamOptimizerConfig"](lr=args.fields_lr, eps=1e-15),
                "scheduler": ns["ExponentialDecaySchedulerConfig"](
                    lr_final=args.fields_lr_final, max_steps=args.fields_lr_decay_steps
                ),
            },
            "camera_opt": {
                "optimizer": ns["AdamOptimizerConfig"](lr=args.camera_opt_lr, eps=1e-15),
                "scheduler": ns["ExponentialDecaySchedulerConfig"](
                    lr_final=args.camera_opt_lr_final, max_steps=args.camera_opt_lr_decay_steps
                ),
            },
        },
        viewer=ns["ViewerConfig"](num_rays_per_chunk=args.eval_num_rays_per_chunk),
        vis=args.vis,
    )
    cfg.machine.seed = args.seed
    cfg.machine.device_type = args.device_type
    cfg.machine.num_devices = args.num_devices
    cfg.machine.num_machines = args.num_machines
    cfg.machine.machine_rank = args.machine_rank
    cfg.machine.dist_url = args.dist_url
    cfg.data = data_path
    return cfg


def _print_summary(cfg: Any, data_path: Path, scene_id: Optional[str]) -> None:
    print("[info] Resolved data path:", data_path)
    if scene_id:
        print("[info] Scene id:", scene_id)
    print("[info] Output root:", cfg.output_dir)
    print("[info] Experiment:", cfg.experiment_name)
    print("[info] Method:", cfg.method_name)
    print("[info] Device:", cfg.machine.device_type, "num_devices=", cfg.machine.num_devices)
    print("[info] Max iterations:", cfg.max_num_iterations)


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = _parse_args()
    _ensure_nerfstudio_import_path(args.nerfstudio_path)
    try:
        ns = _load_nerfstudio_symbols(prefer_parallel_datamanager=args.prefer_parallel_datamanager)
    except Exception as exc:
        raise RuntimeError(
            "Failed to import Nerfstudio. Install nerfstudio>=0.3.0 or pass --nerfstudio_path "
            "to a directory containing the `nerfstudio` package."
        ) from exc

    data_path, resolved_scene_id = _resolve_data_path(
        data=args.data,
        scene_id=args.scene_id,
        dl3dv_root=args.dl3dv_root,
        scene_source=args.scene_source,
    )
    experiment_name = args.experiment_name or _default_experiment_name(resolved_scene_id, data_path)
    cfg = _build_config(args, data_path, experiment_name, ns)
    _print_summary(cfg, data_path, resolved_scene_id)

    if args.dry_run:
        print("[info] Dry run enabled. Exiting before training.")
        return

    if cfg.machine.num_devices != 1 or cfg.machine.num_machines != 1:
        raise ValueError(
            "This standalone launcher currently supports single-process training only "
            "(num_devices=1, num_machines=1)."
        )

    cfg.set_timestamp()
    cfg.print_to_terminal()
    cfg.save_config()

    _set_random_seed(cfg.machine.seed)
    trainer = cfg.setup(local_rank=0, world_size=1)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
