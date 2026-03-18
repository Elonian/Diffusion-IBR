"""
Clean FreeFix orchestrator for DL3DV scenes.

This script keeps the execution logic in this repository while calling the
official FreeFix code paths under `works/FreeFix` for recon/refine/eval.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRAINER_DIR = Path(__file__).resolve().parent
if str(TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINER_DIR))

import freefix_official_runner as official_runner
from utils.freefix_support import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT,
    generate_freefix_scene_assets,
)


@dataclass
class Config:
    stage: str = "full"  # recon | refine | eval | full
    scene_id: Optional[str] = None
    backend: str = "flux"  # flux | sdxl

    repo_root: str = "/mntdatalora/src/Diffusion-IBR"
    dl3dv_root: Optional[str] = None
    scene_data_subdir: str = "gaussian_splat"
    output_root: Optional[str] = None
    freefix_root: Optional[str] = None
    cache_root: Optional[str] = None
    base_cfg: Optional[str] = None

    data_type: str = "colmap"
    data_factor: int = 4
    test_every: int = 8
    max_steps: int = 30000
    eval_steps: str = "7000,30000"
    save_steps: str = "7000,30000"
    sync_exp_base_dir_with_result_dir: bool = True

    prompt: str = DEFAULT_PROMPT
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    strength: Optional[float] = None
    hessian_attrs: Optional[str] = None  # CSV, e.g. means,quats,scales
    num_inference_steps: int = 50
    guide_ratio: float = 1.0
    warp_ratio: float = 0.5
    refine_steps: int = 400
    gen_prob: float = 0.1
    gen_loss_weight: float = 0.2
    load_step: int = 29999

    eval_test: bool = True
    test_from_train: bool = False
    cuda_device: Optional[str] = None
    dry_run: bool = False


def parse_hessian_attrs(value: Optional[str]) -> Optional[list[str]]:
    if value is None:
        return None
    attrs = [token.strip() for token in str(value).split(",") if token.strip()]
    return attrs if attrs else None


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Clean FreeFix runner for DL3DV scenes.")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config.")
    parser.add_argument("--stage", type=str, default="full", choices=["recon", "refine", "eval", "full"])
    parser.add_argument("--scene_id", type=str, default=None, help="DL3DV scene id.")
    parser.add_argument("--backend", type=str, default="flux", choices=["flux", "sdxl"])
    parser.add_argument("--repo_root", type=str, default="/mntdatalora/src/Diffusion-IBR")
    parser.add_argument("--dl3dv_root", type=str, default=None)
    parser.add_argument("--scene_data_subdir", type=str, default="gaussian_splat")
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--freefix_root", type=str, default=None)
    parser.add_argument("--cache_root", type=str, default=None)
    parser.add_argument("--base_cfg", type=str, default=None)
    parser.add_argument("--data_type", type=str, default="colmap", choices=["colmap", "hugsim"])
    parser.add_argument("--data_factor", type=int, default=4)
    parser.add_argument("--test_every", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--eval_steps", type=str, default="7000,30000")
    parser.add_argument("--save_steps", type=str, default="7000,30000")
    parser.add_argument(
        "--sync_exp_base_dir_with_result_dir",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--hessian_attrs", type=str, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guide_ratio", type=float, default=1.0)
    parser.add_argument("--warp_ratio", type=float, default=0.5)
    parser.add_argument("--refine_steps", type=int, default=400)
    parser.add_argument("--gen_prob", type=float, default=0.1)
    parser.add_argument("--gen_loss_weight", type=float, default=0.2)
    parser.add_argument("--load_step", type=int, default=29999)
    parser.add_argument("--eval_test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--test_from_train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cuda_device", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")

    args_pre, _ = parser.parse_known_args()
    if args_pre.config is not None:
        with open(args_pre.config, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        if not isinstance(config_data, dict):
            raise ValueError("Config JSON must be an object.")
        valid_keys = {field.name for field in fields(Config)}
        unknown = sorted(str(key) for key in config_data.keys() if str(key) not in valid_keys)
        if unknown:
            raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
        parser.set_defaults(**config_data)

    args = parser.parse_args()
    data = vars(args)
    data.pop("config", None)
    return Config(**data)


def resolve_paths(cfg: Config) -> tuple[Path, Path, Path, Path, Path, Path]:
    repo_root = Path(cfg.repo_root).resolve()
    dl3dv_root = Path(cfg.dl3dv_root).resolve() if cfg.dl3dv_root else (repo_root / "data" / "DL3DV-10K-Benchmark")
    freefix_root = Path(cfg.freefix_root).resolve() if cfg.freefix_root else (repo_root / "works" / "FreeFix")
    cache_root = Path(cfg.cache_root).resolve() if cfg.cache_root else (repo_root / "cache_weights")
    output_root = Path(cfg.output_root).resolve() if cfg.output_root else (repo_root / "outputs" / "official_freefix_clean")
    scene_data_dir = dl3dv_root / str(cfg.scene_id) / cfg.scene_data_subdir
    return repo_root, dl3dv_root, freefix_root, cache_root, output_root, scene_data_dir


def run(cfg: Config) -> None:
    if cfg.scene_id is None or len(str(cfg.scene_id).strip()) == 0:
        raise ValueError("--scene_id is required")

    (
        _repo_root,
        _dl3dv_root,
        freefix_root,
        cache_root,
        output_root,
        scene_data_dir,
    ) = resolve_paths(cfg)

    output_dir = output_root / cfg.backend / str(cfg.scene_id)
    base_cfg = Path(cfg.base_cfg).resolve() if cfg.base_cfg else (freefix_root / "exp_cfg" / "base.yaml")
    assets = generate_freefix_scene_assets(
        scene_id=str(cfg.scene_id),
        scene_data_dir=scene_data_dir,
        output_dir=output_dir,
        backend=cfg.backend,
        test_every=cfg.test_every,
        prompt=cfg.prompt,
        negative_prompt=cfg.negative_prompt,
        strength=cfg.strength,
        hessian_attr=parse_hessian_attrs(cfg.hessian_attrs),
        num_inference_steps=cfg.num_inference_steps,
        guide_ratio=cfg.guide_ratio,
        warp_ratio=cfg.warp_ratio,
        refine_steps=cfg.refine_steps,
        gen_prob=cfg.gen_prob,
        gen_loss_weight=cfg.gen_loss_weight,
        load_step=cfg.load_step,
        data_type=cfg.data_type,
    )

    print(f"[clean-freefix] scene={assets.scene_id} backend={assets.backend}")
    print(f"[clean-freefix] data_dir={assets.scene_data_dir}")
    print(f"[clean-freefix] output_dir={assets.output_dir}")
    print(f"[clean-freefix] partition={assets.partition_path}")
    print(f"[clean-freefix] exp_cfg={assets.exp_cfg_path}")

    runner_cfg = official_runner.Config(
        stage=cfg.stage,
        freefix_root=str(freefix_root),
        cache_root=str(cache_root),
        cuda_device=cfg.cuda_device,
        dry_run=cfg.dry_run,
        data_dir=str(scene_data_dir),
        result_dir=str(output_dir),
        data_type=cfg.data_type,
        data_factor=int(cfg.data_factor),
        test_every=int(cfg.test_every),
        partition=str(assets.partition_path),
        max_steps=int(cfg.max_steps),
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        sync_exp_base_dir_with_result_dir=bool(cfg.sync_exp_base_dir_with_result_dir),
        backend=cfg.backend,
        exp_cfg=str(assets.exp_cfg_path),
        base_cfg=str(base_cfg),
        eval_test=bool(cfg.eval_test),
        test_from_train=bool(cfg.test_from_train),
    )

    official_runner._set_runtime_env(runner_cfg)
    if runner_cfg.stage in {"recon", "full"}:
        official_runner.run_recon(runner_cfg)
    if runner_cfg.stage in {"refine", "full"}:
        official_runner.run_refine(runner_cfg)
    if runner_cfg.stage in {"eval", "full"}:
        official_runner.run_eval(runner_cfg)


def main() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
