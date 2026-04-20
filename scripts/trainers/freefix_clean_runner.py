"""
Compatibility scene-level wrapper for the self-contained FreeFix runner.

The implementation delegates to `freefix_runner.py`, which uses the local 3DGS
trainer and diffusion fixers.
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

from scripts.trainers.freefix_runner import Config as SelfFreeFixConfig
from scripts.trainers.freefix_runner import run as run_self_freefix
from utils.freefix_support import DEFAULT_NEGATIVE_PROMPT, DEFAULT_PROMPT


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
    refine_num_views: int = 0
    strength: Optional[float] = None
    hessian_attrs: Optional[str] = None  # CSV, e.g. means,quats,scales
    num_inference_steps: int = 50
    guide_ratio: float = 1.0
    warp_ratio: float = 0.5
    refine_steps: int = 300
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
    parser.add_argument("--refine_num_views", type=int, default=0)
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--hessian_attrs", type=str, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guide_ratio", type=float, default=1.0)
    parser.add_argument("--warp_ratio", type=float, default=0.5)
    parser.add_argument("--refine_steps", type=int, default=300)
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


def resolve_paths(cfg: Config) -> tuple[Path, Path, Path, Path, Path]:
    repo_root = Path(cfg.repo_root).resolve()
    dl3dv_root = Path(cfg.dl3dv_root).resolve() if cfg.dl3dv_root else (repo_root / "data" / "DL3DV-10K-Benchmark")
    cache_root = Path(cfg.cache_root).resolve() if cfg.cache_root else (repo_root / "cache_weights")
    output_root = Path(cfg.output_root).resolve() if cfg.output_root else (repo_root / "outputs" / "freefix_self_clean")
    scene_data_dir = dl3dv_root / str(cfg.scene_id) / cfg.scene_data_subdir
    return repo_root, dl3dv_root, cache_root, output_root, scene_data_dir


def run(cfg: Config) -> None:
    if cfg.scene_id is None or len(str(cfg.scene_id).strip()) == 0:
        raise ValueError("--scene_id is required")

    repo_root, dl3dv_root, cache_root, output_root, scene_data_dir = resolve_paths(cfg)
    del dl3dv_root

    output_dir = output_root / cfg.backend / str(cfg.scene_id)
    runner_cfg = SelfFreeFixConfig(
        stage=cfg.stage,
        scene_id=str(cfg.scene_id),
        backend=cfg.backend,
        repo_root=str(repo_root),
        dl3dv_root=cfg.dl3dv_root,
        scene_data_subdir=cfg.scene_data_subdir,
        output_root=str(output_root),
        cache_root=str(cache_root),
        data_dir=str(scene_data_dir),
        result_dir=str(output_dir),
        data_factor=int(cfg.data_factor),
        test_every=cfg.test_every,
        recon_steps=int(cfg.max_steps),
        recon_eval_steps=cfg.eval_steps,
        recon_save_steps=cfg.save_steps,
        prompt=cfg.prompt,
        negative_prompt=cfg.negative_prompt,
        refine_num_views=int(cfg.refine_num_views),
        freefix_strength=0.5 if cfg.strength is None else float(cfg.strength),
        freefix_hessian_attrs=",".join(parse_hessian_attrs(cfg.hessian_attrs) or ["means"]),
        freefix_num_inference_steps=cfg.num_inference_steps,
        freefix_guide_ratio=cfg.guide_ratio,
        freefix_warp_ratio=cfg.warp_ratio,
        refine_steps_per_cycle=cfg.refine_steps,
        gen_prob=cfg.gen_prob,
        gen_loss_weight=cfg.gen_loss_weight,
        device=f"cuda:{cfg.cuda_device}" if cfg.cuda_device else "cuda",
        dry_run=cfg.dry_run,
    )
    run_self_freefix(runner_cfg)


def main() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
