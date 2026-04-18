"""
Compatibility wrapper for the local FreeFix implementation.

This entrypoint preserves the repository-local FreeFix CLI surface while routing
execution through the local FreeFix pipeline under `scripts/freefix_impl`.
"""

from __future__ import annotations

import argparse
import importlib.util
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

import freefix_clean_runner as clean_runner

_FREEFIX_SUPPORT_SPEC = importlib.util.spec_from_file_location(
    "diffusion_ibr_freefix_support",
    PROJECT_ROOT / "utils" / "freefix_support.py",
)
if _FREEFIX_SUPPORT_SPEC is None or _FREEFIX_SUPPORT_SPEC.loader is None:
    raise ImportError("Failed to load utils/freefix_support.py")
_freefix_support = importlib.util.module_from_spec(_FREEFIX_SUPPORT_SPEC)
sys.modules[_FREEFIX_SUPPORT_SPEC.name] = _freefix_support
_FREEFIX_SUPPORT_SPEC.loader.exec_module(_freefix_support)

DEFAULT_NEGATIVE_PROMPT = _freefix_support.DEFAULT_NEGATIVE_PROMPT
DEFAULT_PROMPT = _freefix_support.DEFAULT_PROMPT


@dataclass
class Config:
    stage: str = "full"  # recon | refine | eval | full
    scene_id: Optional[str] = None
    backend: str = "flux"  # flux | sdxl

    repo_root: str = "/mntdatalora/src/Diffusion-IBR"
    dl3dv_root: Optional[str] = None
    scene_data_subdir: str = "gaussian_splat"
    output_root: Optional[str] = None

    data_factor: int = 4
    test_every: int = 8

    recon_steps: int = 30000
    recon_eval_steps: str = "7000,30000"
    recon_save_steps: str = "7000,30000"

    refine_cycles: int = 1
    refine_steps_per_cycle: int = 400
    refine_num_views: int = 0

    prompt: str = DEFAULT_PROMPT
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    freefix_num_inference_steps: int = 50
    freefix_strength: float = 0.6
    freefix_guidance_scale: float = 3.5
    freefix_guide_ratio: float = 1.0
    freefix_warp_ratio: float = 0.5
    freefix_hessian_attrs: str = "means,quats,scales"

    gen_prob: float = 0.1
    gen_loss_weight: float = 0.2
    device: str = "cuda"
    dry_run: bool = False


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Compatibility wrapper for the local FreeFix port.")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config")
    parser.add_argument("--stage", type=str, default="full", choices=["recon", "refine", "eval", "full"])
    parser.add_argument("--scene_id", type=str, default=None)
    parser.add_argument("--backend", type=str, default="flux", choices=["flux", "sdxl"])
    parser.add_argument("--repo_root", type=str, default="/mntdatalora/src/Diffusion-IBR")
    parser.add_argument("--dl3dv_root", type=str, default=None)
    parser.add_argument("--scene_data_subdir", type=str, default="gaussian_splat")
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--data_factor", type=int, default=4)
    parser.add_argument("--test_every", type=int, default=8)
    parser.add_argument("--recon_steps", type=int, default=30000)
    parser.add_argument("--recon_eval_steps", type=str, default="7000,30000")
    parser.add_argument("--recon_save_steps", type=str, default="7000,30000")
    parser.add_argument("--refine_cycles", type=int, default=1)
    parser.add_argument("--refine_steps_per_cycle", type=int, default=400)
    parser.add_argument("--refine_num_views", type=int, default=0)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--freefix_num_inference_steps", type=int, default=50)
    parser.add_argument("--freefix_strength", type=float, default=0.6)
    parser.add_argument("--freefix_guidance_scale", type=float, default=3.5)
    parser.add_argument("--freefix_guide_ratio", type=float, default=1.0)
    parser.add_argument("--freefix_warp_ratio", type=float, default=0.5)
    parser.add_argument("--freefix_hessian_attrs", type=str, default="means,quats,scales")
    parser.add_argument("--gen_prob", type=float, default=0.1)
    parser.add_argument("--gen_loss_weight", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry_run", action="store_true")

    args_pre, _ = parser.parse_known_args()
    if args_pre.config is not None:
        with open(args_pre.config, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        if not isinstance(config_data, dict):
            raise ValueError("Config JSON must be an object")
        valid_keys = {field.name for field in fields(Config)}
        unknown = sorted(str(key) for key in config_data.keys() if str(key) not in valid_keys)
        if unknown:
            raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
        parser.set_defaults(**config_data)

    args = parser.parse_args()
    data = vars(args)
    data.pop("config", None)
    return Config(**data)


def _cuda_device_arg(device: str) -> Optional[str]:
    text = str(device).strip().lower()
    if text == "cpu":
        return None
    if text == "cuda":
        return None
    if text.startswith("cuda:"):
        return text.split(":", 1)[1]
    return None


def run(cfg: Config) -> None:
    if cfg.scene_id is None or len(str(cfg.scene_id).strip()) == 0:
        raise ValueError("--scene_id is required")
    if cfg.refine_cycles != 1:
        raise ValueError(
            "The local official-style FreeFix port only supports a single refinement pass. "
            "Set --refine_cycles=1."
        )

    repo_root = Path(cfg.repo_root).resolve()
    freefix_root = repo_root / "scripts" / "freefix_impl"
    output_root = Path(cfg.output_root).resolve() if cfg.output_root else (repo_root / "outputs" / "freefix_self")

    clean_cfg = clean_runner.Config(
        stage=cfg.stage,
        scene_id=str(cfg.scene_id),
        backend=cfg.backend,
        repo_root=str(repo_root),
        dl3dv_root=cfg.dl3dv_root,
        scene_data_subdir=cfg.scene_data_subdir,
        output_root=str(output_root),
        freefix_root=str(freefix_root),
        data_type="colmap",
        data_factor=int(cfg.data_factor),
        test_every=int(cfg.test_every),
        max_steps=int(cfg.recon_steps),
        eval_steps=cfg.recon_eval_steps,
        save_steps=cfg.recon_save_steps,
        prompt=cfg.prompt,
        negative_prompt=cfg.negative_prompt,
        strength=float(cfg.freefix_strength),
        hessian_attrs=cfg.freefix_hessian_attrs,
        num_inference_steps=int(cfg.freefix_num_inference_steps),
        guide_ratio=float(cfg.freefix_guide_ratio),
        warp_ratio=float(cfg.freefix_warp_ratio),
        refine_steps=int(cfg.refine_steps_per_cycle),
        gen_prob=float(cfg.gen_prob),
        gen_loss_weight=float(cfg.gen_loss_weight),
        load_step=int(cfg.recon_steps - 1),
        eval_test=True,
        test_from_train=False,
        cuda_device=_cuda_device_arg(cfg.device),
        dry_run=bool(cfg.dry_run),
    )
    clean_runner.run(clean_cfg)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
