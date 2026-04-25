"""
Self-contained FreeFix-style runner built on this repository's 3DGS trainer.

The runner keeps the public scene-level CLI while dispatching to
`scripts/trainers/trainer.py` with `training_recipe=freefix`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.freefix_support import DEFAULT_NEGATIVE_PROMPT, DEFAULT_PROMPT


@dataclass
class Config:
    stage: str = "full"  # recon | refine | eval | full
    scene_id: Optional[str] = None
    data_dir: Optional[str] = None
    result_dir: Optional[str] = None
    ckpt: Optional[str] = None
    backend: str = "flux"  # flux | sdxl

    repo_root: str = "/mntdatalora/src/Diffusion-IBR"
    dl3dv_root: Optional[str] = None
    scene_data_subdir: str = "gaussian_splat"
    output_root: Optional[str] = None
    cache_root: Optional[str] = None

    data_factor: int = 4
    test_every: int = 8
    train_split_all: bool = True
    batch_size: int = 1
    num_workers: int = 4
    patch_size: Optional[int] = None
    normalize_world: bool = True
    normalize_align_axes: bool = True

    recon_steps: int = 30000
    recon_eval_steps: str = "7000,30000"
    recon_save_steps: str = "7000,30000"

    refine_cycles: int = 1
    refine_steps_per_cycle: int = 300
    refine_num_views: int = 0

    prompt: str = DEFAULT_PROMPT
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    freefix_num_inference_steps: int = 50
    freefix_strength: float = 0.5
    freefix_guidance_scale: float = 3.5
    freefix_guide_ratio: float = 1.0
    freefix_warp_ratio: float = 0.5
    freefix_hessian_attrs: str = "means"
    freefix_certainty_scales: str = "0.001,0.01,0.1"
    freefix_mask_scheduler: str = "0.3,0.9,1.0"
    freefix_mask_center: float = 0.5
    freefix_mask_softness: float = 10.0
    freefix_use_affine: bool = True

    gen_prob: float = 0.1
    gen_loss_weight: float = 0.2
    freefix_real_lambda: float = 1.0
    seed: int = 42
    device: str = "cuda"
    python_bin: str = "python"
    eval_after_refine: bool = True
    dry_run: bool = False


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Compatibility wrapper for the local FreeFix port.")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config")
    parser.add_argument("--stage", type=str, default="full", choices=["recon", "refine", "eval", "full"])
    parser.add_argument("--scene_id", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--backend", type=str, default="flux", choices=["flux", "sdxl"])
    parser.add_argument("--repo_root", type=str, default="/mntdatalora/src/Diffusion-IBR")
    parser.add_argument("--dl3dv_root", type=str, default=None)
    parser.add_argument("--scene_data_subdir", type=str, default="gaussian_splat")
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--cache_root", type=str, default=None)
    parser.add_argument("--data_factor", type=int, default=4)
    parser.add_argument("--test_every", type=int, default=8)
    parser.add_argument("--train_split_all", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--normalize_world", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize_align_axes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recon_steps", type=int, default=30000)
    parser.add_argument("--recon_eval_steps", type=str, default="7000,30000")
    parser.add_argument("--recon_save_steps", type=str, default="7000,30000")
    parser.add_argument("--refine_cycles", type=int, default=1)
    parser.add_argument("--refine_steps_per_cycle", type=int, default=300)
    parser.add_argument("--refine_num_views", type=int, default=0)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--freefix_num_inference_steps", type=int, default=50)
    parser.add_argument("--freefix_strength", type=float, default=0.5)
    parser.add_argument("--freefix_guidance_scale", type=float, default=3.5)
    parser.add_argument("--freefix_guide_ratio", type=float, default=1.0)
    parser.add_argument("--freefix_warp_ratio", type=float, default=0.5)
    parser.add_argument("--freefix_hessian_attrs", type=str, default="means")
    parser.add_argument("--freefix_certainty_scales", type=str, default="0.001,0.01,0.1")
    parser.add_argument("--freefix_mask_scheduler", type=str, default="0.3,0.9,1.0")
    parser.add_argument("--freefix_mask_center", type=float, default=0.5)
    parser.add_argument("--freefix_mask_softness", type=float, default=10.0)
    parser.add_argument("--freefix_use_affine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gen_prob", type=float, default=0.1)
    parser.add_argument("--gen_loss_weight", type=float, default=0.2)
    parser.add_argument("--freefix_real_lambda", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--python_bin", type=str, default="python")
    parser.add_argument("--eval_after_refine", action=argparse.BooleanOptionalAction, default=True)
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


def _resolve_scene_id(cfg: Config, data_dir: Optional[Path]) -> str:
    if cfg.scene_id is not None and len(str(cfg.scene_id).strip()) > 0:
        return str(cfg.scene_id).strip()
    if data_dir is None:
        raise ValueError("Provide --scene_id or --data_dir.")
    if data_dir.name == cfg.scene_data_subdir:
        return data_dir.parent.name
    return data_dir.name


def _resolve_paths(cfg: Config) -> tuple[Path, Path, str]:
    repo_root = Path(cfg.repo_root).expanduser().resolve()
    explicit_data_dir = Path(cfg.data_dir).expanduser().resolve() if cfg.data_dir else None
    scene_id = _resolve_scene_id(cfg, explicit_data_dir)

    if explicit_data_dir is not None:
        data_dir = explicit_data_dir
    else:
        dl3dv_root = (
            Path(cfg.dl3dv_root).expanduser().resolve()
            if cfg.dl3dv_root
            else repo_root / "data" / "DL3DV-10K-Benchmark"
        )
        data_dir = dl3dv_root / scene_id / cfg.scene_data_subdir

    if cfg.result_dir is not None:
        result_dir = Path(cfg.result_dir).expanduser().resolve()
    else:
        output_root = (
            Path(cfg.output_root).expanduser().resolve()
            if cfg.output_root
            else repo_root / "outputs" / "freefix_self"
        )
        result_dir = output_root / cfg.backend / scene_id

    return data_dir, result_dir, scene_id


def _default_base_ckpt(result_dir: Path, cfg: Config) -> Path:
    return result_dir / "ckpts" / f"ckpt_{int(cfg.recon_steps) - 1}_rank0.pt"


def _default_refined_ckpt(result_dir: Path, cfg: Config) -> Path:
    return result_dir / "ckpts" / f"ckpt_freefix_{cfg.backend}_rank0.pt"


def _resolve_hf_cache_dir(cfg: Config) -> str:
    cache_root = (
        Path(cfg.cache_root).expanduser().resolve()
        if cfg.cache_root
        else Path(cfg.repo_root).expanduser().resolve() / "cache_weights"
    )
    hub_dir = cache_root if cache_root.name == "hub" else cache_root / "huggingface" / "hub"
    hf_home = hub_dir.parent if hub_dir.name == "hub" else cache_root / "huggingface"
    transformers_cache = hf_home / "transformers"
    hub_dir.mkdir(parents=True, exist_ok=True)
    transformers_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(transformers_cache))
    return str(hub_dir)


def _trainer_kwargs(cfg: Config, data_dir: Path, result_dir: Path, recipe: str) -> dict:
    cache_dir = _resolve_hf_cache_dir(cfg)
    return {
        "data_dir": str(data_dir),
        "result_dir": str(result_dir),
        "data_factor": int(cfg.data_factor),
        "test_every": int(cfg.test_every),
        "normalize_world": bool(cfg.normalize_world),
        "normalize_align_axes": bool(cfg.normalize_align_axes),
        "patch_size": cfg.patch_size,
        "batch_size": int(cfg.batch_size),
        "num_workers": int(cfg.num_workers),
        "max_steps": int(cfg.recon_steps),
        "eval_steps": cfg.recon_eval_steps,
        "save_steps": cfg.recon_save_steps,
        "train_split_all": bool(cfg.train_split_all),
        "seed": int(cfg.seed),
        "device": cfg.device,
        "training_recipe": recipe,
        "use_freefix": recipe == "freefix",
        "fix_cache_dir": cache_dir,
        "freefix_backend": cfg.backend,
        "freefix_prompt": cfg.prompt,
        "freefix_negative_prompt": cfg.negative_prompt,
        "freefix_num_views": int(cfg.refine_num_views),
        "freefix_num_inference_steps": int(cfg.freefix_num_inference_steps),
        "freefix_strength": float(cfg.freefix_strength),
        "freefix_guidance_scale": float(cfg.freefix_guidance_scale),
        "freefix_novel_prob": float(cfg.gen_prob),
        "freefix_novel_lambda": float(cfg.gen_loss_weight),
        "freefix_real_lambda": float(cfg.freefix_real_lambda),
        "freefix_refine_steps": int(cfg.refine_steps_per_cycle),
        "freefix_use_affine": bool(cfg.freefix_use_affine),
        "freefix_certainty_scales": cfg.freefix_certainty_scales,
        "freefix_hessian_attrs": cfg.freefix_hessian_attrs,
        "freefix_mask_center": float(cfg.freefix_mask_center),
        "freefix_mask_softness": float(cfg.freefix_mask_softness),
        "freefix_mask_scheduler": cfg.freefix_mask_scheduler,
        "freefix_guide_ratio": float(cfg.freefix_guide_ratio),
        "freefix_warp_ratio": float(cfg.freefix_warp_ratio),
    }


def _make_trainer(cfg: Config, data_dir: Path, result_dir: Path, recipe: str):
    from scripts.trainers.trainer import Config as TrainerConfig
    from scripts.trainers.trainer import Trainer

    return Trainer(TrainerConfig(**_trainer_kwargs(cfg, data_dir, result_dir, recipe)))


def _load_required_checkpoint(trainer, ckpt_path: Path) -> int:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return trainer.load_checkpoint(str(ckpt_path))


def run(cfg: Config) -> None:
    stage = cfg.stage.lower().strip()
    if stage not in {"recon", "refine", "eval", "full"}:
        raise ValueError("stage must be one of: recon, refine, eval, full")
    backend = cfg.backend.lower().strip()
    if backend not in {"flux", "sdxl"}:
        raise ValueError("backend must be 'flux' or 'sdxl'")
    cfg.backend = backend
    if cfg.refine_cycles != 1:
        raise ValueError(
            "The self-contained FreeFix runner currently supports one refinement pass. "
            "Set --refine_cycles=1."
        )

    data_dir, result_dir, scene_id = _resolve_paths(cfg)
    base_ckpt = Path(cfg.ckpt).expanduser().resolve() if cfg.ckpt else _default_base_ckpt(result_dir, cfg)
    refined_ckpt = _default_refined_ckpt(result_dir, cfg)

    cuda_device = _cuda_device_arg(cfg.device)
    if cuda_device is not None:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    if cfg.dry_run:
        print(
            json.dumps(
                {
                    "runner": "freefix_self",
                    "stage": stage,
                    "scene_id": scene_id,
                    "backend": cfg.backend,
                    "data_dir": str(data_dir),
                    "result_dir": str(result_dir),
                    "base_checkpoint": str(base_ckpt),
                    "refined_checkpoint": str(refined_ckpt),
                    "trainer_config": _trainer_kwargs(
                        cfg,
                        data_dir,
                        result_dir,
                        "vanilla" if stage == "recon" else "freefix",
                    ),
                },
                indent=2,
            )
        )
        return

    if stage == "recon":
        trainer = _make_trainer(cfg, data_dir, result_dir, recipe="vanilla")
        trainer.train()
        return

    if stage == "full":
        trainer = _make_trainer(cfg, data_dir, result_dir, recipe="freefix")
        trainer.train()
        if cfg.eval_after_refine:
            trainer.eval(step=int(cfg.recon_steps) - 1)
        return

    trainer = _make_trainer(cfg, data_dir, result_dir, recipe="freefix")
    if stage == "refine":
        loaded_step = _load_required_checkpoint(trainer, base_ckpt)
        trainer.run_native_freefix_refinement(step=loaded_step if loaded_step >= 0 else int(cfg.recon_steps) - 1)
    elif stage == "eval":
        eval_ckpt = refined_ckpt if refined_ckpt.exists() else base_ckpt
        loaded_step = _load_required_checkpoint(trainer, eval_ckpt)
        trainer.eval(step=loaded_step if loaded_step >= 0 else int(cfg.recon_steps) - 1)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
