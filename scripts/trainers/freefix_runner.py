"""
Self-contained FreeFix runner built only on this repository's training stack.

Stages:
1) recon  - vanilla 3DGS reconstruction
2) refine - FreeFix-style iterative pseudo-view updates
3) eval   - quantitative evaluation on the test split
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.freefix_support import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT,
    generate_freefix_scene_assets,
)


@dataclass
class Config:
    stage: str = "full"  # recon | refine | eval | full
    scene_id: Optional[str] = None
    backend: str = "sdxl"  # flux | sdxl
    split_mode: str = "freefix"  # freefix | difix3d (for strict comparison)

    repo_root: str = "/mntdatalora/src/Diffusion-IBR"
    dl3dv_root: Optional[str] = None
    scene_data_subdir: str = "gaussian_splat"
    output_root: Optional[str] = None

    data_factor: int = 4
    test_every: int = 8
    num_workers: int = 2

    recon_steps: int = 30000
    recon_eval_steps: str = "7000,30000"
    recon_save_steps: str = "7000,30000"

    refine_cycles: int = 1
    refine_steps_per_cycle: int = 400
    refine_num_views: int = 0
    refine_eval_every_cycle: bool = False

    prompt: str = DEFAULT_PROMPT
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    freefix_num_inference_steps: int = 50
    freefix_strength: float = 0.6
    freefix_guidance_scale: float = 3.5
    freefix_mask_scheduler: str = "0.3,0.9,1.0"
    freefix_guide_ratio: float = 1.0
    freefix_warp_ratio: float = 0.5
    freefix_certainty_scales: str = "0.001,0.01,0.1"
    freefix_hessian_attrs: str = "means,quats,scales"

    gen_prob: float = 0.1
    gen_loss_weight: float = 0.2
    seed: int = 42
    device: str = "cuda"
    python_bin: str = "python"
    dry_run: bool = False


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Self-contained FreeFix runner.")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config")
    parser.add_argument("--stage", type=str, default="full", choices=["recon", "refine", "eval", "full"])
    parser.add_argument("--scene_id", type=str, default=None)
    parser.add_argument("--backend", type=str, default="sdxl", choices=["flux", "sdxl"])
    parser.add_argument("--split_mode", type=str, default="freefix", choices=["freefix", "difix3d"])
    parser.add_argument("--repo_root", type=str, default="/mntdatalora/src/Diffusion-IBR")
    parser.add_argument("--dl3dv_root", type=str, default=None)
    parser.add_argument("--scene_data_subdir", type=str, default="gaussian_splat")
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--data_factor", type=int, default=4)
    parser.add_argument("--test_every", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--recon_steps", type=int, default=30000)
    parser.add_argument("--recon_eval_steps", type=str, default="7000,30000")
    parser.add_argument("--recon_save_steps", type=str, default="7000,30000")
    parser.add_argument("--refine_cycles", type=int, default=1)
    parser.add_argument("--refine_steps_per_cycle", type=int, default=400)
    parser.add_argument("--refine_num_views", type=int, default=0)
    parser.add_argument(
        "--refine_eval_every_cycle",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--freefix_num_inference_steps", type=int, default=50)
    parser.add_argument("--freefix_strength", type=float, default=0.6)
    parser.add_argument("--freefix_guidance_scale", type=float, default=3.5)
    parser.add_argument("--freefix_mask_scheduler", type=str, default="0.3,0.9,1.0")
    parser.add_argument("--freefix_guide_ratio", type=float, default=1.0)
    parser.add_argument("--freefix_warp_ratio", type=float, default=0.5)
    parser.add_argument("--freefix_certainty_scales", type=str, default="0.001,0.01,0.1")
    parser.add_argument("--freefix_hessian_attrs", type=str, default="means,quats,scales")
    parser.add_argument("--gen_prob", type=float, default=0.1)
    parser.add_argument("--gen_loss_weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--python_bin", type=str, default="python")
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


def _split_csv(text: str) -> list[int]:
    out: list[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if len(token) == 0:
            continue
        out.append(int(token))
    return out


def _ensure_recon_checkpoint(output_dir: Path, recon_steps: int) -> Path:
    ckpt = output_dir / "ckpts" / f"ckpt_{recon_steps - 1}_rank0.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing reconstruction checkpoint: {ckpt}")
    return ckpt


def _final_refine_step(cfg: Config) -> int:
    return int(cfg.recon_steps + cfg.refine_cycles * cfg.refine_steps_per_cycle - 1)


def _run_trainer(repo_root: Path, python_bin: str, trainer_cfg: dict, cfg_path: Path, dry_run: bool) -> None:
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(trainer_cfg, indent=2) + "\n", encoding="utf-8")
    cmd = [
        python_bin,
        str(repo_root / "scripts" / "trainers" / "trainer.py"),
        "--config",
        str(cfg_path),
    ]
    if dry_run:
        print("[dry-run] command:", " ".join(cmd))
        print("[dry-run] config:", cfg_path)
        return
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def run(cfg: Config) -> None:
    if cfg.scene_id is None or len(str(cfg.scene_id).strip()) == 0:
        raise ValueError("--scene_id is required")
    if cfg.refine_cycles <= 0:
        raise ValueError("--refine_cycles must be > 0")
    if cfg.refine_steps_per_cycle <= 0:
        raise ValueError("--refine_steps_per_cycle must be > 0")

    repo_root = Path(cfg.repo_root).resolve()
    dl3dv_root = Path(cfg.dl3dv_root).resolve() if cfg.dl3dv_root else (repo_root / "data" / "DL3DV-10K-Benchmark")
    output_root = Path(cfg.output_root).resolve() if cfg.output_root else (repo_root / "outputs" / "freefix_self")

    scene_data_dir = dl3dv_root / str(cfg.scene_id) / cfg.scene_data_subdir
    output_dir = output_root / cfg.backend / str(cfg.scene_id)
    run_cfg_dir = output_dir / "run_cfg"
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_partition_name: Optional[str] = None
    previous_recon_cfg = run_cfg_dir / "recon.json"
    if previous_recon_cfg.is_file():
        try:
            previous_cfg = json.loads(previous_recon_cfg.read_text(encoding="utf-8"))
            previous_partition = previous_cfg.get("partition_file")
            if isinstance(previous_partition, str) and len(previous_partition) > 0:
                previous_partition_name = Path(previous_partition).name
        except Exception:
            previous_partition_name = None

    assets = generate_freefix_scene_assets(
        scene_id=str(cfg.scene_id),
        scene_data_dir=scene_data_dir,
        output_dir=output_dir,
        backend=cfg.backend,
        split_mode=cfg.split_mode,
        test_every=cfg.test_every,
        prompt=cfg.prompt,
        negative_prompt=cfg.negative_prompt,
        strength=cfg.freefix_strength,
        hessian_attr=[v.strip() for v in cfg.freefix_hessian_attrs.split(",") if v.strip()],
        num_inference_steps=cfg.freefix_num_inference_steps,
        guide_ratio=cfg.freefix_guide_ratio,
        warp_ratio=cfg.freefix_warp_ratio,
        refine_steps=cfg.refine_steps_per_cycle,
        gen_prob=cfg.gen_prob,
        gen_loss_weight=cfg.gen_loss_weight,
    )
    expected_partition_name = Path(assets.partition_path).name
    resume_compatible = True
    if previous_partition_name is not None and previous_partition_name != expected_partition_name:
        resume_compatible = False
        print(
            "[resume] Existing run partition mismatch: "
            f"previous={previous_partition_name} expected={expected_partition_name}. "
            "Will not auto-skip recon/refine."
        )

    print(
        f"[freefix-runner] scene={assets.scene_id} backend={assets.backend} "
        f"split_mode={assets.split_mode}"
    )
    print(f"[freefix-runner] data_dir={assets.scene_data_dir}")
    print(f"[freefix-runner] output_dir={assets.output_dir}")
    print(f"[freefix-runner] partition={assets.partition_path}")

    recon_eval_steps = _split_csv(cfg.recon_eval_steps)
    recon_save_steps = _split_csv(cfg.recon_save_steps)
    if cfg.recon_steps not in recon_eval_steps:
        recon_eval_steps.append(cfg.recon_steps)
    if cfg.recon_steps not in recon_save_steps:
        recon_save_steps.append(cfg.recon_steps)

    run_recon = cfg.stage in {"recon", "full"}
    run_refine = cfg.stage in {"refine", "full"}
    run_eval = cfg.stage in {"eval", "full"}

    if cfg.stage == "full" and not cfg.dry_run and resume_compatible:
        recon_ckpt_existing = output_dir / "ckpts" / f"ckpt_{cfg.recon_steps - 1}_rank0.pt"
        refine_ckpt_existing = output_dir / "ckpts" / f"ckpt_{_final_refine_step(cfg)}_rank0.pt"
        if refine_ckpt_existing.is_file():
            run_recon = False
            run_refine = False
            print(f"[resume] found refine checkpoint, skipping recon/refine: {refine_ckpt_existing}")
        elif recon_ckpt_existing.is_file():
            run_recon = False
            print(f"[resume] found recon checkpoint, skipping recon: {recon_ckpt_existing}")
    elif cfg.stage == "full" and not cfg.dry_run and not resume_compatible:
        print("[resume] forcing full recon/refine due split mismatch with existing checkpoints.")

    if run_recon:
        recon_cfg = {
            "data_dir": str(scene_data_dir),
            "result_dir": str(output_dir),
            "data_factor": int(cfg.data_factor),
            "test_every": int(cfg.test_every),
            "num_workers": int(cfg.num_workers),
            "partition_file": str(assets.partition_path),
            "max_steps": int(cfg.recon_steps),
            "eval_steps": ",".join(str(v) for v in sorted(set(recon_eval_steps))),
            "save_steps": ",".join(str(v) for v in sorted(set(recon_save_steps))),
            "mode": "train",
            "training_recipe": "vanilla",
            "seed": int(cfg.seed),
            "device": cfg.device,
        }
        _run_trainer(
            repo_root=repo_root,
            python_bin=cfg.python_bin,
            trainer_cfg=recon_cfg,
            cfg_path=run_cfg_dir / "recon.json",
            dry_run=cfg.dry_run,
        )

    if run_refine:
        if cfg.dry_run:
            recon_ckpt = output_dir / "ckpts" / f"ckpt_{cfg.recon_steps - 1}_rank0.pt"
        else:
            recon_ckpt = _ensure_recon_checkpoint(output_dir, cfg.recon_steps)
        cycle_steps = [cfg.recon_steps + i * cfg.refine_steps_per_cycle for i in range(cfg.refine_cycles)]
        refine_total_steps = cfg.recon_steps + cfg.refine_cycles * cfg.refine_steps_per_cycle
        refine_eval_steps = [refine_total_steps]
        if cfg.refine_eval_every_cycle:
            refine_eval_steps = [step + cfg.refine_steps_per_cycle for step in cycle_steps]

        refine_cfg = {
            "data_dir": str(scene_data_dir),
            "result_dir": str(output_dir),
            "data_factor": int(cfg.data_factor),
            "test_every": int(cfg.test_every),
            "num_workers": int(cfg.num_workers),
            "partition_file": str(assets.partition_path),
            "mode": "train",
            "ckpt": str(recon_ckpt),
            "max_steps": int(refine_total_steps),
            "save_steps": ",".join(str(v) for v in sorted(set(refine_eval_steps))),
            "eval_steps": ",".join(str(v) for v in sorted(set(refine_eval_steps))),
            "fix_steps": ",".join(str(v) for v in cycle_steps),
            "training_recipe": "freefix",
            "freefix_backend": cfg.backend,
            "freefix_prompt": cfg.prompt,
            "freefix_negative_prompt": cfg.negative_prompt,
            "freefix_start_step": int(cycle_steps[0]),
            "freefix_fix_every": int(cfg.refine_steps_per_cycle),
            "freefix_num_views": int(cfg.refine_num_views),
            "freefix_num_inference_steps": int(cfg.freefix_num_inference_steps),
            "freefix_strength": float(cfg.freefix_strength),
            "freefix_guidance_scale": float(cfg.freefix_guidance_scale),
            "freefix_mask_scheduler": cfg.freefix_mask_scheduler,
            "freefix_guide_ratio": float(cfg.freefix_guide_ratio),
            "freefix_warp_ratio": float(cfg.freefix_warp_ratio),
            "freefix_certainty_scales": cfg.freefix_certainty_scales,
            "freefix_hessian_attrs": cfg.freefix_hessian_attrs,
            "freefix_novel_prob": float(cfg.gen_prob),
            "freefix_novel_lambda": float(cfg.gen_loss_weight),
            "freefix_real_lambda": 1.0,
            "seed": int(cfg.seed),
            "device": cfg.device,
        }
        _run_trainer(
            repo_root=repo_root,
            python_bin=cfg.python_bin,
            trainer_cfg=refine_cfg,
            cfg_path=run_cfg_dir / "refine.json",
            dry_run=cfg.dry_run,
        )

    if run_eval:
        eval_ckpt = output_dir / "ckpts" / f"ckpt_{_final_refine_step(cfg)}_rank0.pt"
        eval_recipe = "freefix"
        if not eval_ckpt.is_file():
            if cfg.dry_run:
                eval_ckpt = output_dir / "ckpts" / f"ckpt_{cfg.recon_steps - 1}_rank0.pt"
            else:
                eval_ckpt = _ensure_recon_checkpoint(output_dir, cfg.recon_steps)
            eval_recipe = "vanilla"

        eval_cfg = {
            "data_dir": str(scene_data_dir),
            "result_dir": str(output_dir),
            "data_factor": int(cfg.data_factor),
            "test_every": int(cfg.test_every),
            "num_workers": int(cfg.num_workers),
            "partition_file": str(assets.partition_path),
            "mode": "eval",
            "ckpt": str(eval_ckpt),
            "training_recipe": eval_recipe,
            "freefix_backend": cfg.backend,
            "seed": int(cfg.seed),
            "device": cfg.device,
        }
        _run_trainer(
            repo_root=repo_root,
            python_bin=cfg.python_bin,
            trainer_cfg=eval_cfg,
            cfg_path=run_cfg_dir / "eval.json",
            dry_run=cfg.dry_run,
        )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
