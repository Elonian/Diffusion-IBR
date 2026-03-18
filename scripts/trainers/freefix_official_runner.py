"""
Run the official FreeFix pipeline from this repository's script entrypoint.

This wrapper executes the official code under `works/FreeFix`:
1) `recon.trainer` for vanilla 3DGS reconstruction
2) `ours.refine_by_flux` / `ours.refine_by_sdxl` for diffusion refinement
3) `ours.evaluation` for metric evaluation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import List, Optional


@dataclass
class Config:
    stage: str = "full"  # recon | refine | eval | full
    freefix_root: str = "/mntdatalora/src/Diffusion-IBR/works/FreeFix"
    cache_root: str = "/mntdatalora/src/Diffusion-IBR/cache_weights"
    cuda_device: Optional[str] = None
    dry_run: bool = False

    # recon stage
    data_dir: Optional[str] = None
    result_dir: Optional[str] = None
    data_type: str = "colmap"
    data_factor: int = 4
    test_every: int = 8
    partition: Optional[str] = None
    max_steps: int = 30000
    eval_steps: str = "7000,30000"
    save_steps: str = "7000,30000"
    sync_exp_base_dir_with_result_dir: bool = True

    # refine/eval stages
    backend: str = "flux"  # flux | sdxl
    exp_cfg: Optional[str] = None
    base_cfg: Optional[str] = None
    eval_test: bool = True
    test_from_train: bool = False


def _parse_int_csv(value: str, default: List[int]) -> List[int]:
    text = str(value).strip()
    if len(text) == 0:
        return list(default)
    out: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if len(token) == 0:
            continue
        out.append(int(token))
    return out if out else list(default)


def _ensure_freefix_on_path(freefix_root: str) -> None:
    resolved = str(Path(freefix_root).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _resolve_from_freefix_root(path_value: Optional[str], freefix_root: str) -> Optional[str]:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if len(text) == 0:
        return None
    path = Path(text)
    if path.is_absolute():
        return str(path)
    return str((Path(freefix_root) / path).resolve())


def _set_runtime_env(cfg: Config) -> None:
    if cfg.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.cuda_device)

    os.environ["DIFFUSION_IBR_CACHE_DIR"] = cfg.cache_root
    hf_home = os.path.join(cfg.cache_root, "huggingface")
    hf_hub_cache = os.path.join(hf_home, "hub")
    transformers_cache = os.path.join(hf_home, "transformers")
    os.makedirs(hf_hub_cache, exist_ok=True)
    os.makedirs(transformers_cache, exist_ok=True)
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_hub_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", transformers_cache)


def _require(cfg: Config, keys: List[str]) -> None:
    missing: List[str] = []
    for key in keys:
        value = getattr(cfg, key)
        if value is None or (isinstance(value, str) and len(value.strip()) == 0):
            missing.append(key)
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(missing))


def run_recon(cfg: Config) -> None:
    _require(cfg, ["data_dir", "result_dir"])

    eval_steps = _parse_int_csv(cfg.eval_steps, default=[7000, cfg.max_steps])
    save_steps = _parse_int_csv(cfg.save_steps, default=[7000, cfg.max_steps])
    data_dir = _resolve_from_freefix_root(cfg.data_dir, cfg.freefix_root)
    result_dir = _resolve_from_freefix_root(cfg.result_dir, cfg.freefix_root)
    partition = _resolve_from_freefix_root(cfg.partition, cfg.freefix_root)
    if cfg.dry_run:
        print("[dry-run] recon config:")
        print(
            json.dumps(
                {
                    "disable_viewer": True,
                    "data_type": cfg.data_type,
                    "data_dir": str(data_dir),
                    "data_factor": int(cfg.data_factor),
                    "result_dir": str(result_dir),
                    "test_every": int(cfg.test_every),
                    "partition": partition,
                    "max_steps": int(cfg.max_steps),
                    "eval_steps": eval_steps,
                    "save_steps": save_steps,
                },
                indent=2,
            )
        )
        return

    _ensure_freefix_on_path(cfg.freefix_root)
    from recon.trainer import Config as FreeFixReconConfig
    from recon.trainer import main as freefix_recon_main

    recon_cfg = FreeFixReconConfig(
        disable_viewer=True,
        data_type=cfg.data_type,  # colmap | hugsim
        data_dir=str(data_dir),
        data_factor=int(cfg.data_factor),
        result_dir=str(result_dir),
        test_every=int(cfg.test_every),
        partition=partition,
        max_steps=int(cfg.max_steps),
        eval_steps=eval_steps,
        save_steps=save_steps,
    )
    freefix_recon_main(recon_cfg)


def _resolve_base_cfg(cfg: Config) -> Path:
    if cfg.base_cfg is not None and len(str(cfg.base_cfg).strip()) > 0:
        return Path(_resolve_from_freefix_root(cfg.base_cfg, cfg.freefix_root))
    return Path(cfg.freefix_root) / "exp_cfg" / "base.yaml"


def _resolve_exp_cfg(cfg: Config) -> Path:
    _require(cfg, ["exp_cfg"])
    return Path(_resolve_from_freefix_root(cfg.exp_cfg, cfg.freefix_root))


def _sync_exp_base_dir(cfg: Config, merged_cfg: object) -> str:
    # Keep refine/eval on the same reconstruction output by default.
    if cfg.sync_exp_base_dir_with_result_dir and cfg.result_dir is not None:
        base_dir = _resolve_from_freefix_root(cfg.result_dir, cfg.freefix_root)
    else:
        if not hasattr(merged_cfg, "base_dir"):
            raise ValueError("Merged exp config is missing required key: base_dir")
        base_dir = str(getattr(merged_cfg, "base_dir"))
        base_dir = _resolve_from_freefix_root(base_dir, cfg.freefix_root)
    setattr(merged_cfg, "base_dir", base_dir)
    return str(base_dir)


def run_refine(cfg: Config) -> None:
    _require(cfg, ["exp_cfg"])

    base_cfg_path = _resolve_base_cfg(cfg)
    exp_cfg_path = _resolve_exp_cfg(cfg)

    backend = cfg.backend.lower().strip()
    if backend not in {"flux", "sdxl"}:
        raise ValueError("backend must be 'flux' or 'sdxl'")

    if cfg.dry_run:
        synced_base_dir = (
            _resolve_from_freefix_root(cfg.result_dir, cfg.freefix_root)
            if cfg.sync_exp_base_dir_with_result_dir and cfg.result_dir is not None
            else "(from exp_cfg.base_dir)"
        )
        print("[dry-run] refine backend:", backend)
        print("[dry-run] base_cfg:", base_cfg_path)
        print("[dry-run] exp_cfg:", exp_cfg_path)
        print("[dry-run] refine base_dir:", synced_base_dir)
        return

    _ensure_freefix_on_path(cfg.freefix_root)
    from omegaconf import OmegaConf

    merged_cfg = OmegaConf.merge(OmegaConf.load(base_cfg_path), OmegaConf.load(exp_cfg_path))
    _sync_exp_base_dir(cfg, merged_cfg)
    if backend == "flux":
        from ours.refine_by_flux import refine as freefix_refine
    else:
        from ours.refine_by_sdxl import refine as freefix_refine
    freefix_refine(merged_cfg)


def run_eval(cfg: Config) -> None:
    _require(cfg, ["exp_cfg"])

    base_cfg_path = _resolve_base_cfg(cfg)
    exp_cfg_path = _resolve_exp_cfg(cfg)

    if cfg.dry_run:
        synced_base_dir = (
            _resolve_from_freefix_root(cfg.result_dir, cfg.freefix_root)
            if cfg.sync_exp_base_dir_with_result_dir and cfg.result_dir is not None
            else "(from exp_cfg.base_dir)"
        )
        print("[dry-run] evaluation")
        print("[dry-run] base_cfg:", base_cfg_path)
        print("[dry-run] exp_cfg:", exp_cfg_path)
        print("[dry-run] eval base_dir:", synced_base_dir)
        print("[dry-run] eval_test:", cfg.eval_test)
        print("[dry-run] test_from_train:", cfg.test_from_train)
        return

    _ensure_freefix_on_path(cfg.freefix_root)
    from omegaconf import OmegaConf
    from ours.evaluation import eval as freefix_eval

    merged_cfg = OmegaConf.merge(OmegaConf.load(base_cfg_path), OmegaConf.load(exp_cfg_path))
    _sync_exp_base_dir(cfg, merged_cfg)
    freefix_eval(
        merged_cfg,
        load_step=int(merged_cfg.load_step),
        eval_test=bool(cfg.eval_test),
        test_from_train=bool(cfg.test_from_train),
    )


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Official FreeFix runner wrapper.")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config file.")
    parser.add_argument("--stage", type=str, default="full", choices=["recon", "refine", "eval", "full"])
    parser.add_argument("--freefix_root", type=str, default="/mntdatalora/src/Diffusion-IBR/works/FreeFix")
    parser.add_argument("--cache_root", type=str, default="/mntdatalora/src/Diffusion-IBR/cache_weights")
    parser.add_argument("--cuda_device", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--data_type", type=str, default="colmap", choices=["colmap", "hugsim"])
    parser.add_argument("--data_factor", type=int, default=4)
    parser.add_argument("--test_every", type=int, default=8)
    parser.add_argument("--partition", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--eval_steps", type=str, default="7000,30000")
    parser.add_argument("--save_steps", type=str, default="7000,30000")
    parser.add_argument(
        "--sync_exp_base_dir_with_result_dir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Override exp_cfg.base_dir with result_dir for refine/eval (recommended for stage=full).",
    )

    parser.add_argument("--backend", type=str, default="flux", choices=["flux", "sdxl"])
    parser.add_argument("--exp_cfg", type=str, default=None)
    parser.add_argument("--base_cfg", type=str, default=None)
    parser.add_argument("--eval_test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--test_from_train", action=argparse.BooleanOptionalAction, default=False)

    args_pre, _ = parser.parse_known_args()
    if args_pre.config is not None:
        with open(args_pre.config, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
        if not isinstance(cfg_data, dict):
            raise ValueError("Config JSON must be an object.")
        valid_fields = {f.name for f in fields(Config)}
        unknown_fields = sorted(str(k) for k in cfg_data.keys() if str(k) not in valid_fields)
        if unknown_fields:
            raise ValueError(f"Unknown config keys: {', '.join(unknown_fields)}")
        parser.set_defaults(**cfg_data)

    args = parser.parse_args()
    args_dict = vars(args)
    args_dict.pop("config", None)
    return Config(**args_dict)


def main() -> None:
    cfg = parse_args()
    _set_runtime_env(cfg)

    if cfg.stage in {"recon", "full"}:
        run_recon(cfg)
    if cfg.stage in {"refine", "full"}:
        run_refine(cfg)
    if cfg.stage in {"eval", "full"}:
        run_eval(cfg)


if __name__ == "__main__":
    main()
