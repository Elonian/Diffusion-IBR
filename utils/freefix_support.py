"""
Shared helpers for generating FreeFix scene assets (partition + exp cfg).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_PROMPT = (
    "A photorealistic real-world scene with consistent geometry, detailed textures, "
    "and natural lighting."
)
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, foggy, overall gray, subtitles, incomplete, ghost image, "
    "too close to camera"
)


@dataclass
class FreeFixSceneAssets:
    scene_id: str
    backend: str
    scene_data_dir: str
    output_dir: str
    num_images: int
    num_train: int
    num_test: int
    partition_path: str
    exp_cfg_path: str
    metadata_path: str
    prompt: str
    negative_prompt: str
    test_every: int
    split_mode: str


def _yaml_quote(value: str) -> str:
    return json.dumps(value)


def _normalize_backend(backend: str) -> str:
    value = str(backend).strip().lower()
    if value not in {"flux", "sdxl"}:
        raise ValueError(f"backend must be 'flux' or 'sdxl', got: {backend}")
    return value


def _normalize_split_mode(split_mode: str) -> str:
    value = str(split_mode).strip().lower()
    if value not in {"freefix", "difix3d"}:
        raise ValueError(f"split_mode must be 'freefix' or 'difix3d', got: {split_mode}")
    return value


def _scene_images(scene_data_dir: Path) -> list[Path]:
    image_dir = scene_data_dir / "images"
    if not image_dir.is_dir():
        raise ValueError(f"Missing COLMAP image directory: {image_dir}")
    image_paths = sorted(path for path in image_dir.rglob("*") if path.is_file())
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {image_dir}")
    return image_paths


def build_partition_indices(
    num_images: int,
    test_every: int,
    split_mode: str = "freefix",
) -> tuple[list[int], list[int]]:
    split_mode = _normalize_split_mode(split_mode)
    if test_every <= 0:
        raise ValueError(f"test_every must be > 0, got {test_every}")
    if split_mode == "difix3d":
        train_indices = [idx for idx in range(num_images) if idx % test_every == 0]
        test_indices = [idx for idx in range(num_images) if idx % test_every != 0]
    else:
        test_indices = [idx for idx in range(num_images) if idx % test_every == 0]
        train_indices = [idx for idx in range(num_images) if idx % test_every != 0]
    if len(train_indices) == 0 or len(test_indices) == 0:
        raise ValueError(
            f"Invalid split: num_images={num_images}, test_every={test_every}, "
            f"train={len(train_indices)}, test={len(test_indices)}"
        )
    return train_indices, test_indices


def write_partition_file(
    *,
    scene_id: str,
    scene_data_dir: Path,
    split_mode: str,
    test_every: int,
    num_images: int,
    train_indices: Iterable[int],
    test_indices: Iterable[int],
) -> Path:
    split_mode = _normalize_split_mode(split_mode)
    if split_mode == "difix3d":
        partition_name = f"difix3d_testevery{test_every}.json"
    else:
        partition_name = f"freefix_testevery{test_every}.json"
    partition_relpath = Path("partitions") / partition_name
    partition_path = scene_data_dir / partition_relpath
    partition_path.parent.mkdir(parents=True, exist_ok=True)
    partition = {
        "scene_id": scene_id,
        "split_mode": split_mode,
        "test_every": int(test_every),
        "num_images": int(num_images),
        "train": [int(i) for i in train_indices],
        "test": [int(i) for i in test_indices],
    }
    partition_path.write_text(json.dumps(partition, indent=2) + "\n", encoding="utf-8")
    return partition_path


def _backend_defaults(backend: str) -> tuple[str, float, list[str]]:
    # Keep defaults close to the released FreeFix MipNeRF examples.
    if backend == "flux":
        return "flux", 0.6, ["means", "quats", "scales"]
    return "sdxl", 0.6, ["means", "quats", "scales"]


def build_exp_cfg_text(
    *,
    output_dir: Path,
    backend: str,
    prompt: str,
    negative_prompt: str,
    num_train: int,
    num_test: int,
    strength: Optional[float] = None,
    hessian_attr: Optional[list[str]] = None,
    num_inference_steps: int = 50,
    guide_ratio: float = 1.0,
    warp_ratio: float = 0.5,
    refine_steps: int = 400,
    gen_prob: float = 0.1,
    gen_loss_weight: float = 0.2,
    affine: bool = True,
    load_step: int = 29999,
    data_type: str = "colmap",
) -> str:
    exp_name, default_strength, default_hessian_attr = _backend_defaults(backend)
    effective_strength = default_strength if strength is None else float(strength)
    effective_hessian_attr = default_hessian_attr if hessian_attr is None else list(hessian_attr)
    hessian_text = ", ".join(_yaml_quote(v) for v in effective_hessian_attr)

    return f"""base_dir: {_yaml_quote(str(output_dir))}
exp_name: {exp_name}
gs_cfg_file: "cfg.json"
refine_start_idx: 0
refine_end_idx: {int(num_test)}
train_start_idx: 0
train_end_idx: {int(num_train)}
test_split: test
test_trans: [ 0, 0, 0 ]
test_rots: [ 0, 0, 0 ]
prompt: {_yaml_quote(prompt)}
negative_prompt: {_yaml_quote(negative_prompt)}
strength: {effective_strength}
num_inference_steps: {int(num_inference_steps)}
guide_ratio: {float(guide_ratio)}
warp_ratio: {float(warp_ratio)}
refine_steps: {int(refine_steps)}
hessian_attr: [ {hessian_text} ]
c_exp_index: [ 0.001, 0.01, 0.1 ]
c_scheduler: [ 0.3, 0.9, 1.0 ]
gen_prob: {float(gen_prob)}
gen_loss_weight: {float(gen_loss_weight)}
affine: {str(bool(affine))}
load_step: {int(load_step)}
data_type: {data_type}
"""


def generate_freefix_scene_assets(
    *,
    scene_id: str,
    scene_data_dir: str | Path,
    output_dir: str | Path,
    backend: str,
    split_mode: str = "freefix",
    test_every: int = 8,
    prompt: str = DEFAULT_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    strength: Optional[float] = None,
    hessian_attr: Optional[list[str]] = None,
    num_inference_steps: int = 50,
    guide_ratio: float = 1.0,
    warp_ratio: float = 0.5,
    refine_steps: int = 400,
    gen_prob: float = 0.1,
    gen_loss_weight: float = 0.2,
    affine: bool = True,
    load_step: int = 29999,
    data_type: str = "colmap",
) -> FreeFixSceneAssets:
    normalized_backend = _normalize_backend(backend)
    normalized_split_mode = _normalize_split_mode(split_mode)
    scene_data = Path(scene_data_dir).resolve()
    out_dir = Path(output_dir).resolve()
    image_paths = _scene_images(scene_data)
    num_images = len(image_paths)
    train_indices, test_indices = build_partition_indices(
        num_images,
        int(test_every),
        split_mode=normalized_split_mode,
    )

    partition_path = write_partition_file(
        scene_id=scene_id,
        scene_data_dir=scene_data,
        split_mode=normalized_split_mode,
        test_every=int(test_every),
        num_images=num_images,
        train_indices=train_indices,
        test_indices=test_indices,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = out_dir / "exp_cfg"
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_cfg_path = exp_dir / f"freefix_{normalized_backend}.yaml"
    exp_cfg_text = build_exp_cfg_text(
        output_dir=out_dir,
        backend=normalized_backend,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_train=len(train_indices),
        num_test=len(test_indices),
        strength=strength,
        hessian_attr=hessian_attr,
        num_inference_steps=num_inference_steps,
        guide_ratio=guide_ratio,
        warp_ratio=warp_ratio,
        refine_steps=refine_steps,
        gen_prob=gen_prob,
        gen_loss_weight=gen_loss_weight,
        affine=affine,
        load_step=load_step,
        data_type=data_type,
    )
    exp_cfg_path.write_text(exp_cfg_text, encoding="utf-8")

    metadata_path = out_dir / "freefix_scene_assets.json"
    assets = FreeFixSceneAssets(
        scene_id=scene_id,
        backend=normalized_backend,
        scene_data_dir=str(scene_data),
        output_dir=str(out_dir),
        num_images=num_images,
        num_train=len(train_indices),
        num_test=len(test_indices),
        partition_path=str(partition_path),
        exp_cfg_path=str(exp_cfg_path),
        metadata_path=str(metadata_path),
        prompt=prompt,
        negative_prompt=negative_prompt,
        test_every=int(test_every),
        split_mode=normalized_split_mode,
    )
    metadata_path.write_text(json.dumps(asdict(assets), indent=2) + "\n", encoding="utf-8")
    return assets
