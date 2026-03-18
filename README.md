# Diffusion-IBR

Diffusion-IBR is a research workspace for iterative image-based refinement on top of radiance field / Gaussian Splatting training loops.

At the repository level, the main production path is:

1. Train a base 3DGS model.
2. Periodically render novel/test views.
3. Fix those views with diffusion backends (Difix3D, SDXL, or FLUX).
4. Feed fixed views back into training.
5. Evaluate with PSNR/SSIM/LPIPS.

## What Is In Scope For This README

This README focuses on the Diffusion-IBR project-owned orchestration and training code:

- `scripts/trainers/*.py`
- `scripts/priors/*.py`
- `scripts/rendering/*.py`
- `utils/*.py`
- `evaluation/evaluate_metrics.py`
- `data/DL3DV-10K-Benchmark/download.py`


Third-party vendored/cached Python code (especially huge trees in `cache_weights/` ) is treated as dependency/runtime payload, not the primary maintained surface.

## Repository Layout

- `scripts/trainers/trainer.py`: core standalone 3DGS trainer with `vanilla`, `freefix`, and `difix3d` recipes.
- `scripts/trainers/freefix_runner.py`: self-contained FreeFix-style recon/refine/eval orchestrator built on `trainer.py`.
- `scripts/trainers/nerfacto_vanilla_trainer.py`: standalone Nerfacto baseline trainer.
- `scripts/priors/fixer.py`: unified diffusion fixer entrypoint (`difix`, `sdxl`, `flux`).
- `scripts/priors/difix.py`: DifixPipeline wrapper.
- `scripts/priors/sdxl.py`, `scripts/priors/flux.py`: SDXL/FLUX img2img wrappers with mask+warp blending.
- `scripts/rendering/3dgs_render.py`: standalone splatfacto train/render helper.
- `scripts/rendering/nerfactor_render.py`: nerfacto train/render helper.
- `utils/data_colmap.py`, `utils/data_dataset.py`: COLMAP parsing, splits, dataset batches.
- `utils/freefix_support.py`: partition + YAML asset generation for FreeFix workflows.
- `evaluation/evaluate_metrics.py`: PSNR/SSIM/LPIPS image-folder evaluator.
- `configs/*.json`: preset configs for Difix3D/Difix3D+/FreeFix runners.
- `execution_scripts/*`: shell wrappers for common official/ours experiments.

## Environment Setup

```bash
cd /mntdatalora/src/Diffusion-IBR
python -m pip install -r requirements.txt
```

Notes:

- `requirements.txt` installs the self-contained stack (PyTorch, gsplat, diffusers ecosystem, pycolmap).
- Nerfstudio and some official stacks are optional and may require separate setup.
- Caches default to `cache_weights/` (HF cache variables are set by helper code if not provided).

## Data Setup (DL3DV)

Expected default root:

- `/mntdatalora/src/Diffusion-IBR/data/DL3DV-10K-Benchmark`

Download helper:

```bash
python data/DL3DV-10K-Benchmark/download.py \
  --subset hash \
  --hash <scene_hash> \
  --odir /mntdatalora/src/Diffusion-IBR/data/DL3DV-10K-Benchmark
```

For training scripts, a scene usually resolves to:

- `<dl3dv_root>/<scene_id>/gaussian_splat`

with COLMAP artifacts and images inside.

## Main Training Workflows

### 1) Vanilla 3DGS (Standalone)

```bash
python scripts/trainers/trainer.py \
  --data_dir /path/to/scene/gaussian_splat \
  --result_dir /path/to/output \
  --training_recipe vanilla \
  --max_steps 30000 \
  --eval_steps 7000,30000 \
  --save_steps 7000,30000
```

### 2) Difix3D / Difix3D+ Style Training

Preset:

- `configs/difix3d_train.json`
- `configs/difix3d_plus_train.json`

Example:

```bash
python scripts/trainers/trainer.py \
  --config configs/difix3d_plus_train.json \
  --data_dir /path/to/scene/gaussian_splat \
  --result_dir /path/to/output
```

### 3) Self-Contained FreeFix Pipeline

Preset:

- `configs/freefix_self_sdxl.json`
- `configs/freefix_self_flux.json`

Full pipeline (recon + refine + eval):

```bash
python scripts/trainers/freefix_runner.py \
  --config configs/freefix_self_flux.json
```

Stages:

- `--stage recon`
- `--stage refine`
- `--stage eval`
- `--stage full`

### 4) FreeFix Runner (Direct Scene Args)

Example:

```bash
python scripts/trainers/freefix_runner.py \
  --scene_id <scene_hash> \
  --backend flux \
  --stage full
```

### 5) Nerfacto Baseline

```bash
python scripts/trainers/nerfacto_vanilla_trainer.py \
  --scene_id <scene_hash> \
  --dl3dv_root /mntdatalora/src/Diffusion-IBR/data/DL3DV-10K-Benchmark \
  --output_dir /mntdatalora/src/Diffusion-IBR/outputs \
  --max_num_iterations 30000
```

## Diffusion Fixer Backends

Unified wrapper: `scripts/priors/fixer.py`

- `difix`: `CustomDifixFixer` via `works/Difix3D/src/pipeline_difix.py`
- `sdxl`: `CustomSDXLFixer`
- `flux`: `CustomFluxFixer`

Quick single-image CLI:

```bash
python scripts/priors/fixer.py \
  --backend difix \
  --input_image in.png \
  --ref_image ref.png \
  --output_image out.png \
  --prompt "remove degradation" \
  --steps 1 \
  --timestep 199 \
  --guidance_scale 0.0
```

## Rendering Helpers

### Splatfacto baseline helper

```bash
python scripts/rendering/3dgs_render.py \
  --mode train_and_render \
  --data_dir /path/to/nerfstudio_data \
  --output_dir /path/to/output \
  --experiment_name scene_name
```

### Nerfacto helper

```bash
python scripts/rendering/nerfactor_render.py \
  --mode train_and_render \
  --data_dir /path/to/nerfstudio_data \
  --output_dir /path/to/output \
  --experiment_name scene_name
```

## Evaluation

```bash
python evaluation/evaluate_metrics.py \
  --pred-dir /path/to/pred \
  --gt-dir /path/to/gt \
  --recursive \
  --json-out /path/to/report.json
```

Supports:

- relative-path pairing (default)
- filename fallback pairing
- optional resizing (`--allow-resize`)
- LPIPS backbone choice (`alex` or `vgg`)

## Output Structure (Typical)

`trainer.py` writes:

- `result_dir/cfg.json`
- `result_dir/ckpts/ckpt_<step>_rank0.pt`
- `result_dir/stats/train_step<step>.json`
- `result_dir/stats/eval_step<step>.json`
- `result_dir/renders/eval_<step>/*_pred.png,*_gt.png`
- `result_dir/renders/novel/<step>/Pred|Ref|Fixed|Alpha|Mask/...` (when fixer runs)



