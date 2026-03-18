# Diffusion-IBR Trainer Configs

These JSON files are loaded with `--config` in `scripts/trainers/trainer.py`.

## Available presets

- `difix3d_train.json`: DIFIX3D training with progressive pose updates and no post-render enhancer.
- `difix3d_plus_train.json`: DIFIX3D+ training schedule with progressive pose updates and post-render enhancer disabled by default.
- `freefix_self_flux.json`: Self-sufficient FreeFix pipeline preset using FLUX backend.
- `freefix_self_sdxl.json`: Self-sufficient FreeFix pipeline preset using SDXL backend.

## Example usage

```bash
python /mntdatalora/src/Diffusion-IBR/scripts/trainers/trainer.py \
  --config /mntdatalora/src/Diffusion-IBR/configs/difix3d_plus_train.json \
  --data_dir /path/to/scene \
  --result_dir /path/to/output
```

Override any field directly on CLI:

```bash
python /mntdatalora/src/Diffusion-IBR/scripts/trainers/trainer.py \
  --config /mntdatalora/src/Diffusion-IBR/configs/difix3d_plus_train.json \
  --data_dir /path/to/scene \
  --result_dir /path/to/output \
  --max_steps 15000 \
  --difix_fix_every 1000
```

## Self-Sufficient FreeFix Pipeline

Run reconstruction + refinement + evaluation through the local self-sufficient runner:

```bash
python /mntdatalora/src/Diffusion-IBR/scripts/trainers/freefix_runner.py \
  --config /mntdatalora/src/Diffusion-IBR/configs/freefix_self_flux.json
```

Equivalent shell entrypoint:

```bash
bash /mntdatalora/src/Diffusion-IBR/execution_scripts/freefix/run_freefix_self_dl3dv_scene_06da7966.sh
```

You can also run stages separately:

```bash
python /mntdatalora/src/Diffusion-IBR/scripts/trainers/freefix_runner.py \
  --stage recon \
  --scene_id <scene_id> \
  --backend sdxl

python /mntdatalora/src/Diffusion-IBR/scripts/trainers/freefix_runner.py \
  --stage refine \
  --scene_id <scene_id> \
  --backend sdxl
```
