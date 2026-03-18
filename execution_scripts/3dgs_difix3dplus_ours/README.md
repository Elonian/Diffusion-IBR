# Our Difix3D+ 3DGS Scripts

These scripts run the in-repo trainer (`scripts/trainers/trainer.py`) with the
Difix3D+ recipe and keep outputs isolated from official runs.

- Main script:
  - `run_ours_difix3dplus_dl3dv_scene.sh`
- Scene wrappers:
  - `run_ours_difix3dplus_dl3dv_scene_*.sh`

## Default behavior

- Resumes from 30k vanilla checkpoint (step `29999`) if available:
  - `outputs/official_difix3d/vanilla_gs/<scene>/ckpts/ckpt_29999_rank0.pt`
  - legacy fallback is disabled by default and can be enabled with `ALLOW_LEGACY_VANILLA_FALLBACK=1`
- Uses 4 data-loader workers by default (`NUM_WORKERS=4`)
- Writes to:
  - `outputs/ours_difix3dplus_gs/<scene>`
- Enforces official-aligned Difix scheduling by default:
  - `STRICT_OFFICIAL_DIFIX=1`
  - `ALLOW_RUNTIME_DIFIX_OVERRIDES=0`

## Example

```bash
CUDA_DEVICE=0 MAX_STEPS=60000 \
bash /mntdatalora/src/Diffusion-IBR/execution_scripts/3dgs_difix3dplus_ours/run_ours_difix3dplus_dl3dv_scene_0569e83f.sh
```

## Useful env overrides

- `OUTPUT_ROOT`: custom output root (default `outputs/ours_difix3dplus_gs`)
- `MAX_STEPS`: final training step
- `EVAL_EVERY`, `SAVE_EVERY` (defaults set to 10000 for official-like cadence)
- `START_CKPT`: explicit checkpoint path
- `FORCE_FROM_30000=0`: prefer resuming latest checkpoint under our output directory
- `ALLOW_LEGACY_VANILLA_FALLBACK=1`: allow fallback to `official_3dgs_full_baseline` ckpt
- `ALLOW_RUNTIME_DIFIX_OVERRIDES=1`: allow runtime schedule overrides
- `DIFIX_START_STEP`: override fixer start step (only used when `ALLOW_RUNTIME_DIFIX_OVERRIDES=1`)
- `DIFIX_FIX_EVERY`: override fixer interval (only used when `ALLOW_RUNTIME_DIFIX_OVERRIDES=1`)
- `FIX_STEPS`: override official-style absolute fix step list (only used when `ALLOW_RUNTIME_DIFIX_OVERRIDES=1`)
