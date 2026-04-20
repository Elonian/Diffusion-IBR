# Our Difix3D+ 3DGS Scripts

These scripts run the in-repo trainer (`scripts/trainers/trainer.py`) with the
Difix3D+ recipe and keep outputs isolated from comparison runs.

- Main script:
  - `run_ours_difix3dplus_dl3dv_scene.sh`
- Scene wrappers:
  - `run_ours_difix3dplus_dl3dv_scene_*.sh`

## Default behavior

- Resumes from the latest checkpoint in `outputs/ours_difix3dplus_gs/<scene>/ckpts` if available.
- Starts from scratch when no local checkpoint exists (`START_FROM_SCRATCH=1` by default).
- Stored comparison checkpoints are opt-in only via `ALLOW_REFERENCE_CKPT_FALLBACK=1`.
- Uses 4 data-loader workers by default (`NUM_WORKERS=4`)
- Writes to:
  - `outputs/ours_difix3dplus_gs/<scene>`
- Enforces reference-aligned Difix scheduling by default:
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
- `EVAL_EVERY`, `SAVE_EVERY` (defaults set to 10000 for reference-like cadence)
- `START_CKPT`: explicit checkpoint path
- `FORCE_FROM_30000=0`: prefer resuming latest checkpoint under our output directory
- `START_FROM_SCRATCH=1`: train from scratch when no local checkpoint exists
- `ALLOW_REFERENCE_CKPT_FALLBACK=1`: allow fallback to stored comparison checkpoints
- `ALLOW_LEGACY_VANILLA_FALLBACK=1`: allow fallback to the legacy baseline checkpoint when reference fallback is enabled
- `ALLOW_RUNTIME_DIFIX_OVERRIDES=1`: allow runtime schedule overrides
- `DIFIX_START_STEP`: override fixer start step (only used when `ALLOW_RUNTIME_DIFIX_OVERRIDES=1`)
- `DIFIX_FIX_EVERY`: override fixer interval (only used when `ALLOW_RUNTIME_DIFIX_OVERRIDES=1`)
- `FIX_STEPS`: override reference-style absolute fix step list (only used when `ALLOW_RUNTIME_DIFIX_OVERRIDES=1`)
