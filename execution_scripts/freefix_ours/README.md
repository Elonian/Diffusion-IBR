# Our FreeFix Scripts

These scripts run the in-repo self-sufficient FreeFix pipeline and keep
execution isolated from `works/*` runtime imports.

- Main script:
  - `run_ours_freefix_dl3dv_scene.sh`
  - `run_ours_freefix_flux_dl3dv_scene.sh`
- Scene wrappers:
  - `run_ours_freefix_dl3dv_scene_*.sh`
  - `run_ours_freefix_flux_dl3dv_scene_*.sh`

## Example

```bash
FREEFIX_BACKEND=sdxl FREEFIX_STAGE=full \
bash /mntdatalora/src/Diffusion-IBR/execution_scripts/freefix_ours/run_ours_freefix_dl3dv_scene_06da7966.sh
```

Flux fixed-backend wrapper:

```bash
FREEFIX_STAGE=full \
bash /mntdatalora/src/Diffusion-IBR/execution_scripts/freefix_ours/run_ours_freefix_flux_dl3dv_scene_06da7966.sh
```

## Useful env overrides

- `FREEFIX_STAGE`: `recon`, `refine`, `eval`, or `full`
- `FREEFIX_BACKEND`: `sdxl` (default) or `flux`
- `RECON_STEPS`: reconstruction steps (default `30000`)
- `REFINE_CYCLES`: number of fix-and-train cycles (default `1`)
- `REFINE_STEPS_PER_CYCLE`: train steps per cycle (default `400`)
- `REFINE_NUM_VIEWS`: number of test views to fix each cycle (`0` means all)
- `GEN_PROB`, `GEN_LOSS_WEIGHT`: generated-view sampling controls
- `OUTPUT_ROOT`: output root directory (default `outputs/freefix_self`)
- `DRY_RUN=1`: print generated commands/configs without running training
