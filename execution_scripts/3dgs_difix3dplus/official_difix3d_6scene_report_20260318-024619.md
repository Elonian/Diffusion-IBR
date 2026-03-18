# Official Difix3D 6-Scene Metrics Report

- Generated (UTC): 2026-03-18T03:06:40.779169Z
- Source root: `/mntdatalora/src/Diffusion-IBR/outputs/official_difix3d`

## Metric Sources

- `vanilla_gs_30k` and `difix3d_official` metrics are read from official `stats/val_step*.json` files.
- `difix3dplus_fixed_vs_pred` is computed with existing `evaluation/evaluate_metrics.py` using `Fixed` vs `Pred` on the latest complete novel step.
- `difix3dplus_fixed_vs_pred` is **not GT-based** (reference is Pred), so treat it as a consistency/change indicator, not benchmark PSNR/SSIM/LPIPS against ground truth.

## Summary Table

| scene | vanilla (PSNR/SSIM/LPIPS) | difix3d (PSNR/SSIM/LPIPS) | difix3d+ fixed-vs-pred (PSNR/SSIM/LPIPS) | GS vanilla->difix | data train start->difix end | fix steps completed |
|---|---:|---:|---:|---:|---:|---:|
| 032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7 | 24.0766/0.8501/0.1308 | 25.8726/0.8673/0.1059 | 27.4275/0.8964/0.1063 | 1750290->1750290 | 40->4504 | 16 |
| 0569e83fdc248a51fc0ab082ce5e2baff15755c53c207f545e6d02d91f01d166 | 25.1954/0.8227/0.0699 | 25.4569/0.8274/0.0671 | 27.3584/0.8845/0.0591 | 2955341->2955341 | 39->4327 | 16 |
| 06da796666297fe4c683c231edf56ec00148a6a52ab5bb159fe1be31f53a58df | 20.7274/0.7523/0.2285 | 21.8935/0.7779/0.1897 | 27.6048/0.9096/0.1279 | 1801841->1801841 | 40->4488 | 16 |
| 073f5a9b983ced6fb28b23051260558b165f328a16b2d33fe20585b7ee4ad561 | 19.0835/0.5367/0.2593 | 19.3105/0.5414/0.2513 | 24.0638/0.7999/0.1815 | 882078->882078 | 52->5780 | 16 |
| 07d9f9724ca854fae07cb4c57d7ea22bf667d5decd4058f547728922f909956b | 35.5538/0.9736/0.0168 | 36.1725/0.9751/0.0155 | 33.0893/0.9581/0.0284 | 313307->313307 | 39->4359 | 16 |
| 0853979305f7ecb80bd8fc2c8df916410d471ef04ed5f1a64e9651baa41d7695 | 31.5481/0.9408/0.0554 | 32.2981/0.9441/0.0523 | 31.9979/0.9491/0.0518 | 850247->850247 | 45->4427 | 14 |

## Loss And Data Growth Details

| scene | vanilla loss@~3000 | vanilla loss@final | difix loss@start | difix loss@final | train start | after first fix | final effective train pool | total fixed images added |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7 | 0.032596 (step 3000) | 0.017091 (step 29900) | 0.024939 (step 30000) | 0.020202 (step 59900) | 40 | 319 | 4504 | 4464 |
| 0569e83fdc248a51fc0ab082ce5e2baff15755c53c207f545e6d02d91f01d166 | 0.064647 (step 3000) | 0.019289 (step 29900) | 0.020930 (step 30000) | 0.012679 (step 59900) | 39 | 307 | 4327 | 4288 |
| 06da796666297fe4c683c231edf56ec00148a6a52ab5bb159fe1be31f53a58df | 0.042576 (step 3000) | 0.007296 (step 29900) | 0.018449 (step 30000) | 0.009811 (step 59900) | 40 | 318 | 4488 | 4448 |
| 073f5a9b983ced6fb28b23051260558b165f328a16b2d33fe20585b7ee4ad561 | 0.118315 (step 3000) | 0.045935 (step 29900) | 0.053471 (step 30000) | 0.023371 (step 59900) | 52 | 410 | 5780 | 5728 |
| 07d9f9724ca854fae07cb4c57d7ea22bf667d5decd4058f547728922f909956b | 0.018369 (step 3000) | 0.009074 (step 29900) | 0.011908 (step 30000) | 0.010982 (step 59900) | 39 | 309 | 4359 | 4320 |
| 0853979305f7ecb80bd8fc2c8df916410d471ef04ed5f1a64e9651baa41d7695 | 0.032414 (step 3000) | 0.013004 (step 29900) | 0.021408 (step 30000) | 0.013637 (step 55000) | 45 | 358 | 4427 | 4382 |

## Files Written

- `/mntdatalora/src/Diffusion-IBR/execution_scripts/3dgs_difix3dplus/official_difix3d_6scene_metrics_20260318-024619.csv`
- `/mntdatalora/src/Diffusion-IBR/execution_scripts/3dgs_difix3dplus/official_difix3d_6scene_metrics_long_20260318-024619.csv`
- `/mntdatalora/src/Diffusion-IBR/execution_scripts/3dgs_difix3dplus/official_difix3d_6scene_report_20260318-024619.md`
- `/mntdatalora/src/Diffusion-IBR/execution_scripts/3dgs_difix3dplus/official_metrics_032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7.csv`
- `/mntdatalora/src/Diffusion-IBR/execution_scripts/3dgs_difix3dplus/official_metrics_0569e83fdc248a51fc0ab082ce5e2baff15755c53c207f545e6d02d91f01d166.csv`
- `/mntdatalora/src/Diffusion-IBR/execution_scripts/3dgs_difix3dplus/official_metrics_06da796666297fe4c683c231edf56ec00148a6a52ab5bb159fe1be31f53a58df.csv`
- `/mntdatalora/src/Diffusion-IBR/execution_scripts/3dgs_difix3dplus/official_metrics_073f5a9b983ced6fb28b23051260558b165f328a16b2d33fe20585b7ee4ad561.csv`
- `/mntdatalora/src/Diffusion-IBR/execution_scripts/3dgs_difix3dplus/official_metrics_07d9f9724ca854fae07cb4c57d7ea22bf667d5decd4058f547728922f909956b.csv`
- `/mntdatalora/src/Diffusion-IBR/execution_scripts/3dgs_difix3dplus/official_metrics_0853979305f7ecb80bd8fc2c8df916410d471ef04ed5f1a64e9651baa41d7695.csv`
