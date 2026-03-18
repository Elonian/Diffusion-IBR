# 3DGS Full Baseline Re-Run Report (6 scenes)

- Generated (UTC): 2026-03-18T03:18:14.997567Z
- Runner script: `/mntdatalora/src/Diffusion-IBR/execution_scripts/3dgs_full_baseline/run_official_3dgs_full_baseline_dl3dv_scene.sh`
- Command env: `INSTALL_DEPS=0 INSTALL_BUILD_DEPS=0 SKIP_CUDA_PREFLIGHT=1`
- Note: Existing 60k checkpoints were reused when present.

## Summary

| scene | exit | mode | ckpt_59999 | val_59999 | traj_59999 | PSNR@59999 | SSIM@59999 | LPIPS@59999 | num_GS@59999 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7 | 0 | skip-existing | 1 | 1 | 1 | 24.381027 | 0.845987 | 0.131507 | 1823396 |
| 0569e83fdc248a51fc0ab082ce5e2baff15755c53c207f545e6d02d91f01d166 | 0 | skip-existing | 1 | 1 | 1 | 24.791637 | 0.810033 | 0.075979 | 3653757 |
| 06da796666297fe4c683c231edf56ec00148a6a52ab5bb159fe1be31f53a58df | 0 | skip-existing | 1 | 1 | 1 | 20.462080 | 0.731991 | 0.259258 | 2180257 |
| 073f5a9b983ced6fb28b23051260558b165f328a16b2d33fe20585b7ee4ad561 | 0 | skip-existing | 1 | 1 | 1 | 18.949219 | 0.522764 | 0.249825 | 1098075 |
| 07d9f9724ca854fae07cb4c57d7ea22bf667d5decd4058f547728922f909956b | 0 | skip-existing | 1 | 1 | 1 | 35.482227 | 0.972853 | 0.017166 | 331116 |
| 0853979305f7ecb80bd8fc2c8df916410d471ef04ed5f1a64e9651baa41d7695 | 0 | skip-existing | 1 | 1 | 1 | 31.686201 | 0.939587 | 0.053411 | 1008045 |

## Per-Scene Run Logs

### 032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7
- exit_code: 0
- mode: skip-existing
- log: `/mntdatalora/src/Diffusion-IBR/logs/execution/3dgs_full_baseline/032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7/20260318-031728_var-pod-diffusion-0_run_official_3dgs_full_baseline_dl3dv_scene.log`
- output tail:
```text
[env] CC=/usr/bin/gcc
[env] CXX=/usr/bin/g++

[1/1] Skipping training because final baseline checkpoint already exists:
      /mntdatalora/src/Diffusion-IBR/outputs/official_3dgs_full_baseline/032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7/ckpts/ckpt_59999_rank0.pt

[done] Baseline checkpoint ready:
      /mntdatalora/src/Diffusion-IBR/outputs/official_3dgs_full_baseline/032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7/ckpts/ckpt_59999_rank0.pt
```

### 0569e83fdc248a51fc0ab082ce5e2baff15755c53c207f545e6d02d91f01d166
- exit_code: 0
- mode: skip-existing
- log: `/mntdatalora/src/Diffusion-IBR/logs/execution/3dgs_full_baseline/0569e83fdc248a51fc0ab082ce5e2baff15755c53c207f545e6d02d91f01d166/20260318-031755_var-pod-diffusion-0_run_official_3dgs_full_baseline_dl3dv_scene.log`
- output tail:
```text
[env] CC=/usr/bin/gcc
[env] CXX=/usr/bin/g++

[1/1] Skipping training because final baseline checkpoint already exists:
      /mntdatalora/src/Diffusion-IBR/outputs/official_3dgs_full_baseline/0569e83fdc248a51fc0ab082ce5e2baff15755c53c207f545e6d02d91f01d166/ckpts/ckpt_59999_rank0.pt

[done] Baseline checkpoint ready:
      /mntdatalora/src/Diffusion-IBR/outputs/official_3dgs_full_baseline/0569e83fdc248a51fc0ab082ce5e2baff15755c53c207f545e6d02d91f01d166/ckpts/ckpt_59999_rank0.pt
```

### 06da796666297fe4c683c231edf56ec00148a6a52ab5bb159fe1be31f53a58df
- exit_code: 0
- mode: skip-existing
- log: `/mntdatalora/src/Diffusion-IBR/logs/execution/3dgs_full_baseline/06da796666297fe4c683c231edf56ec00148a6a52ab5bb159fe1be31f53a58df/20260318-031759_var-pod-diffusion-0_run_official_3dgs_full_baseline_dl3dv_scene.log`
- output tail:
```text
[env] CC=/usr/bin/gcc
[env] CXX=/usr/bin/g++

[1/1] Skipping training because final baseline checkpoint already exists:
      /mntdatalora/src/Diffusion-IBR/outputs/official_3dgs_full_baseline/06da796666297fe4c683c231edf56ec00148a6a52ab5bb159fe1be31f53a58df/ckpts/ckpt_59999_rank0.pt

[done] Baseline checkpoint ready:
      /mntdatalora/src/Diffusion-IBR/outputs/official_3dgs_full_baseline/06da796666297fe4c683c231edf56ec00148a6a52ab5bb159fe1be31f53a58df/ckpts/ckpt_59999_rank0.pt
```

### 073f5a9b983ced6fb28b23051260558b165f328a16b2d33fe20585b7ee4ad561
- exit_code: 0
- mode: skip-existing
- log: `/mntdatalora/src/Diffusion-IBR/logs/execution/3dgs_full_baseline/073f5a9b983ced6fb28b23051260558b165f328a16b2d33fe20585b7ee4ad561/20260318-031803_var-pod-diffusion-0_run_official_3dgs_full_baseline_dl3dv_scene.log`
- output tail:
```text
[env] CC=/usr/bin/gcc
[env] CXX=/usr/bin/g++

[1/1] Skipping training because final baseline checkpoint already exists:
      /mntdatalora/src/Diffusion-IBR/outputs/official_3dgs_full_baseline/073f5a9b983ced6fb28b23051260558b165f328a16b2d33fe20585b7ee4ad561/ckpts/ckpt_59999_rank0.pt

[done] Baseline checkpoint ready:
      /mntdatalora/src/Diffusion-IBR/outputs/official_3dgs_full_baseline/073f5a9b983ced6fb28b23051260558b165f328a16b2d33fe20585b7ee4ad561/ckpts/ckpt_59999_rank0.pt
```

### 07d9f9724ca854fae07cb4c57d7ea22bf667d5decd4058f547728922f909956b
- exit_code: 0
- mode: skip-existing
- log: `/mntdatalora/src/Diffusion-IBR/logs/execution/3dgs_full_baseline/07d9f9724ca854fae07cb4c57d7ea22bf667d5decd4058f547728922f909956b/20260318-031806_var-pod-diffusion-0_run_official_3dgs_full_baseline_dl3dv_scene.log`
- output tail:
```text
[env] CC=/usr/bin/gcc
[env] CXX=/usr/bin/g++

[1/1] Skipping training because final baseline checkpoint already exists:
      /mntdatalora/src/Diffusion-IBR/outputs/official_3dgs_full_baseline/07d9f9724ca854fae07cb4c57d7ea22bf667d5decd4058f547728922f909956b/ckpts/ckpt_59999_rank0.pt

[done] Baseline checkpoint ready:
      /mntdatalora/src/Diffusion-IBR/outputs/official_3dgs_full_baseline/07d9f9724ca854fae07cb4c57d7ea22bf667d5decd4058f547728922f909956b/ckpts/ckpt_59999_rank0.pt
```

### 0853979305f7ecb80bd8fc2c8df916410d471ef04ed5f1a64e9651baa41d7695
- exit_code: 0
- mode: skip-existing
- log: `/mntdatalora/src/Diffusion-IBR/logs/execution/3dgs_full_baseline/0853979305f7ecb80bd8fc2c8df916410d471ef04ed5f1a64e9651baa41d7695/20260318-031810_var-pod-diffusion-0_run_official_3dgs_full_baseline_dl3dv_scene.log`
- output tail:
```text
[env] CC=/usr/bin/gcc
[env] CXX=/usr/bin/g++

[1/1] Skipping training because final baseline checkpoint already exists:
      /mntdatalora/src/Diffusion-IBR/outputs/official_3dgs_full_baseline/0853979305f7ecb80bd8fc2c8df916410d471ef04ed5f1a64e9651baa41d7695/ckpts/ckpt_59999_rank0.pt

[done] Baseline checkpoint ready:
      /mntdatalora/src/Diffusion-IBR/outputs/official_3dgs_full_baseline/0853979305f7ecb80bd8fc2c8df916410d471ef04ed5f1a64e9651baa41d7695/ckpts/ckpt_59999_rank0.pt
```

