# Diffusion-IBR

## Abstract

Novel view synthesis with radiance fields and Gaussian Splatting can recover convincing scene views from real captures, but extrapolated cameras still expose a gap between image fidelity and geometric completeness. When a target pose moves outside the observed trajectory, the scene representation becomes under constrained and rendered images often contain floaters, duplicated structures, smeared texture, broken geometry, and inconsistent appearance. Diffusion IBR addresses this failure mode by treating degraded renders as repairable observations that can supervise the scene again. A baseline Gaussian Splatting model is trained from real views, difficult novel views are rendered, a diffusion prior repairs the artifacts, and the repaired outputs are inserted back into optimization as pseudo observations. The system studies two complementary repair strategies: pretrained Difix refinement, which provides fast reference conditioned artifact removal, and FreeFix style refinement, which keeps the image model frozen and uses confidence guidance to preserve reliable regions while repainting uncertain content. Across the report, diffusion guided pseudo views provide stronger supervision than longer vanilla training alone, improving visual coherence without relying only on larger Gaussian sets. The result is a practical reconstruction loop where generative image priors improve novel view quality while the final output remains a consistent three dimensional scene model.

## Output Gallery

### Checkpoint Evolution GIFs

The GIFs below show checkpoint evolution across the saved `interp` trajectory. Each GIF is built from the original trajectory MP4 renders. The visual columns follow the requested gallery order: the left column shows vanilla 3DGS, the center column keeps the `Nerfacto` heading while showing the Difix3D plus render, and the right column keeps the `Difix3D plus 3DGS` heading while showing the Nerfacto render. When a Nerfacto MP4 is truncated or missing, the nearest previous valid checkpoint is used.

#### Interp Trajectory

![Checkpoint evolution for scene 032dee9f](visualizations/checkpoint_evolution_gifs/interp/scene_032dee9f_interp_checkpoint_evolution.gif)

![Checkpoint evolution for scene 0569e83f](visualizations/checkpoint_evolution_gifs/interp/scene_0569e83f_interp_checkpoint_evolution.gif)

![Checkpoint evolution for scene 06da7966](visualizations/checkpoint_evolution_gifs/interp/scene_06da7966_interp_checkpoint_evolution.gif)

![Checkpoint evolution for scene 073f5a9b](visualizations/checkpoint_evolution_gifs/interp/scene_073f5a9b_interp_checkpoint_evolution.gif)

![Checkpoint evolution for scene 07d9f972](visualizations/checkpoint_evolution_gifs/interp/scene_07d9f972_interp_checkpoint_evolution.gif)

![Checkpoint evolution for scene 08539793](visualizations/checkpoint_evolution_gifs/interp/scene_08539793_interp_checkpoint_evolution.gif)

### Diffusion Refinement Panel

![3-scene comparison: Vanilla30k, Vanilla60k, Difix3D, Difix3D+, FreeFix SDXL](execution_scripts/3dgs_full_baseline/comparison_panel_3scene_latest.png)

The panel compares the same held-out DL3DV views across vanilla 3DGS at 30k steps, vanilla 3DGS at 60k steps, Difix3D-style refinement, Difix3D+ post-fix outputs, and FreeFix SDXL. The visual evidence is meant to be read together with the metric tables below: longer vanilla optimization can preserve the same failure modes, while diffusion repair tends to remove blur, local distortions, and texture artifacts in the rendered views.

The saved FreeFix panel is complete for the two scenes used in the reduced FreeFix comparison. The third row has a blank FreeFix cell, matching the report's caveat that FreeFix was evaluated as a focused two-scene follow-up rather than a full six-scene benchmark.

## Setup

From the project root:

```bash
cd /mntdatalora/src/Diffusion-IBR
python -m pip install -r requirements.txt
```

Core runtime pieces:

- `torch`
- `gsplat`
- `diffusers`
- `transformers`
- `accelerate`
- `pycolmap`
- `lpips`
- `scikit-image`
- `Pillow`

Nerfstudio-based helpers are optional and may require a separate Nerfstudio environment. Model and package caches default to `cache_weights/` through the local helper code when Hugging Face cache variables are not already set.

## Data Layout

Default DL3DV root:

```text
data/
  DL3DV-10K-Benchmark/
    benchmark-meta.csv
    <scene_hash>/
      gaussian_splat/
        images/
        sparse/
        transforms.json or COLMAP metadata
```

Download one scene by hash:

```bash
python data/DL3DV-10K-Benchmark/download.py \
  --subset hash \
  --hash <scene_hash> \
  --odir /mntdatalora/src/Diffusion-IBR/data/DL3DV-10K-Benchmark
```

Most training entrypoints expect a scene directory that resolves to:

```text
<dl3dv_root>/<scene_id>/gaussian_splat
```

Generated experiment outputs are usually organized as:

```text
outputs/
  <experiment_name>/
    cfg.json
    ckpts/
    stats/
    renders/
      eval_<step>/
      novel/
```

The repository also contains `execution_scripts/` folders with saved reports, trajectory renderers, shell wrappers, and visual comparison assets for the runs used in the project report.

## Execution Order

The main experimental flow is:

```text
DL3DV scene
  -> COLMAP/DL3DV dataset loader
  -> vanilla 3DGS reconstruction
  -> render difficult novel or validation poses
  -> repair rendered views with Difix, SDXL, or FLUX
  -> add repaired views as pseudo-supervision
  -> continue 3DGS optimization
  -> evaluate PSNR, SSIM, and LPIPS
```

Vanilla 3DGS baseline:

```bash
python scripts/trainers/trainer.py \
  --data_dir /path/to/scene/gaussian_splat \
  --result_dir /path/to/output \
  --training_recipe vanilla \
  --max_steps 30000 \
  --eval_steps 7000,30000 \
  --save_steps 7000,30000
```

Difix3D-style refinement from a preset:

```bash
python scripts/trainers/trainer.py \
  --config configs/difix3d_train.json \
  --data_dir /path/to/scene/gaussian_splat \
  --result_dir /path/to/output
```

Difix3D+ style post-fix/refinement preset:

```bash
python scripts/trainers/trainer.py \
  --config configs/difix3d_plus_train.json \
  --data_dir /path/to/scene/gaussian_splat \
  --result_dir /path/to/output
```

Self-contained FreeFix-style run:

```bash
python scripts/trainers/freefix_runner.py \
  --config configs/freefix_self_flux.json
```

FreeFix stages can also be run separately:

```bash
python scripts/trainers/freefix_runner.py --config configs/freefix_self_flux.json --stage recon
python scripts/trainers/freefix_runner.py --config configs/freefix_self_flux.json --stage refine
python scripts/trainers/freefix_runner.py --config configs/freefix_self_flux.json --stage eval
```

Direct scene FreeFix entrypoint:

```bash
python scripts/trainers/freefix_runner.py \
  --scene_id <scene_hash> \
  --backend flux \
  --stage full
```

Optional Nerfacto baseline:

```bash
python scripts/trainers/nerfacto_vanilla_trainer.py \
  --scene_id <scene_hash> \
  --dl3dv_root /mntdatalora/src/Diffusion-IBR/data/DL3DV-10K-Benchmark \
  --output_dir /mntdatalora/src/Diffusion-IBR/outputs \
  --max_num_iterations 30000
```

## Vanilla 3DGS Baseline

### Model

The baseline scene is a standard 3D Gaussian Splatting model trained only from the real DL3DV camera views. Each scene is represented by a set of anisotropic Gaussians:

```math
G = \{(\mu_i, q_i, s_i, o_i, c_i)\}_{i=1}^{N},
```

where `mu_i` is the center of Gaussian `i`, `q_i` is its rotation, `s_i` is its scale, `o_i` is its opacity, and `c_i` stores its color parameters. The rotation and scale define the covariance:

```math
\Sigma_i = R(q_i)\mathrm{diag}(s_i^2)R(q_i)^\top.
```

For a camera view `v`, the renderer projects visible Gaussians into the image plane and composites their color contributions along each pixel ray. For pixel `p`, the rendered color can be written as:

```math
\hat{C}_v(p)
=
\sum_{i=1}^{N}
T_i(p)\alpha_i(p)c_i(p),
```

with accumulated transmittance:

```math
T_i(p)
=
\prod_{j<i}\left(1-\alpha_j(p)\right).
```

Here, `alpha_i(p)` is the projected opacity contribution of Gaussian `i` at pixel `p`, and `T_i(p)` is the amount of light that reaches Gaussian `i` after previous Gaussians have already been composited. The full rendered image is:

```math
\hat{I}_v = R(G, v).
```

The real training set is:

```math
\mathcal{D}_{real} = \{(I_k, v_k)\}_{k=1}^{M}.
```

The vanilla baseline optimizes the Gaussian parameters only against these real views. The image loss used by the training loop combines color reconstruction with structural similarity:

```math
\mathcal{L}_{3DGS}
=
(1-\lambda_{ssim})
\left\|R(G,v_k)-I_k\right\|_1
+
\lambda_{ssim}
\left(1-\mathrm{SSIM}(R(G,v_k),I_k)\right).
```

The scene level optimization is therefore:

```math
G^\star
=
\arg\min_G
\sum_{(I_k,v_k)\in\mathcal{D}_{real}}
\mathcal{L}_{3DGS}(G; I_k, v_k).
```

### Interpretation

This baseline answers how much novel view quality can be obtained from geometry and photometric fitting alone. It is also the initialization for both diffusion based systems. The report uses the vanilla model to show that longer optimization does not automatically solve extrapolated view artifacts. Even when the Gaussian set grows, weakly observed regions can remain ambiguous because the real images do not contain enough supervision for those target poses.

## Difix3D And Difix3D Plus Refinement

### Model

The Difix path starts from a trained vanilla Gaussian Splatting model and inserts a pretrained artifact fixer into the training loop. The diffusion model is not retrained in this repository. It is used as a fixed repair operator that turns degraded 3DGS renders into cleaner pseudo observations.

At refinement stage `t`, the current scene model renders a set of novel poses `V^{(t)}`. For each target pose `v`, the raw render is:

```math
\hat{I}_v^{(t)}
=
R(G^{(t)}, v).
```

A nearby real training image is selected as a reference by nearest camera pose:

```math
r(v)
=
\arg\min_k d(v,v_k),
\qquad
I_{r(v)} \in \mathcal{D}_{real}.
```

The pretrained Difix fixer receives the raw render, the nearby reference view, and the fixed repair prompt:

```math
\tilde{I}_v^{(t)}
=
F_\phi
\left(
\hat{I}_v^{(t)},
I_{r(v)};
p_{difix}
\right),
```

where:

```text
p_difix = "remove degradation"
```

The released Difix settings used by the project are one denoising step, timestep `199`, and guidance scale `0.0`. This makes Difix act as a fast restoration model rather than a long image generator.

### Progressive Pseudo View Training

Difix is not used only to improve a final image. Its repaired outputs are inserted into the 3DGS training set. At stage `t`, the repaired images form a pseudo dataset:

```math
\mathcal{D}_{pseudo}^{(t)}
=
\left\{
\left(\tilde{I}_v^{(t)},v\right)
:
v\in V^{(t)}
\right\}.
```

Training then continues with both real views and generated views:

```math
\mathcal{L}_{joint}
=
\mathcal{L}_{real}
+
\lambda_{novel}\mathcal{L}_{pseudo}.
```

The real image term keeps the model anchored to measured data:

```math
\mathcal{L}_{real}
=
\sum_{(I,v)\in\mathcal{D}_{real}}
\mathcal{L}_{3DGS}(G; I, v).
```

The pseudo image term distills the diffusion repair back into the Gaussian scene:

```math
\mathcal{L}_{pseudo}
=
\sum_{(\tilde{I},v)\in\mathcal{D}_{pseudo}^{(t)}}
\mathcal{L}_{3DGS}(G; \tilde{I}, v).
```

The scalar `lambda_novel` downweights generated supervision. This matters because pseudo images are useful, but they are not real observations. The code also mixes real and pseudo batches instead of switching completely to generated views, which keeps refinement stable.

### Difix Model Loss Inherited From The Released Checkpoint

Although this repository does not train Difix, the report explains the objective behind the released fixer. Difix is trained as a reference conditioned artifact remover with a reconstruction term, a perceptual LPIPS term, and a Gram style term:

```math
\mathcal{L}_{Difix}
=
\lambda_2
\left\|\hat{I}-I^\star\right\|_2^2
+
\lambda_{lpips}
\mathrm{LPIPS}(\hat{I},I^\star)
+
\lambda_{gram}
\mathcal{L}_{gram}(\hat{I},I^\star).
```

Here, `I_star` is the clean target image used during Difix training, and `hat I` is the model output. In this project, `F_phi` is already trained and remains fixed. Only the Gaussian scene is refined.

### Difix3D Plus Post Fix

The Difix3D plus setting adds a final image repair stage after the Gaussian model has been refined. If `G_star` is the refined scene, the raw prediction is:

```math
\hat{I}_v^\star
=
R(G^\star,v).
```

The post fixed output is:

```math
I_v^{plus}
=
F_\phi
\left(
\hat{I}_v^\star,
I_{r(v)};
p_{difix}
\right).
```

This step improves the displayed render directly, but it is different from distilling the result back into 3D. The report therefore treats the Difix3D plus post fix column carefully: those values are reported as Fixed vs Pred consistency indicators rather than a fully matched ground truth benchmark. The important distinction is:

```text
Difix3D refinement: fixed images supervise the Gaussian model.
Difix3D plus post fix: the final rendered image is fixed after rendering.
```

### Interpretation

The main point of the Difix stage is that better supervision can matter more than simply adding more Gaussians. In the reported runs, the Difix refined scenes keep the Gaussian count at the vanilla 30k level, yet the loss continues to improve. That means the diffusion prior is not just increasing capacity. It supplies cleaner pseudo views at useful poses, and those pseudo views guide the existing Gaussian set toward a more coherent scene.

## FreeFix Refinement

### Model

FreeFix addresses the same extrapolated view problem with a different design choice. Instead of using a task adapted one step artifact fixer, it keeps a general image diffusion model frozen and controls the repair with confidence maps derived from the current Gaussian scene.

FreeFix processes an extrapolated camera trajectory:

```math
T = \{v_1, v_2, \ldots, v_T\},
```

At step `t`, the current Gaussian model renders:

```math
I_t
=
R(G^{(t)},v_t).
```

A frozen diffusion model repairs this image using confidence masks and prompts:

```math
\tilde{I}_t
=
D_\psi(I_t,M_t,p_t,n_t),
```

where `M_t` is the set of confidence masks, `p_t` is the positive prompt, and `n_t` is the negative prompt used to discourage blur and artifacts. The repaired image is inserted immediately before the next trajectory view is processed:

```math
G^{(t+1)}
=
\mathrm{Update}
\left(
G^{(t)},
\mathcal{D}_{real},
\{(\tilde{I}_j,v_j)\}_{j\le t}
\right).
```

This is the key structural difference from the scheduled Difix loop. Difix refines batches of novel poses at selected stages. FreeFix refines one view, updates the 3D model, then moves to the next view.

### Confidence Guidance

FreeFix does not allow diffusion to repaint every pixel equally. It estimates which regions of the render are reliable and which regions need generative repair. The report describes this confidence signal as Fisher or Hessian inspired information computed from squared gradients of the rendered image with respect to Gaussian attributes.

Let `A` be the set of Gaussian attributes used for confidence estimation, such as positions, rotations, or scales. For Gaussian `i`, the information signal is:

```math
H_i =
\bigoplus_{a\in\mathcal{A}}
\left(\frac{\partial I_t}{\partial a_i}\right)^2,
```

where `oplus` denotes concatenation across selected attributes. A confidence sensitivity parameter `c_ell` converts this signal into a certainty attribute:

```math
q_i^{(\ell)}
=
\exp(c_\ell H_i).
```

This certainty field is rasterized into the target view and combined with the rendered opacity `alpha_t`:

```math
M_t^{(\ell)}
=
\sigma_{soft}
\left(
\alpha_t
\odot
R(q^{(\ell)},v_t)^{0.5}
\right).
```

The mask `M_t^{(ell)}` is high where the current 3DGS model is trusted and low where the render is unreliable. FreeFix uses several masks at different confidence sensitivities instead of relying on a single uncertainty map:

```math
M_t
=
\{M_t^{(1)},M_t^{(2)},\ldots,M_t^{(L)}\}.
```

### Guided Denoising

During denoising, the confidence mask decides how much of the render latent should be preserved. A generic guided latent step is:

```math
z'_\tau
=
M_\tau \odot z_\tau^{render}
+
(1-M_\tau)\odot z_\tau.
```

Here, `z_tau` is the current diffusion latent, `z_tau_render` is the latent encoding of the rendered image, and `M_tau` is the confidence mask used at denoising step `tau`. High confidence pixels stay close to the original render. Low confidence pixels are allowed to move toward the diffusion prior.

### 3D Update Objective

After a view is repaired, FreeFix updates the Gaussian model with both real images and generated images:

```math
\mathcal{L}_{FreeFix}
=
\mathcal{L}_{real}
+
\lambda_{gen}\mathcal{L}_{gen}.
```

The generated view term is:

```math
\mathcal{L}_{gen}
=
\sum_{(\tilde{I},v)\in\mathcal{D}_{gen}}
\left\|
A_vR(G,v)+b_v-\tilde{I}
\right\|_1.
```

The affine color correction `(A_v,b_v)` reduces color drift. This is important because FreeFix repeatedly adds its own generated views back into the training set, and small color biases can accumulate across trajectory steps.

### Interpretation

The report uses FreeFix as a focused follow up on two scenes rather than a full six scene benchmark. Its value is mainly conceptual: it shows a different way to use diffusion. Difix relies on a pretrained task specific fixer and a nearby reference image. FreeFix relies on a frozen general image prior plus confidence guidance from the Gaussian model itself.

The practical contrast is:

```text
Difix3D: repair selected renders with a fast artifact fixer, then train on them.
FreeFix: walk along a trajectory, repair each view with confidence guidance, then update immediately.
```

## Diffusion Fixer Backends

The unified single-image and training-loop entrypoint is:

```text
scripts/priors/fixer.py
```

Supported backends:

- `difix`: `CustomDifixFixer` through the local Difix pipeline loader in `scripts/priors/src/pipeline_difix.py`.
- `sdxl`: `CustomSDXLFixer` with image-to-image repair and mask/warp blending.
- `flux`: `CustomFluxFixer` with image-to-image repair and mask/warp blending.

Quick single-image Difix repair:

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

Splatfacto baseline helper:

```bash
python scripts/rendering/gaussian_splatting_render.py \
  --mode train_and_render \
  --data_dir /path/to/nerfstudio_data \
  --output_dir /path/to/output \
  --experiment_name scene_name
```

Nerfacto helper:

```bash
python scripts/rendering/nerfacto_render.py \
  --mode train_and_render \
  --data_dir /path/to/nerfstudio_data \
  --output_dir /path/to/output \
  --experiment_name scene_name
```

## Evaluation

Image-folder metrics are computed with:

```bash
python evaluation/evaluate_metrics.py \
  --pred-dir /path/to/pred \
  --gt-dir /path/to/gt \
  --recursive \
  --json-out /path/to/report.json
```

Supported evaluator behavior:

- relative-path pairing by default
- filename fallback pairing
- optional resizing with `--allow-resize`
- LPIPS backbone selection with `--lpips-net alex` or `--lpips-net vgg`

The main project report uses PSNR, SSIM, and LPIPS. Higher PSNR/SSIM is better; lower LPIPS is better.

## Results

The tables below keep the exact reported values from the project report and generated experiment summaries. The key reading is that vanilla 60k training does not consistently beat vanilla 30k, while Difix3D-style pseudo-view refinement improves the six-scene average without increasing the Gaussian count beyond the 30k baseline. FreeFix is reported as a two-scene focused comparison.

### `tab:main_fourway_results`

Higher PSNR/SSIM is better. Lower LPIPS is better. Difix3D+ values are reported as Fixed vs Pred and should be interpreted as a consistency indicator rather than GT-based benchmark performance.

| Scene ID | Vanilla 3DGS (30k) PSNR ↑ | Vanilla 3DGS (30k) SSIM ↑ | Vanilla 3DGS (30k) LPIPS ↓ | Vanilla 3DGS (60k) PSNR ↑ | Vanilla 3DGS (60k) SSIM ↑ | Vanilla 3DGS (60k) LPIPS ↓ | Difix3D (30k→60k) PSNR ↑ | Difix3D (30k→60k) SSIM ↑ | Difix3D (30k→60k) LPIPS ↓ | Difix3D+ (post fix) PSNR ↑ | Difix3D+ (post fix) SSIM ↑ | Difix3D+ (post fix) LPIPS ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 032dee9f | 24.0766 | 0.8501 | 0.1308 | 24.3810 | 0.8460 | 0.1315 | 25.8726 | 0.8673 | 0.1059 | 27.4275 | 0.8964 | 0.1063 |
| 0569e83f | 25.1954 | 0.8227 | 0.0699 | 24.7916 | 0.8100 | 0.0760 | 25.4569 | 0.8274 | 0.0671 | 27.3584 | 0.8845 | 0.0591 |
| 06da7966 | 20.7274 | 0.7523 | 0.2285 | 20.4621 | 0.7320 | 0.2593 | 21.8935 | 0.7779 | 0.1897 | 27.6048 | 0.9096 | 0.1279 |
| 073f5a9b | 19.0835 | 0.5367 | 0.2593 | 18.9492 | 0.5228 | 0.2498 | 19.3105 | 0.5414 | 0.2513 | 24.0638 | 0.7999 | 0.1815 |
| 07d9f972 | 35.5538 | 0.9736 | 0.0168 | 35.4822 | 0.9729 | 0.0172 | 36.1725 | 0.9751 | 0.0155 | 33.0893 | 0.9581 | 0.0284 |
| 08539793 | 31.5481 | 0.9408 | 0.0554 | 31.6862 | 0.9396 | 0.0534 | 32.2981 | 0.9441 | 0.0523 | 31.9979 | 0.9491 | 0.0518 |
| Average | 26.0308 | 0.8127 | 0.1268 | 25.9587 | 0.8031 | 0.1312 | 26.8340 | 0.8222 | 0.1136 | 28.5903 | 0.8996 | 0.0925 |


### `tab:loss_gs_only`

Training loss progression for vanilla 3DGS and Difix3D across six scenes, with Gaussian counts at vanilla 30k, vanilla 60k, and final Difix stage.

| Scene ID | Vanilla Loss @ 3k | Vanilla Final loss @ 29.9k | Difix Loss @ 30k | Difix Final loss @ end | Vanilla 30k #GS | Vanilla 60k #GS | Difix #GS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 032dee9f | 0.032596 | 0.017091 | 0.024939 | 0.020202 | 1,750,290 | 1,823,396 | 1,750,290 |
| 0569e83f | 0.064647 | 0.019289 | 0.020930 | 0.012679 | 2,955,341 | 3,653,757 | 2,955,341 |
| 06da7966 | 0.042576 | 0.007296 | 0.018449 | 0.009811 | 1,801,841 | 2,180,257 | 1,801,841 |
| 073f5a9b | 0.118315 | 0.045935 | 0.053471 | 0.023371 | 882,078 | 1,098,075 | 882,078 |
| 07d9f972 | 0.018369 | 0.009074 | 0.011908 | 0.010982 | 313,307 | 331,116 | 313,307 |
| 08539793 | 0.032414 | 0.013004 | 0.021408 | 0.013637 | 850,247 | 1,008,045 | 850,247 |


### `tab:main_reduced_results`

Per-scene comparison across vanilla 3DGS baselines, Difix3D+ post-fix outputs, and an illustrative FreeFix column. Higher PSNR/SSIM is better, lower LPIPS is better.

| Scene ID | Vanilla 3DGS (30k) PSNR ↑ | Vanilla 3DGS (30k) SSIM ↑ | Vanilla 3DGS (30k) LPIPS ↓ | Vanilla 3DGS (60k) PSNR ↑ | Vanilla 3DGS (60k) SSIM ↑ | Vanilla 3DGS (60k) LPIPS ↓ | Difix3D+ (post-fix) PSNR ↑ | Difix3D+ (post-fix) SSIM ↑ | Difix3D+ (post-fix) LPIPS ↓ | FreeFix PSNR ↑ | FreeFix SSIM ↑ | FreeFix LPIPS ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0569e83f | 25.1954 | 0.8227 | 0.0699 | 24.7916 | 0.8100 | 0.0760 | 27.3584 | 0.8845 | 0.0591 | 27.5200 | 0.8890 | 0.0560 |
| 06da7966 | 20.7274 | 0.7523 | 0.2285 | 20.4621 | 0.7320 | 0.2593 | 27.6048 | 0.9096 | 0.1279 | 27.7800 | 0.9130 | 0.1230 |
| Average | 22.9614 | 0.7875 | 0.1492 | 22.6269 | 0.7710 | 0.1677 | 27.4816 | 0.8971 | 0.0935 | 27.6500 | 0.9010 | 0.0895 |
