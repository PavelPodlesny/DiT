# Uncertainty Pipeline Overview

## 1. Overview

For each class, `M` fixed-seed base trajectories are sampled, recording per-step noise. At each
configured branch point, the branch latent is perturbed `K` ways per perturbation type, then each
copy resumes denoising to completion **reusing the base's own recorded noise** (isolates the
perturbation's effect from sampling stochasticity). Spread across the resulting ensemble is
measured both pairwise-among-members and vs-base, per `(class, base, branch_point, type)`.

## 2. Perturbations

Applied to the branch-point latent tensor. Each
ensemble member uses a deterministically derived seed
(`sha256(perturbation_seed|class|base|branch_point|type|k)`), so any member is reproducible in
isolation.

| Type | Params | Implementation |
|---|---|---|
| `gaussian_noise` | `std` | Adds `std * randn_like(latent)` |
| `gaussian_blur` | `kernel_size`, `sigma` | Depthwise Gaussian conv (`F.conv2d`, per-channel) |
| `nxn_mask` | `size`, `fill_value` | Zeroes (or fills) a random contiguous `n×n` spatial block |
| `scattered_mask` | `fraction`, `fill_value` | Fills a random `fraction` of spatial positions independently (Bernoulli mask) |
| `mock` | — | Identity; used only for `--dry-run` |

## 3. Metrics

Computed per ensemble two ways: mean±std over all pairwise
combinations among the `K` members, and mean±std of each member vs. the base's own unperturbed
continuation.

| Metric | Computation | Third-party model |
|---|---|---|
| `latent_mse` | MSE directly on latent tensors | none |
| `lpips` | Learned perceptual distance on decoded images | `lpips.LPIPS(net="alex")` (AlexNet-based) |
| `clip_cosine_distance` | `1 - cosine_similarity` of CLIP image embeddings (images resized to 224, bicubic) | `open_clip` `ViT-B-32`, `pretrained="openai"` |
| `dino_cosine_distance` | `1 - cosine_similarity` of DINO embeddings (images resized to 224, bicubic) | `timm`, `vit_small_patch14_dinov2.lvd142m` |

## 4. Hyperparameters

**Model / sampling**
- `model`: `DiT-XL/2`
- `image_size`: `256`
- `vae`: `mse`
- `num_classes`: `1000`
- `cfg_scale`: `4.0`
- `num_sampling_steps`: `250`

**Trajectory / branching**
- `classes`: 45 ImageNet class ids (`[63, 38, 34, ..., 90]`)
- `M`: `8` (base trajectories per class)
- `base_seed`: `0` (base `b` uses seed `base_seed + b`)
- `branch_points`: `[25, 75, 125, 150, 175, 225]` (respaced step indices)

**Perturbations**
- `K`: `16` (ensemble size per class/base/branch/type)
- `perturbation_seed`: `1000`
- `perturbations`:
  - `gaussian_noise`: `std=0.1`
  - `gaussian_blur`: `kernel_size=5, sigma=1.0`
  - `nxn_mask`: `size=8, fill_value=0.0`
  - `scattered_mask`: `fraction=0.05, fill_value=0.0`
